"""The two-stage LinearRAG retrieval algorithm.

Stage 1 — *local semantic bridging*: starting from the entities mentioned in the query,
repeatedly hop entity -> most question-relevant sentence mentioning it -> entities
co-mentioned in that sentence, decaying the activation multiplicatively. This is what
gives multi-hop reach without any relation extraction.

Stage 2 — *global importance aggregation*: turn the activated entities and a dense-
retrieval prior into a personalized-PageRank reset vector over the Tri-Graph and rank
passages by the resulting stationary mass.

Everything here is pure: no I/O, no LangChain, no Milvus. Ports of
``calculate_entity_scores``, ``calculate_passage_scores`` and ``run_ppr``.
"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .config import GraphRetrieverConfig
from .store import TriGraph

logger = logging.getLogger(__name__)


@dataclass
class ActivatedEntity:
    """An entity reached by semantic bridging."""

    index: int
    score: float
    tier: int
    """Hop at which the entity was reached; seeds are tier 1. Divides the entity's
    contribution to the passage bonus, so far-away entities count for less."""


def _merge(incumbent: ActivatedEntity, candidate: ActivatedEntity) -> ActivatedEntity:
    """Keep the strongest evidence for an entity reached more than once.

    An entity is routinely re-reached at a later hop with a weaker score. Upstream lets the
    later, weaker reading win, which demotes the very entities the question is about: a seed
    re-reached at hop 2 drops from its linking similarity to the propagated value *and* has
    its tier raised, so ``score_passages`` divides its contribution twice over. Measured on
    40 MedHop questions, 91% of seeds end up demoted that way.

    Max score and min tier are the monotone reading of the same evidence: reaching an entity
    again is corroboration, never a reason to trust it less.
    """
    return ActivatedEntity(
        index=incumbent.index,
        score=max(incumbent.score, candidate.score),
        tier=min(incumbent.tier, candidate.tier),
    )


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def link_seed_entities(
    tri_graph: TriGraph,
    query_entity_embeddings: np.ndarray,
    min_similarity: float,
) -> List[ActivatedEntity]:
    """Link each query entity to its nearest corpus entity, above a similarity floor.

    Upstream takes an unconditional argmax, so a query entity absent from the corpus still
    seeds the reset vector with whatever happens to be closest. ``min_similarity`` makes
    that failure mode explicit: no seed above the floor means no graph search at all.
    """
    if tri_graph.num_entities == 0 or query_entity_embeddings.size == 0:
        return []

    similarities = tri_graph.entity_embeddings @ query_entity_embeddings.T  # (E, Q)

    best: Dict[int, float] = {}
    for column in range(similarities.shape[1]):
        scores = similarities[:, column]
        entity_idx = int(np.argmax(scores))
        score = float(scores[entity_idx])
        if score < min_similarity:
            continue
        if score > best.get(entity_idx, -np.inf):
            best[entity_idx] = score

    return [
        ActivatedEntity(index=idx, score=score, tier=1) for idx, score in best.items()
    ]


def activate_entities(
    tri_graph: TriGraph,
    question_embedding: np.ndarray,
    seeds: Sequence[ActivatedEntity],
    config: GraphRetrieverConfig,
) -> Tuple[np.ndarray, Dict[int, ActivatedEntity]]:
    """Semantic bridging. Returns ``(entity_weights, activated_entities)``."""
    entity_weights = np.zeros(tri_graph.num_entities, dtype=np.float64)

    activated: Dict[int, ActivatedEntity] = {}
    for seed in seeds:
        activated[seed.index] = seed
        entity_weights[seed.index] = seed.score

    entity_to_sentence = tri_graph.entity_to_sentence
    sentence_to_entity = tri_graph.sentence_to_entity

    # Sentences are consumed globally: once a sentence has bridged one pair of entities it
    # cannot be reused, which keeps the frontier from cycling between co-occurring names.
    used_sentences: set = set()
    frontier = dict(activated)
    iteration = 1

    while frontier and iteration < config.max_iterations:
        next_frontier: Dict[int, ActivatedEntity] = {}

        for entity in frontier.values():
            if entity.score < config.iteration_threshold:
                continue

            row = slice(
                entity_to_sentence.indptr[entity.index],
                entity_to_sentence.indptr[entity.index + 1],
            )
            candidates = [
                int(s)
                for s in entity_to_sentence.indices[row]
                if s not in used_sentences
            ]
            if not candidates:
                continue

            sentence_ids = np.asarray(candidates, dtype=np.int64)
            similarities = np.asarray(
                tri_graph.sentence_embeddings[sentence_ids] @ question_embedding
            ).ravel()
            top = np.argsort(similarities)[::-1][: config.top_k_sentence]

            for position in top:
                sentence_id = int(sentence_ids[position])
                used_sentences.add(sentence_id)
                propagated = entity.score * float(similarities[position])
                if propagated < config.iteration_threshold:
                    continue

                neighbours = sentence_to_entity.indices[
                    sentence_to_entity.indptr[sentence_id] : sentence_to_entity.indptr[
                        sentence_id + 1
                    ]
                ]
                for neighbour in neighbours:
                    neighbour = int(neighbour)
                    entity_weights[neighbour] += propagated
                    reached = ActivatedEntity(
                        index=neighbour, score=propagated, tier=iteration + 1
                    )
                    incumbent = next_frontier.get(neighbour)
                    next_frontier[neighbour] = (
                        reached
                        if incumbent is None or config.activation_merge == "overwrite"
                        else _merge(incumbent, reached)
                    )

        if config.activation_merge == "overwrite":
            activated.update(next_frontier)
        else:
            for index, entity in next_frontier.items():
                incumbent = activated.get(index)
                activated[index] = (
                    entity if incumbent is None else _merge(incumbent, entity)
                )

        # The frontier keeps this hop's freshly propagated scores even under `best`: the
        # merge fixes the bookkeeping `score_passages` reads, not how far the walk travels.
        # Expanding a re-reached seed at its original score would restart the cascade from
        # it at every hop.
        frontier = next_frontier
        iteration += 1

    return entity_weights, activated


def score_passages(
    tri_graph: TriGraph,
    activated: Mapping[int, ActivatedEntity],
    dense_scores: Mapping[int, float],
    config: GraphRetrieverConfig,
) -> np.ndarray:
    """Passage prior: dense relevance plus a mention bonus from the activated entities.

    Upstream evaluates this for every passage in the corpus and recomputes mention counts
    with ``str.count`` on each query. Here the candidate pool is bounded (dense top-N plus
    every passage incident to an activated entity) and the counts come straight from the
    passage-entity matrix built at index time. Passages outside the pool keep a reset
    weight of 0; they can still accumulate PageRank mass through their entity edges.
    """
    passage_weights = np.zeros(tri_graph.num_passages, dtype=np.float64)

    dense_ids = np.asarray(sorted(dense_scores), dtype=np.int64)
    normalized_dense: Dict[int, float] = {}
    if dense_ids.size:
        values = min_max_normalize(
            np.asarray([dense_scores[int(i)] for i in dense_ids], dtype=np.float64)
        )
        normalized_dense = {int(i): float(v) for i, v in zip(dense_ids, values)}

    bonus: Dict[int, float] = {}
    entity_to_passage = tri_graph.entity_to_passage
    for entity in activated.values():
        start = entity_to_passage.indptr[entity.index]
        end = entity_to_passage.indptr[entity.index + 1]
        passages = entity_to_passage.indices[start:end]
        counts = entity_to_passage.data[start:end]
        denominator = float(max(entity.tier, 1))
        for passage_idx, count in zip(passages, counts):
            bonus[int(passage_idx)] = bonus.get(int(passage_idx), 0.0) + (
                entity.score * math.log(1.0 + float(count)) / denominator
            )

    for passage_idx in set(normalized_dense) | set(bonus):
        score = config.passage_ratio * normalized_dense.get(
            passage_idx, 0.0
        ) + math.log(1.0 + bonus.get(passage_idx, 0.0))
        passage_weights[passage_idx] = score * config.passage_node_weight

    return passage_weights


def run_ppr(
    tri_graph: TriGraph,
    entity_weights: np.ndarray,
    passage_weights: np.ndarray,
    damping: float,
) -> np.ndarray:
    """Personalized PageRank over the Tri-Graph. Returns one score per passage row."""
    num_passages = tri_graph.num_passages
    reset = np.zeros(num_passages + tri_graph.num_entities, dtype=np.float64)
    reset[:num_passages] = passage_weights
    reset[num_passages:] = entity_weights
    reset = np.where(np.isnan(reset) | (reset < 0), 0.0, reset)

    if reset.sum() <= 0:
        # igraph rejects an all-zero personalization vector; nothing was activated.
        return np.zeros(num_passages, dtype=np.float64)

    scores = tri_graph.graph.personalized_pagerank(
        damping=damping,
        directed=False,
        weights="weight" if tri_graph.graph.ecount() else None,
        reset=reset.tolist(),
        implementation="prpack",
    )
    return np.asarray(scores, dtype=np.float64)[:num_passages]
