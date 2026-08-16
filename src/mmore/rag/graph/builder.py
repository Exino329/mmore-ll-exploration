"""Index-time construction of the Tri-Graph.

Port of ``LinearRAG.index()`` and its helpers (``extract_nodes_and_edges``,
``add_entity_to_passage_edges``, ``add_adjacent_passage_edges``, ``augment_graph``).

Deviations from the reference implementation, all deliberate:

* **Passage identity** comes from mmore's chunk id (the Milvus primary key), so the graph
  points straight at Milvus rows instead of duplicating passage text and vectors.
* **Adjacency edges** are chained per document using the ``"{document_id}+{chunk_idx}"``
  id produced by the chunker, rather than parsing a ``"<int>:"`` prefix that upstream
  prepends to the passage text (which leaks into both the embedding and the LLM prompt,
  and chains unrelated documents into one global sequence).
* **Entity canonicalization** merges surface forms differing only by case/whitespace.
* **Mention counts** are stored in the passage-entity matrix at index time; upstream
  recomputes them with ``str.count`` over the whole corpus on every query.
"""

import gzip
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from langchain_core.embeddings import Embeddings

from ...type import MultimodalSample
from ..model.dense.base import DenseModel, DenseModelConfig
from .config import GraphBuildConfig
from .ner import (
    DocumentEntities,
    SpacyEntityExtractor,
    extractor_from_build_config,
    normalize_entity,
)
from .store import TriGraph, TriGraphManifest, artifacts_path, load_manifest

logger = logging.getLogger(__name__)

_NER_CACHE = "ner.jsonl.gz"


class TriGraphBuilder:
    """Builds (or incrementally refreshes) the Tri-Graph of one collection."""

    def __init__(
        self,
        config: GraphBuildConfig,
        dense_model: Embeddings,
        dense_model_config: DenseModelConfig,
        extractor: Optional[SpacyEntityExtractor] = None,
    ):
        self.config = config
        self.dense_model = dense_model
        self.dense_model_config = dense_model_config
        self._extractor = extractor

    @classmethod
    def from_config(
        cls,
        config: GraphBuildConfig,
        dense_model_config: DenseModelConfig,
        device: Optional[str] = None,
    ) -> "TriGraphBuilder":
        return cls(
            config=config,
            dense_model=DenseModel.from_config(dense_model_config, device=device),
            dense_model_config=dense_model_config,
        )

    @property
    def extractor(self) -> SpacyEntityExtractor:
        # Loading a spaCy pipeline is expensive and pointless when every chunk is cached.
        if self._extractor is None:
            self._extractor = extractor_from_build_config(self.config)
        return self._extractor

    # --- entry point -------------------------------------------------------------------

    def build(
        self, samples: Sequence[MultimodalSample], collection_name: str
    ) -> TriGraph:
        directory = artifacts_path(self.config.artifacts_dir, collection_name)
        samples = _deduplicate(samples)

        ner_cache = {} if self.config.rebuild else _load_ner_cache(directory)
        missing = [s for s in samples if s.id not in ner_cache]
        logger.info(
            f"Entity extraction: {len(samples) - len(missing)} chunks cached, "
            f"{len(missing)} to process"
        )
        if missing:
            for sample, extracted in zip(
                missing, self.extractor.extract_documents([s.text for s in missing])
            ):
                ner_cache[sample.id] = extracted
            os.makedirs(directory, exist_ok=True)
            _save_ner_cache(directory, ner_cache)
            # getattr: `extractor` is duck-typed — anything with extract_documents does.
            stats = getattr(self.extractor, "extraction_stats", dict)()
            if stats:
                logger.info(
                    "Entity extraction stats: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(stats.items()))
                )

        registry = _EntityRegistry(normalize=self.config.normalize_entities)
        for sample in samples:
            registry.add(sample.id, ner_cache[sample.id])

        previous = self._load_previous(directory)
        entity_embeddings = self._embed(
            registry.entity_surfaces,
            cache=_embedding_cache(previous.entity_surfaces, previous.entity_embeddings)
            if previous
            else None,
            what="entities",
        )
        sentence_embeddings = self._embed(
            registry.sentence_texts,
            cache=_embedding_cache(
                previous.sentence_texts, previous.sentence_embeddings
            )
            if previous
            else None,
            what="sentences",
        )

        entity_to_sentence = registry.entity_to_sentence_matrix()
        passage_to_entity = registry.passage_to_entity_matrix(
            {s.id: s.text for s in samples}, normalize=self.config.normalize_entities
        )
        graph = _build_graph(registry, passage_to_entity, samples)

        tri_graph = TriGraph(
            manifest=TriGraphManifest(
                collection_name=collection_name,
                dense_model=asdict(self.dense_model_config),
                spacy_model=self.config.spacy_model,
                normalize_entities=self.config.normalize_entities,
                ner_backend=self.config.ner_backend,
                ner_options=asdict(self.config.medspacy)
                if self.config.ner_backend == "medspacy"
                else {},
                embedding_dim=int(entity_embeddings.shape[1])
                if entity_embeddings.size
                else 0,
                num_passages=len(registry.passage_ids),
                num_entities=len(registry.entity_keys),
                num_sentences=len(registry.sentence_texts),
            ),
            graph=graph,
            passage_ids=registry.passage_ids,
            entity_keys=registry.entity_keys,
            entity_surfaces=registry.entity_surfaces,
            entity_embeddings=entity_embeddings,
            sentence_texts=registry.sentence_texts,
            sentence_passage_idx=np.asarray(
                registry.sentence_passage_idx, dtype=np.int32
            ),
            sentence_embeddings=sentence_embeddings,
            entity_to_sentence=entity_to_sentence,
            passage_to_entity=passage_to_entity,
        )
        tri_graph.save(directory)
        return tri_graph

    # --- helpers -----------------------------------------------------------------------

    def _load_previous(self, directory: str) -> Optional[TriGraph]:
        """Previous artifacts, used only to avoid re-embedding unchanged text."""
        if self.config.rebuild or load_manifest(directory) is None:
            return None
        try:
            return TriGraph.load(directory, mmap_sentences=True)
        except Exception as e:
            logger.warning(f"Could not reuse existing embeddings in {directory}: {e}")
            return None

    def _embed(
        self,
        texts: List[str],
        cache: Optional[Dict[str, np.ndarray]],
        what: str,
    ) -> np.ndarray:
        if not texts:
            dim = self.embedding_dim()
            return np.zeros((0, dim), dtype=np.float32)

        cache = cache or {}
        todo = [t for t in texts if t not in cache]
        logger.info(
            f"Embedding {what}: {len(texts) - len(todo)} reused, {len(todo)} to encode"
        )

        encoded: Dict[str, np.ndarray] = {}
        batch = self.config.batch_size
        for start in range(0, len(todo), batch):
            window = todo[start : start + batch]
            vectors = self.dense_model.embed_documents(window)
            for text, vector in zip(window, vectors):
                encoded[text] = np.asarray(vector, dtype=np.float32)
            if start and start % (batch * 20) == 0:
                logger.info(f"  ... {start}/{len(todo)} {what}")

        matrix = np.vstack(
            [np.asarray(cache.get(t, encoded.get(t)), dtype=np.float32) for t in texts]
        )
        return _l2_normalize(matrix)

    def embedding_dim(self) -> int:
        return len(self.dense_model.embed_query("dimension probe"))


# --- entity / sentence bookkeeping ---------------------------------------------------


class _EntityRegistry:
    """Accumulates the node sets and the entity-sentence incidence of the corpus."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

        self.passage_ids: List[str] = []
        self._passage_idx: Dict[str, int] = {}

        self.entity_keys: List[str] = []
        self.entity_surfaces: List[str] = []
        self._entity_idx: Dict[str, int] = {}

        self.sentence_texts: List[str] = []
        self.sentence_passage_idx: List[int] = []
        self._sentence_idx: Dict[str, int] = {}

        self._sentence_entities: List[set] = []
        self._passage_entities: Dict[int, set] = defaultdict(set)

    def key_of(self, surface: str) -> str:
        return normalize_entity(surface) if self.normalize else surface.strip()

    def add(self, passage_id: str, extracted: DocumentEntities) -> None:
        p_idx = self._passage_idx.get(passage_id)
        if p_idx is None:
            p_idx = len(self.passage_ids)
            self._passage_idx[passage_id] = p_idx
            self.passage_ids.append(passage_id)

        for surface in extracted.entities:
            self._passage_entities[p_idx].add(self._entity(surface))

        for sentence, surfaces in extracted.sentences:
            s_idx = self._sentence_idx.get(sentence)
            if s_idx is None:
                s_idx = len(self.sentence_texts)
                self._sentence_idx[sentence] = s_idx
                self.sentence_texts.append(sentence)
                # A sentence repeated verbatim across chunks collapses into one node, as
                # upstream; we remember the first chunk that produced it for provenance.
                self.sentence_passage_idx.append(p_idx)
                self._sentence_entities.append(set())
            for surface in surfaces:
                self._sentence_entities[s_idx].add(self._entity(surface))

    def _entity(self, surface: str) -> int:
        key = self.key_of(surface)
        idx = self._entity_idx.get(key)
        if idx is None:
            idx = len(self.entity_keys)
            self._entity_idx[key] = idx
            self.entity_keys.append(key)
            self.entity_surfaces.append(surface.strip())
        return idx

    def entity_to_sentence_matrix(self):
        from scipy import sparse

        rows: List[int] = []
        cols: List[int] = []
        for s_idx, entities in enumerate(self._sentence_entities):
            for e_idx in entities:
                rows.append(e_idx)
                cols.append(s_idx)
        shape = (len(self.entity_keys), len(self.sentence_texts))
        return sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=shape
        )

    def passage_to_entity_matrix(self, texts: Dict[str, str], normalize: bool):
        """Mention counts, the weights behind both the graph edges and the entity bonus."""
        from scipy import sparse

        rows: List[int] = []
        cols: List[int] = []
        counts: List[float] = []
        for p_idx, passage_id in enumerate(self.passage_ids):
            text = texts.get(passage_id, "")
            haystack = text.lower() if normalize else text
            for e_idx in self._passage_entities[p_idx]:
                needle = (
                    self.entity_keys[e_idx]
                    if normalize
                    else self.entity_surfaces[e_idx]
                )
                # NER found the entity in this chunk, so a count of 0 is only possible
                # when normalization changed the surface; keep it at 1 in that case.
                count = haystack.count(needle) or 1
                rows.append(p_idx)
                cols.append(e_idx)
                counts.append(float(count))
        shape = (len(self.passage_ids), len(self.entity_keys))
        return sparse.csr_matrix(
            (np.asarray(counts, dtype=np.float32), (rows, cols)), shape=shape
        )


def _build_graph(
    registry: _EntityRegistry, passage_to_entity, samples: Sequence[MultimodalSample]
):
    """Undirected igraph: passage vertices first, then entity vertices."""
    import igraph as ig

    num_passages = len(registry.passage_ids)
    num_entities = len(registry.entity_keys)

    graph = ig.Graph(n=num_passages + num_entities, directed=False)
    graph.vs["name"] = registry.passage_ids + registry.entity_keys
    graph.vs["kind"] = ["passage"] * num_passages + ["entity"] * num_entities

    edges: List[Tuple[int, int]] = []
    weights: List[float] = []

    # Passage <-> entity, weighted by the share of the passage's mentions.
    coo = passage_to_entity.tocoo()
    totals = np.asarray(passage_to_entity.sum(axis=1)).ravel()
    for p_idx, e_idx, count in zip(coo.row, coo.col, coo.data):
        total = totals[p_idx]
        if total <= 0:
            continue
        edges.append((int(p_idx), num_passages + int(e_idx)))
        weights.append(float(count) / float(total))

    # Passage <-> passage, chaining consecutive chunks of the same document.
    for a, b in _adjacent_passage_pairs(samples, registry.passage_ids):
        edges.append((a, b))
        weights.append(1.0)

    graph.add_edges(edges)
    graph.es["weight"] = weights
    return graph


def _adjacent_passage_pairs(
    samples: Sequence[MultimodalSample], passage_ids: List[str]
) -> Iterable[Tuple[int, int]]:
    index_of = {pid: i for i, pid in enumerate(passage_ids)}
    per_document: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for position, sample in enumerate(samples):
        document_id = sample.document_id or sample.id
        chunk_idx = _chunk_index(sample.id)
        # Fall back to input order for documents whose ids carry no chunk index.
        order = chunk_idx if chunk_idx is not None else position
        per_document[document_id].append((order, index_of[sample.id]))

    for chunks in per_document.values():
        chunks.sort(key=lambda pair: pair[0])
        for (_, left), (_, right) in zip(chunks, chunks[1:]):
            if left != right:
                yield left, right


def _chunk_index(sample_id: str) -> Optional[int]:
    """Chunk position encoded by the chunker as ``"{document_id}+{chunk_idx}"``."""
    if "+" not in sample_id:
        return None
    suffix = sample_id.rsplit("+", 1)[1]
    return int(suffix) if suffix.isdigit() else None


# --- small utilities ------------------------------------------------------------------


def _deduplicate(samples: Sequence[MultimodalSample]) -> List[MultimodalSample]:
    seen = set()
    unique = []
    for sample in samples:
        if sample.id in seen:
            continue
        seen.add(sample.id)
        unique.append(sample)
    if len(unique) != len(samples):
        logger.warning(
            f"Dropped {len(samples) - len(unique)} chunks with duplicate ids"
        )
    return unique


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (matrix / norms).astype(np.float32)


def _embedding_cache(texts: List[str], embeddings: np.ndarray) -> Dict[str, np.ndarray]:
    return {text: embeddings[i] for i, text in enumerate(texts)}


def _load_ner_cache(directory: str) -> Dict[str, DocumentEntities]:
    path = os.path.join(directory, _NER_CACHE)
    if not os.path.exists(path):
        return {}
    cache: Dict[str, DocumentEntities] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cache[record["passage_id"]] = DocumentEntities(
                entities=record["entities"],
                sentences=[(text, ents) for text, ents in record["sentences"]],
            )
    return cache


def _save_ner_cache(directory: str, cache: Dict[str, DocumentEntities]) -> None:
    path = os.path.join(directory, _NER_CACHE)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for passage_id, extracted in cache.items():
            f.write(
                json.dumps(
                    {
                        "passage_id": passage_id,
                        "entities": extracted.entities,
                        "sentences": [list(pair) for pair in extracted.sentences],
                    }
                )
                + "\n"
            )
