"""Retrieval accuracy: does the right passage come back, and how fast?

No LLM is involved, so this layer is deterministic and free to re-run. It is the part of
the comparison that isolates the retrieval strategy: same collection, same questions, same
k, only ``retriever.type`` differs.
"""

import json
import logging
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from mmore.rag.factory import load_retriever
from mmore.utils import load_config

from . import strategies
from .collection import key_by_chunk_id
from .config import BenchConfig, write_stage_configs
from .qa import Question, read_questions

logger = logging.getLogger(__name__)

STRATEGIES = (
    "hybrid",
    "graph",
    "lexical",
    "dense",
    "sparse",
    "graph+dense",
    "graph+lexical",
    "graph+hybrid",
)
"""``lexical`` is BM25 with no model at all — the floor the others have to clear. It takes
no stage config: it reads the indexed passages straight out of the collection. ``dense``
and ``sparse`` are the hybrid retriever with one half of the ranker switched off, which
separates the two signals ``hybrid`` mixes. The ``a+b`` strategies fuse by reciprocal rank;
see :mod:`.strategies`."""

DEFAULT_STRATEGIES = ("hybrid", "graph", "lexical")
"""What a bare `retrieval` run scores, kept as it was so existing invocations are unchanged."""

_GROUP_METRIC = re.compile(r"^[a-z0-9]+_at_\d+$")


def _ranked_keys(docs, keys: Dict[str, str]) -> List[str]:
    """Retrieved passages as a deduplicated, rank-ordered list of gold keys."""
    ordered: List[str] = []
    for doc in docs:
        key = keys.get(doc.metadata.get("id", ""))
        if key and key not in ordered:
            ordered.append(key)
    return ordered


def _rank_of(golds: List[str], predicted: List[str]) -> Optional[int]:
    """Rank of the first gold passage. With several, the earliest one found."""
    ranks = [predicted.index(g) + 1 for g in golds if g in predicted]
    return min(ranks) if ranks else None


def _found(golds: List[str], predicted: List[str], k: int) -> bool:
    """Whether any gold of the group is in the top k."""
    return any(g in predicted[:k] for g in golds)


def _all_found(golds: List[str], predicted: List[str], k: int) -> bool:
    """Whether *every* gold is in the top k.

    ``recall_at_k`` is satisfied by the first gold, which on a multi-hop question is the
    easy one: HotpotQA answers need both supporting articles, and that is the figure its
    literature reports (HippoRAG's Recall@2/@5). Only emitted when a question has more
    than one gold, so single-gold tasks keep the results they already had."""
    return bool(golds) and all(g in predicted[:k] for g in golds)


def _percentile(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def _suffix(tag: Optional[str]) -> str:
    return f"_{tag}" if tag else ""


def summarize(
    records: List[Dict[str, Any]],
    strategy: str,
    ks: List[int],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate per-query records into the numbers the report reads."""
    ranks = [r["rank"] for r in records]
    latencies = [r["seconds"] for r in records]
    total = len(records) or 1
    return {
        "strategy": strategy,
        "tag": tag,
        "questions": len(records),
        "hit_at_1": sum(1 for r in ranks if r == 1) / total,
        **{
            f"recall_at_{k}": sum(1 for r in ranks if r is not None and r <= k) / total
            for k in ks
        },
        "mrr": sum(1.0 / r for r in ranks if r is not None) / total,
        # Multi-hop: which hop was missed matters more than the aggregate. On MedHop the
        # hop-2 passage holds the answer and never names the question's subject, so it is
        # the one dense retrieval cannot reach from the query wording.
        #
        # Scored over the questions that *define* the group, not over all of them. Every
        # MedHop question defines hop1 and hop2, so those figures are unchanged; HotpotQA
        # defines `bridge` on 811 questions of 1,000 and a hop split on 726, and dividing
        # those by 1,000 would cap them below 1 and read as a retrieval failure.
        **{
            key: sum(1 for r in records if r.get(key))
            / max(sum(1 for r in records if key in r), 1)
            for key in sorted({k for r in records for k in r if _GROUP_METRIC.match(k)})
        },
        "empty_results": sum(1 for r in records if r["n_docs"] == 0) / total,
        "latency_mean_s": float(np.mean(latencies)) if latencies else 0.0,
        "latency_median_s": float(np.median(latencies)) if latencies else 0.0,
        "latency_p95_s": _percentile(latencies, 95),
    }


def build_retriever(
    strategy: str, config: BenchConfig, paths: Dict[str, Path], max_k: int
) -> Any:
    """The retriever a strategy name stands for.

    Recursive, because a fusion is defined by the names of its parts rather than by a
    configuration of its own — ``graph+dense`` builds a ``graph`` and a ``dense`` and puts
    an RRF in front of them.
    """
    from mmore.rag.retriever import RetrieverConfig

    if strategy == "lexical":
        from . import lexical

        return lexical.build(config, k=max_k)

    if strategy in strategies.FUSIONS:
        names = strategies.FUSIONS[strategy]
        # `lexical` reads the passages through a MilvusClient of its own and closes it,
        # which tears down the connection alias every other client shares — so it has to be
        # built before any Milvus-backed part, never after. Query order is unaffected: the
        # parts are handed to the fusion in their declared order.
        built = {
            name: build_retriever(name, config, paths, max_k)
            for name in sorted(names, key=lambda n: n != "lexical")
        }
        parts = [(name, built[name]) for name in names]
        return strategies.ReciprocalRankFusion(
            parts,
            k=max_k,
            depth=config.retrieval.fusion_depth,
            constant=config.retrieval.fusion_constant,
        )

    # `dense` and `sparse` are the hybrid retriever with the ranker leaning all the way to
    # one side, so they share its stage config and differ only by the pinned search_type.
    config_key = "retriever_graph" if strategy == "graph" else "retriever_hybrid"
    retriever = load_retriever(load_config(str(paths[config_key]), RetrieverConfig))
    if strategy in strategies.SEARCH_TYPES:
        return strategies.PinnedSearchType(retriever, strategies.SEARCH_TYPES[strategy])
    return retriever


def evaluate_strategy(
    strategy: str,
    config: BenchConfig,
    questions: List[Question],
    keys: Dict[str, str],
    paths: Dict[str, Path],
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    max_k = max(config.retrieval.ks)
    retriever = build_retriever(strategy, config, paths, max_k)

    def retrieve(question: str):
        return retriever.invoke(
            {"input": question, "collection_name": config.index.collection_name}
        )

    for question in questions[: config.retrieval.warmup_queries]:
        retrieve(question.query)

    records: List[Dict[str, Any]] = []
    for question in tqdm(questions, desc=f"Retrieving [{strategy}]", unit="q"):
        started = time.perf_counter()
        # `query`, not `question`: MedHop's candidate list would otherwise put the answer
        # accession in the retrieval query. See Question.retrieval_query.
        docs = retrieve(question.query)
        elapsed = time.perf_counter() - started

        predicted = _ranked_keys(docs, keys)
        record_groups = {
            f"{group}_at_{k}": _found(members, predicted, k)
            for group, members in question.gold_groups.items()
            for k in config.retrieval.ks
        }
        if len(question.gold_keys) > 1:
            record_groups.update(
                {
                    f"all_at_{k}": _all_found(question.gold_keys, predicted, k)
                    for k in config.retrieval.ks
                }
            )
        records.append(
            {
                "query_id": question.query_id,
                "gold_keys": question.gold_keys,
                "predicted_keys": predicted[:max_k],
                "rank": _rank_of(question.gold_keys, predicted[:max_k]),
                **record_groups,
                "n_docs": len(docs),
                "seconds": elapsed,
                # A graph query that finds no entity to seed with falls back to hybrid
                # retrieval; those results are not graph results and are counted apart.
                "fell_back": strategy == "graph"
                and bool(docs)
                and "graph_score" not in docs[0].metadata,
            }
        )

    summary = summarize(records, strategy, config.retrieval.ks, tag)
    if strategy == "graph":
        # Only observable on the bare graph: once fused, a top result with no `graph_score`
        # is the normal contribution of the other part, not evidence of a fallback.
        summary["hybrid_fallback_rate"] = sum(
            1 for r in records if r["fell_back"]
        ) / max(len(records), 1)
    if strategies.uses_graph(strategy):
        # The knobs a sweep varies, recorded next to the numbers they produced.
        summary["graph_settings"] = {
            key: getattr(config.graph, key)
            for key in (
                "passage_ratio",
                "passage_node_weight",
                "iteration_threshold",
                "max_iterations",
                "top_k_sentence",
                "damping",
                "seed_min_similarity",
                "dense_candidates",
                "activation_merge",
                "spacy_model",
                "ner_backend",
                "artifacts_name",
            )
        }
        if config.graph.ner_backend == "medspacy":
            summary["graph_settings"]["medspacy"] = asdict(config.graph.medspacy)

    path = config.results_path(f"retrieval_{strategy}{_suffix(tag)}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "queries": records}, indent=2), encoding="utf-8"
    )
    logger.info(
        f"[{strategy}{_suffix(tag)}] hit@1={summary['hit_at_1']:.3f} "
        f"recall@{max_k}={summary[f'recall_at_{max_k}']:.3f} "
        f"MRR={summary['mrr']:.3f} median={summary['latency_median_s'] * 1000:.0f}ms"
    )
    return summary


def run(
    config: BenchConfig,
    selected: List[str] = list(DEFAULT_STRATEGIES),
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    if not config.qa_path.exists():
        raise FileNotFoundError(
            f"{config.qa_path} not found. Run the `prepare` stage first."
        )

    paths = write_stage_configs(config)
    questions = read_questions(config.qa_path)
    keys = key_by_chunk_id(config)

    summaries = {
        strategy: evaluate_strategy(strategy, config, questions, keys, paths, tag)
        for strategy in selected
    }

    path = config.results_path(f"retrieval{_suffix(tag)}.json")
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
