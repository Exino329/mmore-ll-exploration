"""Turn a run directory into one readable comparison."""

import json
import platform
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import BenchConfig

_GROUP_METRIC = re.compile(r"^(?!hit_at|recall_at)[a-z0-9]+_at_\d+$")
"""Per-hop metrics written by ``retrieval_bench``, e.g. ``hop2_at_10``."""


def _read(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _delta(graph: float, hybrid: float, unit: str = "", digits: int = 3) -> str:
    difference = graph - hybrid
    return f"{difference:+.{digits}f}{unit}"


def _environment() -> Dict[str, str]:
    try:
        import psutil

        cores = str(psutil.cpu_count(logical=False) or psutil.cpu_count())
        memory = f"{psutil.virtual_memory().total / 1024**3:.1f} GB"
    except Exception:
        cores, memory = "?", "?"
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        gpu = "unknown"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "physical cores": cores,
        "memory": memory,
        "device": gpu,
    }


def _indexing_section(index: Dict[str, Any]) -> str:
    standard, graph = index["standard"], index["graph"]
    rows = [
        [
            "Wall clock",
            f"{standard['wall_seconds']:.1f} s",
            f"{graph['wall_seconds']:.1f} s",
            f"+{graph['increment_wall_seconds']:.1f} s "
            f"({index['overhead_ratio']:.2f}× total)",
        ],
        [
            "CPU time",
            f"{standard['cpu_seconds']:.1f} s",
            f"{graph['cpu_seconds']:.1f} s",
            _delta(graph["cpu_seconds"], standard["cpu_seconds"], " s", 1),
        ],
        [
            "Peak RSS",
            f"{standard['peak_rss_mb']:.0f} MB",
            f"{graph['peak_rss_mb']:.0f} MB",
            _delta(graph["peak_rss_mb"], standard["peak_rss_mb"], " MB", 0),
        ],
        [
            "Disk",
            f"{standard['disk_mb']:.1f} MB",
            f"{graph['disk_mb']:.1f} MB",
            f"+{graph['increment_disk_mb']:.1f} MB",
        ],
    ]

    # mmore's stages report their own duration with model loading subtracted. On a small
    # corpus that fixed cost dwarfs the work, so both numbers are shown.
    if standard.get("compute_seconds") and graph.get("compute_seconds"):
        rows.insert(
            1,
            [
                "Compute (excl. model loading)",
                f"{standard['compute_seconds']:.1f} s",
                f"{graph['compute_seconds']:.1f} s",
                f"+{graph['increment_compute_seconds']:.1f} s "
                f"({graph['compute_seconds'] / standard['compute_seconds']:.2f}× total)",
            ],
        )

    stages = {stage["name"]: stage for stage in index["stages"]}
    graph_extra = stages.get("graph-index", {}).get("extra", {})
    counts = ", ".join(
        f"{graph_extra[key]:,} {key.replace('num_', '')}"
        for key in ("num_passages", "num_entities", "num_sentences")
        if graph_extra.get(key) is not None
    )

    return "\n".join(
        [
            "## Indexing cost",
            "",
            f"Corpus: {index['documents']:,} documents.",
            "",
            _table(["", "Standard RAG", "Graph RAG", "Difference"], rows),
            "",
            "Graph RAG is *additive*: `mmore graph-index` reads its passages back out of "
            "the Milvus collection, so its column is `index` + `graph-index`, not an "
            "alternative to `index`. The difference column is what the graph costs on top.",
            "",
            f"Tri-Graph: {counts}." if counts else "",
        ]
    )


_STRATEGY_ORDER = ("lexical", "hybrid", "graph")
_STRATEGY_LABEL = {"lexical": "BM25", "hybrid": "Hybrid", "graph": "Graph"}


def _group_metrics(summaries: List[Dict[str, Any]], ks: List[int]) -> List[tuple]:
    """Per-hop metrics, when the task defines hops. ``hop2_at_k`` is the one that matters
    on MedHop: the passage holding the answer, which hop-1 retrieval cannot reach."""
    groups = sorted(
        {
            key.rsplit("_at_", 1)[0]
            for summary in summaries
            for key in summary
            if _GROUP_METRIC.match(key)
        }
    )
    return [
        (f"{group}_at_{k}", f"{group}@{k}", 3)
        for group in groups
        for k in ks
        if any(f"{group}_at_{k}" in s for s in summaries)
    ]


def _retrieval_section(retrieval: Dict[str, Any], ks: List[int]) -> str:
    present = [s for s in _STRATEGY_ORDER if retrieval.get(s)] + [
        s for s in retrieval if s not in _STRATEGY_ORDER and retrieval.get(s)
    ]
    if not present:
        return "## Retrieval accuracy\n\nNo strategy present."

    summaries = [retrieval[s] for s in present]
    metrics = [("hit_at_1", "hit@1", 3), ("mrr", "MRR", 3)]
    metrics += [(f"recall_at_{k}", f"recall@{k}", 3) for k in ks]
    metrics += _group_metrics(summaries, ks)
    metrics += [
        ("latency_median_s", "median latency (s)", 3),
        ("latency_p95_s", "p95 latency (s)", 3),
    ]

    hybrid, graph = retrieval.get("hybrid"), retrieval.get("graph")
    compare = bool(hybrid and graph)

    rows = []
    for key, label, digits in metrics:
        if not any(key in s for s in summaries):
            continue
        row = [label] + [f"{s[key]:.{digits}f}" if key in s else "—" for s in summaries]
        if compare:
            row.append(
                _delta(graph[key], hybrid[key], "", digits)
                if key in hybrid and key in graph
                else "—"
            )
        rows.append(row)

    header = [""] + [_STRATEGY_LABEL.get(s, s.title()) for s in present]
    if compare:
        header.append("Graph − Hybrid")

    notes = []
    fallback = (graph or {}).get("hybrid_fallback_rate")
    if fallback is not None:
        notes.append(
            f"\n{fallback:.1%} of graph queries found no entity to seed with and fell "
            "back to hybrid retrieval; those are counted in the graph column as they "
            "occurred."
        )
    if "lexical" in present:
        notes.append(
            "\nBM25 uses no model and no index of its own. Read it as the floor: a "
            "retriever that does not clear it is not being helped by its embeddings on "
            "this task."
        )

    return "\n".join(
        [
            "## Retrieval accuracy",
            "",
            f"{summaries[0]['questions']} questions, gold = the passage the question was "
            "written from.",
            "",
            _table(header, rows),
            *notes,
            "",
        ]
    )


def _answer_section(answers: Dict[str, Any]) -> str:
    hybrid, graph = answers.get("hybrid"), answers.get("graph")
    if not (hybrid and graph):
        available = ", ".join(answers) or "none"
        return f"## Answer accuracy\n\nOnly one strategy present ({available})."

    k = hybrid.get("k")
    scoring = hybrid.get("scoring", "judge")
    metrics = [
        (
            "accuracy",
            "accuracy (exact label)"
            if scoring == "exact_label"
            else "LLM-judge accuracy",
        ),
        ("contain_accuracy", "contain accuracy"),
        (f"gold_retrieved_at_{k}", f"gold passage in context (k={k})"),
        ("latency_mean_s", "mean latency (s)"),
    ]
    rows = [
        [
            label,
            f"{hybrid[key]:.3f}",
            f"{graph[key]:.3f}",
            _delta(graph[key], hybrid[key]),
        ]
        for key, label in metrics
        if key in hybrid and key in graph
    ]
    baseline = hybrid.get("majority_baseline")
    notes = []
    if baseline is not None:
        notes.append(
            f"\nAlways answering the majority class scores {baseline:.3f}. Read the "
            "accuracy row against that, not against zero."
        )
    unparsed = max(hybrid.get("unparsed_answers", 0), graph.get("unparsed_answers", 0))
    if unparsed:
        notes.append(
            f"{unparsed} answers contained none of the expected labels and count as wrong."
        )

    return "\n".join(
        [
            "## Answer accuracy",
            "",
            f"{hybrid['questions']} questions, same generator for both strategies.",
            "",
            _table(["", "Hybrid", "Graph", "Difference"], rows),
            *notes,
        ]
    )


def _describe_ner(config: BenchConfig) -> str:
    """The NER model, and what wraps it — a medspaCy run differs from its baseline only by
    the wrapper, so leaving it out would make two reports look identical."""
    if config.graph.ner_backend != "medspacy":
        return config.graph.spacy_model
    dropped = ", ".join(config.graph.medspacy.drop_asserted) or "nothing"
    return (
        f"{config.graph.spacy_model} + medspaCy "
        f"({', '.join(config.graph.medspacy.components)}; drops {dropped})"
    )


def build(config: BenchConfig) -> str:
    index = _read(config.results_path("index.json"))
    retrieval = _read(config.results_path("retrieval.json"))
    answers = _read(config.results_path("answers.json"))
    corpus = _read(config.results_path("corpus.json"))

    from .tasks import load_task

    setup = {
        **load_task(config.task).describe(config),
        "indexed documents": f"{corpus['documents']:,}" if corpus else "unknown",
        "dense model": config.index.dense_model,
        "sparse model": config.index.sparse_model,
        "entity extraction": _describe_ner(config),
        "reranker": config.retrieval.reranker_model_name or "disabled",
        "generator": config.answer.llm.llm_name,
        "answer scoring": config.answer.scoring,
        **_environment(),
    }

    sections = [
        "# Standard RAG vs LinearRAG graph retrieval",
        "",
        "## Setup",
        "",
        _table(["", ""], [[key, str(value)] for key, value in setup.items()]),
        "",
    ]
    sections += [
        _indexing_section(index) if index else "## Indexing cost\n\n_Not run._",
        "",
        _retrieval_section(retrieval, config.retrieval.ks)
        if retrieval
        else "## Retrieval accuracy\n\n_Not run._",
        "",
        _answer_section(answers) if answers else "## Answer accuracy\n\n_Not run._",
        "",
    ]
    return "\n".join(sections)


def run(config: BenchConfig) -> Path:
    markdown = build(config)
    path = config.root / "report.md"
    path.write_text(markdown, encoding="utf-8")

    summary = {
        "config": asdict(config),
        "index": _read(config.results_path("index.json")),
        "retrieval": _read(config.results_path("retrieval.json")),
        "answers": _read(config.results_path("answers.json")),
        "corpus": _read(config.results_path("corpus.json")),
        "environment": _environment(),
    }
    config.results_path("summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return path
