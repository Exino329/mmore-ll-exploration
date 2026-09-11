"""Indexing cost: standard RAG vs graph RAG.

The two are not alternatives here. ``mmore graph-index`` reads its passages back out of
the Milvus collection, so the graph strategy pays for the standard index *and* the graph
on top:

    standard = index
    graph    = index + graph-index

Both numbers are reported, along with the increment, because the increment is the part a
deployment actually chooses to pay.
"""

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .config import BenchConfig, write_stage_configs
from .timing import (
    StageTiming,
    mmore_command,
    path_size_mb,
    reported_seconds,
    run_stage,
)

logger = logging.getLogger(__name__)


def _reset_artifacts(config: BenchConfig, graph_only: bool = False) -> None:
    """Drop the previous run's index. Only ever touches this run's own directory."""
    milvus = Path(config.milvus_uri)
    graph = Path(config.graph_artifacts_dir)
    targets = (graph,) if graph_only else (milvus, Path(f"{milvus}.lock"), graph)
    for path in targets:
        if path.is_dir():
            logger.info(f"Removing {path}")
            shutil.rmtree(path)
        elif path.exists():
            logger.info(f"Removing {path}")
            path.unlink()


def _collection_stats(config: BenchConfig) -> Dict[str, Any]:
    from pymilvus import MilvusClient

    client = MilvusClient(uri=config.milvus_uri, db_name="bench")
    try:
        stats = client.get_collection_stats(config.index.collection_name)
        return {"row_count": int(stats.get("row_count", 0))}
    finally:
        client.close()


def _graph_stats(config: BenchConfig) -> Dict[str, Any]:
    from mmore.rag.graph.store import artifacts_path

    directory = Path(
        artifacts_path(config.graph_artifacts_dir, config.index.collection_name)
    )
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "num_passages": manifest.get("num_passages"),
        "num_entities": manifest.get("num_entities"),
        "num_sentences": manifest.get("num_sentences"),
        "embedding_dim": manifest.get("embedding_dim"),
        "artifact_breakdown_mb": {
            item.name: round(path_size_mb(item), 3)
            for item in sorted(directory.iterdir())
            if item.is_file()
        },
    }


def run(
    config: BenchConfig,
    keep_existing: bool = False,
    graph_only: bool = False,
    tag: Optional[str] = None,
) -> Dict[str, Any]:
    """Time `mmore index`, then `mmore graph-index`.

    ``graph_only`` reuses the Milvus collection already built and only (re)builds the
    graph, which is what comparing index-time graph settings needs — a different NER
    model, say. Point ``graph.artifacts_name`` at a fresh subdirectory so the two graphs
    coexist and can be evaluated against the same collection.
    """
    if not config.corpus_path.exists():
        raise FileNotFoundError(
            f"{config.corpus_path} not found. Run the `prepare` stage first."
        )

    paths = write_stage_configs(config)
    suffix = f"_{tag}" if tag else ""

    if graph_only:
        if not Path(config.milvus_uri).exists():
            raise FileNotFoundError(
                f"{config.milvus_uri} not found; --graph-only needs a collection that "
                "`index` already built."
            )
        _reset_artifacts(config, graph_only=True)
    elif keep_existing and Path(config.milvus_uri).exists():
        # `mmore index` would insert the corpus a second time: chunk ids come from
        # `str(hash(text))`, which Python salts per process, so the rows would not
        # collide with the existing ones — they would silently duplicate them.
        raise RuntimeError(
            f"{config.milvus_uri} already holds a collection. Re-running `index` over it "
            "would duplicate every document. Use --graph-only to rebuild just the graph, "
            "or drop --keep-existing to start clean."
        )
    else:
        _reset_artifacts(config)

    logs = config.root / "logs"
    documents = sum(1 for _ in open(config.corpus_path, encoding="utf-8"))

    index_timing = None
    if not graph_only:
        index_timing = run_stage(
            "index",
            mmore_command("index", "-c", str(paths["index"])),
            logs / "index.log",
        )
        _require_success(index_timing)
    milvus_mb = path_size_mb(Path(config.milvus_uri))

    graph_timing = run_stage(
        "graph-index",
        mmore_command("graph-index", "-c", str(paths["graph_index"])),
        logs / f"graph_index{suffix}.log",
    )
    _require_success(graph_timing)
    graph_mb = path_size_mb(Path(config.graph_artifacts_dir))

    graph_compute = reported_seconds(Path(graph_timing.log_path))
    graph_timing.extra = {
        "documents": documents,
        "compute_seconds": graph_compute,
        "docs_per_second": documents / graph_timing.wall_seconds,
        "artifact_mb": round(graph_mb, 3),
        "spacy_model": config.graph.spacy_model,
        **_graph_stats(config),
    }

    if index_timing is None:
        # Only the graph was rebuilt: there is no standard-RAG column to compare against.
        summary = {
            "documents": documents,
            "graph_only": True,
            "stages": [asdict(graph_timing)],
        }
        path = config.results_path(f"index_graph_only{suffix}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(
            f"graph-index ({config.graph.spacy_model}) "
            f"{graph_timing.wall_seconds:.1f}s → {path}"
        )
        return summary

    index_compute = reported_seconds(Path(index_timing.log_path))
    index_timing.extra = {
        "documents": documents,
        "compute_seconds": index_compute,
        "docs_per_second": documents / index_timing.wall_seconds,
        "artifact_mb": round(milvus_mb, 3),
        **_collection_stats(config),
    }

    total = index_timing.wall_seconds + graph_timing.wall_seconds
    summary = {
        "documents": documents,
        "stages": [asdict(index_timing), asdict(graph_timing)],
        "standard": {
            "wall_seconds": index_timing.wall_seconds,
            "compute_seconds": index_compute,
            "cpu_seconds": index_timing.cpu_seconds,
            "peak_rss_mb": index_timing.peak_rss_mb,
            "disk_mb": round(milvus_mb, 3),
        },
        "graph": {
            "wall_seconds": total,
            "compute_seconds": (index_compute + graph_compute)
            if index_compute is not None and graph_compute is not None
            else None,
            "cpu_seconds": index_timing.cpu_seconds + graph_timing.cpu_seconds,
            "peak_rss_mb": max(index_timing.peak_rss_mb, graph_timing.peak_rss_mb),
            "disk_mb": round(milvus_mb + graph_mb, 3),
            "increment_wall_seconds": graph_timing.wall_seconds,
            "increment_compute_seconds": graph_compute,
            "increment_disk_mb": round(graph_mb, 3),
        },
        "overhead_ratio": total / index_timing.wall_seconds
        if index_timing.wall_seconds
        else None,
    }

    path = config.results_path("index.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        f"index {index_timing.wall_seconds:.1f}s + graph-index "
        f"{graph_timing.wall_seconds:.1f}s = {total:.1f}s "
        f"({summary['overhead_ratio']:.2f}x standard) → {path}"
    )
    return summary


def _require_success(timing: StageTiming) -> None:
    if timing.returncode != 0:
        raise RuntimeError(
            f"Stage '{timing.name}' exited with {timing.returncode}. "
            f"See {timing.log_path}."
        )
