import argparse
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Union, cast

from dotenv import load_dotenv

from mmore.index.indexer import IndexerConfig, get_model_from_index
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.rag.graph.builder import TriGraphBuilder
from mmore.rag.graph.config import GraphBuildConfig
from mmore.rag.model.dense.base import DenseModelConfig
from mmore.type import MultimodalSample
from mmore.utils import load_config
from mmore.ux import (
    model_loading_seconds,
    quiet_noisy_libs,
    setup_logging,
    step_intro,
    step_summary,
)

GRAPH_INDEX_NAME = "Graph Index"
GRAPH_INDEX_EMOJI = "🕸️"
logger = setup_logging(GRAPH_INDEX_NAME, GRAPH_INDEX_EMOJI)

load_dotenv()

_PAGE_SIZE = 1000


@dataclass
class GraphIndexConfig:
    indexer: IndexerConfig
    collection_name: str
    graph: GraphBuildConfig = field(default_factory=GraphBuildConfig)


@profile_function()
def graph_index(
    config_file: Union[GraphIndexConfig, str],
    collection_name: Optional[str] = None,
):
    """Build the LinearRAG Tri-Graph over an already-indexed collection.

    The passages come from the Milvus collection rather than from the JSONL `mmore index`
    consumed: `MultimodalSample.id` defaults to `str(hash(text))`, which Python salts per
    process and never serializes, so ids read back from a JSONL would not match the
    primary keys the graph has to point at.
    """
    from pymilvus import MilvusClient

    quiet_noisy_libs()
    config: GraphIndexConfig = load_config(config_file, GraphIndexConfig)
    if collection_name is None:
        collection_name = config.collection_name

    client = MilvusClient(uri=config.indexer.db.uri, db_name=config.indexer.db.name)
    if not client.has_collection(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' does not exist in "
            f"{config.indexer.db.uri}. Run `mmore index` first: graph retrieval scores "
            "passages that live in Milvus, it does not store them itself."
        )

    documents = _load_passages(client, collection_name)
    if not documents:
        raise ValueError(f"Collection '{collection_name}' is empty; nothing to build.")

    dense_model_config = _resolve_dense_model(config, client, collection_name)

    step_intro(
        GRAPH_INDEX_NAME,
        GRAPH_INDEX_EMOJI,
        "Build the entity graph used by graph retrieval",
        [
            f"{len(documents)} chunks",
            f"collection: {collection_name}",
            f"NER: {config.graph.spacy_model} ({config.graph.ner_backend})",
        ],
    )

    start = time.time()
    loading_start = model_loading_seconds()
    builder = TriGraphBuilder.from_config(config.graph, dense_model_config)
    tri_graph = builder.build(documents, collection_name)
    elapsed = time.time() - start - (model_loading_seconds() - loading_start)

    step_summary(
        GRAPH_INDEX_NAME,
        GRAPH_INDEX_EMOJI,
        elapsed,
        {
            "entities": str(tri_graph.num_entities),
            "sentences": str(tri_graph.num_sentences),
            "edges": str(tri_graph.graph.ecount()),
            "throughput": f"{len(documents) / elapsed:.1f} chunks/s"
            if elapsed
            else "-",
        },
    )


def _load_passages(client, collection_name: str) -> List[MultimodalSample]:
    """Read every chunk of the collection, keeping the Milvus primary key as the id."""
    samples: List[MultimodalSample] = []
    for row in _iter_rows(client, collection_name):
        text = row.get("text") or ""
        if not text.strip():
            continue
        samples.append(
            MultimodalSample(
                text=text,
                modalities=[],
                id=row["id"],
                document_id=row.get("document_id") or row["id"].split("+")[0],
            )
        )
    logger.info(f"Loaded {len(samples)} chunks from collection '{collection_name}'")
    return samples


def _iter_rows(client, collection_name: str) -> Iterable[Dict[str, Any]]:
    fields = ["id", "document_id", "text"]
    try:
        iterator = client.query_iterator(
            collection_name=collection_name,
            filter="",
            output_fields=fields,
            batch_size=_PAGE_SIZE,
        )
    except Exception as e:
        # Older / lite backends without an iterator: fall back to offset paging, which
        # Milvus caps at a 16384-row window.
        logger.debug(f"query_iterator unavailable ({e}); falling back to offset paging")
        offset = 0
        while True:
            page = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=fields,
                limit=_PAGE_SIZE,
                offset=offset,
            )
            if not page:
                return
            yield from page
            offset += len(page)
        return

    try:
        while True:
            page = iterator.next()
            if not page:
                return
            yield from page
    finally:
        iterator.close()


def _resolve_dense_model(
    config: GraphIndexConfig, client, collection_name: str
) -> DenseModelConfig:
    """Entity/sentence vectors must share the passage vectors' embedding space.

    The collection is authoritative: it already holds the passage vectors the graph will
    be scored against.
    """
    from_collection = cast(
        DenseModelConfig,
        get_model_from_index(client, "dense_embedding", collection_name),
    )
    if from_collection.model_name != config.indexer.dense_model.model_name:
        logger.warning(
            f"Config declares dense model '{config.indexer.dense_model.model_name}' but "
            f"collection '{collection_name}' was indexed with "
            f"'{from_collection.model_name}'. Using the collection's model."
        )
    return from_collection


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        required=True,
        help="Path to the graph index configuration file.",
    )
    parser.add_argument(
        "--collection-name",
        "-n",
        required=False,
        help="Name of the collection to build the graph for.",
    )
    args = parser.parse_args()

    graph_index(args.config_file, args.collection_name)
