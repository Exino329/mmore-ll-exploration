"""Retrieval-strategy dispatch.

mmore ships two retrieval strategies over the same Milvus collection:

``hybrid``
    Dense + sparse ANN with an optional cross-encoder reranker. The default.

``graph``
    LinearRAG Tri-Graph search: entity activation by semantic bridging, then personalized
    PageRank over the entity/passage graph. Requires a graph built by ``mmore graph-index``.

Selected with ``retriever.type`` in the YAML config.
"""

from typing import Dict, Union

from ..utils import load_config
from .retriever import Retriever, RetrieverConfig

RETRIEVER_TYPES = ["hybrid", "graph"]


def load_retriever(config: Union[str, Dict, RetrieverConfig]) -> Retriever:
    """Instantiate the retriever named by ``config.type``."""
    config = load_config(config, RetrieverConfig)

    if config.type == "hybrid":
        return Retriever.from_config(config)

    if config.type == "graph":
        # Imported lazily so that igraph / spaCy stay optional for the default strategy.
        from .graph.retriever import GraphRetriever

        return GraphRetriever.from_config(config)

    raise ValueError(
        f"Unrecognized retriever type: {config.type}. Expected one of {RETRIEVER_TYPES}."
    )
