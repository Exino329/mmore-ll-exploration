"""Graph-based retrieval (LinearRAG Tri-Graph).

Only the lightweight pieces are re-exported here; ``builder`` and ``retriever`` are
imported by module path so that igraph, spaCy and pymilvus stay optional dependencies.
"""

from .config import GraphBuildConfig, GraphRetrieverConfig, MedspacyConfig
from .store import TriGraph, TriGraphManifest, artifacts_path, load_manifest

__all__ = [
    "GraphBuildConfig",
    "GraphRetrieverConfig",
    "MedspacyConfig",
    "TriGraph",
    "TriGraphManifest",
    "artifacts_path",
    "load_manifest",
]
