"""On-disk representation of the Tri-Graph.

The reference implementation keeps the graph, the entity/sentence bipartite maps and the
node index in memory only — it calls ``write_graphml`` once and never reads it back, so
indexing has to run in the same process as retrieval. Here the whole structure round-trips
through disk, which is what makes ``mmore graph-index`` a separate pipeline stage.

Vertex layout convention (relied upon by :mod:`.search`): the igraph holds the ``P``
passage vertices first, at indices ``0..P-1``, followed by the ``E`` entity vertices at
``P..P+E-1``. Sentences are *not* graph vertices — as in the paper they only exist in the
entity-sentence bipartite matrices used for semantic bridging.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

ARTIFACT_VERSION = 1

_MANIFEST = "manifest.json"
_GRAPH = "graph.pkl"
_PASSAGES = "passages.parquet"
_ENTITIES = "entities.parquet"
_SENTENCES = "sentences.parquet"
_ENTITY_EMB = "entity_embeddings.npy"
_SENTENCE_EMB = "sentence_embeddings.npy"
_ENTITY_TO_SENTENCE = "entity_to_sentence.npz"
_PASSAGE_TO_ENTITY = "passage_to_entity.npz"


def artifacts_path(artifacts_dir: str, collection_name: str) -> str:
    """Directory holding the artifacts of one collection."""
    return os.path.join(artifacts_dir, collection_name)


@dataclass
class TriGraphManifest:
    """Everything needed to validate a graph against a live Milvus collection."""

    collection_name: str
    dense_model: Dict[str, Any]
    spacy_model: str
    normalize_entities: bool
    embedding_dim: int
    num_passages: int
    num_entities: int
    num_sentences: int
    ner_backend: str = "spacy"
    """Which extractor produced these entities. Defaulted for graphs built before the
    medspaCy backend existed, which were all plain spaCy."""

    ner_options: Dict[str, Any] = field(default_factory=dict)
    """Backend-specific settings, so a retriever can reproduce index-time extraction
    without being told them again."""

    version: int = ARTIFACT_VERSION
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TriGraphManifest":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class TriGraph:
    """The Tri-Graph and its companion matrices, fully materialized."""

    manifest: TriGraphManifest

    graph: Any  # igraph.Graph — typed as Any to keep igraph an optional import

    passage_ids: List[str]
    """Milvus primary keys, index-aligned with the rows of ``passage_to_entity``."""

    entity_keys: List[str]
    """Canonical entity keys (lowercased when ``normalize_entities`` is on)."""

    entity_surfaces: List[str]
    """A representative surface form per entity, for display and query-side matching."""

    entity_embeddings: np.ndarray  # (E, d) float32, L2-normalized

    sentence_texts: List[str]
    sentence_passage_idx: np.ndarray  # (S,) int32, row index into ``passage_ids``
    sentence_embeddings: np.ndarray  # (S, d) float32, L2-normalized

    entity_to_sentence: Any  # scipy.sparse.csr_matrix (E, S), binary
    passage_to_entity: Any  # scipy.sparse.csr_matrix (P, E), raw mention counts

    def __post_init__(self):
        self.entity_key_to_idx = {key: i for i, key in enumerate(self.entity_keys)}
        self.passage_id_to_idx = {pid: i for i, pid in enumerate(self.passage_ids)}
        # Both transposes are consumed on every query: sentence -> co-mentioned entities
        # during bridging, entity -> passages when assembling the candidate pool.
        self.sentence_to_entity = self.entity_to_sentence.T.tocsr()
        self.entity_to_passage = self.passage_to_entity.T.tocsr()

    @property
    def num_passages(self) -> int:
        return len(self.passage_ids)

    @property
    def num_entities(self) -> int:
        return len(self.entity_keys)

    @property
    def num_sentences(self) -> int:
        return len(self.sentence_texts)

    def passage_vertices(self) -> np.ndarray:
        """igraph vertex ids of the passage nodes (see the layout convention above)."""
        return np.arange(self.num_passages, dtype=np.int64)

    def entity_vertex(self, entity_idx: int) -> int:
        return self.num_passages + entity_idx

    def entity_vertices(self) -> np.ndarray:
        return np.arange(
            self.num_passages, self.num_passages + self.num_entities, dtype=np.int64
        )

    # --- persistence -----------------------------------------------------------------

    def save(self, directory: str) -> None:
        import pandas as pd
        from scipy import sparse

        os.makedirs(directory, exist_ok=True)

        self.graph.write_pickle(os.path.join(directory, _GRAPH))

        pd.DataFrame({"passage_id": self.passage_ids}).to_parquet(
            os.path.join(directory, _PASSAGES), index=False
        )
        pd.DataFrame(
            {"key": self.entity_keys, "surface": self.entity_surfaces}
        ).to_parquet(os.path.join(directory, _ENTITIES), index=False)
        pd.DataFrame(
            {
                "text": self.sentence_texts,
                "passage_idx": self.sentence_passage_idx,
            }
        ).to_parquet(os.path.join(directory, _SENTENCES), index=False)

        np.save(os.path.join(directory, _ENTITY_EMB), self.entity_embeddings)
        np.save(os.path.join(directory, _SENTENCE_EMB), self.sentence_embeddings)

        sparse.save_npz(
            os.path.join(directory, _ENTITY_TO_SENTENCE),
            self.entity_to_sentence.tocsr(),
        )
        sparse.save_npz(
            os.path.join(directory, _PASSAGE_TO_ENTITY), self.passage_to_entity.tocsr()
        )

        with open(os.path.join(directory, _MANIFEST), "w") as f:
            json.dump(asdict(self.manifest), f, indent=2)

        logger.info(
            f"Saved Tri-Graph to {directory} "
            f"({self.num_passages} passages, {self.num_entities} entities, "
            f"{self.num_sentences} sentences, {self.graph.ecount()} edges)"
        )

    @classmethod
    def load(cls, directory: str, mmap_sentences: bool = True) -> "TriGraph":
        import igraph as ig
        import pandas as pd
        from scipy import sparse

        manifest = load_manifest(directory)
        if manifest is None:
            raise FileNotFoundError(
                f"No Tri-Graph found in {directory}. Build one with: mmore graph-index -c <config>"
            )
        if manifest.version != ARTIFACT_VERSION:
            raise ValueError(
                f"Tri-Graph in {directory} was built with artifact version "
                f"{manifest.version}, but this mmore expects {ARTIFACT_VERSION}. "
                "Rebuild it with: mmore graph-index -c <config>"
            )

        passages = pd.read_parquet(os.path.join(directory, _PASSAGES))
        entities = pd.read_parquet(os.path.join(directory, _ENTITIES))
        sentences = pd.read_parquet(os.path.join(directory, _SENTENCES))

        return cls(
            manifest=manifest,
            graph=ig.Graph.Read_Pickle(os.path.join(directory, _GRAPH)),
            passage_ids=passages["passage_id"].tolist(),
            entity_keys=entities["key"].tolist(),
            entity_surfaces=entities["surface"].tolist(),
            entity_embeddings=np.load(os.path.join(directory, _ENTITY_EMB)),
            sentence_texts=sentences["text"].tolist(),
            sentence_passage_idx=sentences["passage_idx"].to_numpy(dtype=np.int32),
            # Sentence vectors are the largest artifact and are only ever read through
            # fancy indexing on a handful of rows per hop, so memory-mapping them keeps
            # the retriever's resident set small.
            sentence_embeddings=np.load(
                os.path.join(directory, _SENTENCE_EMB),
                mmap_mode="r" if mmap_sentences else None,
            ),
            entity_to_sentence=sparse.load_npz(
                os.path.join(directory, _ENTITY_TO_SENTENCE)
            ).tocsr(),
            passage_to_entity=sparse.load_npz(
                os.path.join(directory, _PASSAGE_TO_ENTITY)
            ).tocsr(),
        )


def load_manifest(directory: str) -> Optional[TriGraphManifest]:
    """Read a manifest without materializing the graph. ``None`` if absent."""
    path = os.path.join(directory, _MANIFEST)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return TriGraphManifest.from_dict(json.load(f))
