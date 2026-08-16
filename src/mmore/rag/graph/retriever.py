"""LangChain retriever backed by the LinearRAG Tri-Graph.

``GraphRetriever`` subclasses the Milvus :class:`~mmore.rag.retriever.Retriever` rather
than ``BaseRetriever`` directly. That buys three things: the dense half of the passage
prior is served by the existing Milvus ANN index instead of a brute-force scan, passage
text never has to be duplicated outside Milvus, and every consumer that duck-types the
retriever (the corrective-RAG judge, the web fallback, the retriever API, the RAG CLI)
keeps working unchanged.
"""

import logging
import time
from typing import Any, Dict, List, Optional, cast

import numpy as np
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from ...index.indexer import get_model_from_index
from ...utils import load_config
from ..model.dense.base import DenseModelConfig
from ..retriever import Retriever, RetrieverConfig
from .config import GraphRetrieverConfig, MedspacyConfig
from .ner import SpacyEntityExtractor, build_entity_extractor
from .search import activate_entities, link_seed_entities, run_ppr, score_passages
from .store import TriGraph, TriGraphManifest, artifacts_path

logger = logging.getLogger(__name__)


class GraphRetriever(Retriever):
    """Retrieves passages by activating entities and running personalized PageRank."""

    tri_graph: Any
    """The loaded :class:`~mmore.rag.graph.store.TriGraph`."""

    graph_config: Any
    """The :class:`~mmore.rag.graph.config.GraphRetrieverConfig` in force."""

    entity_extractor: Any
    """:class:`~mmore.rag.graph.ner.SpacyEntityExtractor` used on the query."""

    @classmethod
    def from_config(cls, config: str | RetrieverConfig) -> "GraphRetriever":
        if isinstance(config, str):
            config = load_config(config, RetrieverConfig)

        base_kwargs = cls._base_kwargs_from_config(config)
        # `config.graph` stays an untyped dict on RetrieverConfig (dacite has no
        # discriminated unions), so it is re-parsed here into the real dataclass.
        graph_config: GraphRetrieverConfig = load_config(
            config.graph, GraphRetrieverConfig
        )

        directory = artifacts_path(graph_config.artifacts_dir, config.collection_name)
        tri_graph = TriGraph.load(directory)
        _check_embedding_space(
            tri_graph,
            cast(
                DenseModelConfig,
                get_model_from_index(
                    base_kwargs["client"], "dense_embedding", config.collection_name
                ),
            ),
            directory,
        )

        logger.info(
            f"Loaded Tri-Graph from {directory}: {tri_graph.num_passages} passages, "
            f"{tri_graph.num_entities} entities, {tri_graph.num_sentences} sentences"
        )

        return cls(
            **base_kwargs,
            tri_graph=tri_graph,
            graph_config=graph_config,
            entity_extractor=_query_entity_extractor(graph_config, tri_graph.manifest),
        )

    # --- retrieval ---------------------------------------------------------------------

    def _get_relevant_documents(
        self,
        query: str | Dict[str, Any],
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        if isinstance(query, str):
            query_input: str = query
            collection_name: str = kwargs.get("collection_name", "my_docs")
            document_ids: List[str] = kwargs.get("document_ids", [])
        else:
            if "input" not in query:
                raise ValueError("Missing query input")
            query_input = query["input"]
            collection_name = query.get("collection_name", "my_docs")
            document_ids = query.get("document_ids", [])

        k: int = kwargs.get("k", self.k)
        if k == 0:
            return []

        self._emit_stage("retrieve")
        time_start = time.perf_counter()

        ranked = self._graph_search(query_input, collection_name, document_ids, k)
        if ranked is None:
            # No query entity could be linked into the graph: the graph carries no signal
            # for this query, so fall back to the hybrid dense+sparse retriever. Upstream
            # falls back to dense-only.
            logger.debug(
                f"No seed entity linked for query {query_input!r}; falling back to hybrid retrieval"
            )
            return super()._get_relevant_documents(
                query, run_manager=run_manager, **kwargs
            )

        docs = self._materialize(ranked, collection_name)
        retrieve_elapsed = time.perf_counter() - time_start

        if self.use_web:
            web_docs = self._get_web_documents(query_input, max_results=self.k)
            for offset, doc in enumerate(docs):
                doc.metadata["rank"] = len(web_docs) + offset + 1
            docs = web_docs + docs

        rerank_elapsed = 0.0
        if self.reranker_model:
            self._emit_stage("rerank")
            time_start = time.perf_counter()
            docs = self.rerank(query_input, docs)
            rerank_elapsed = time.perf_counter() - time_start

        self._retrieve_seconds += retrieve_elapsed
        self._rerank_seconds += rerank_elapsed

        return docs

    def _graph_search(
        self,
        query: str,
        collection_name: str,
        document_ids: List[str],
        k: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run the two-stage search. ``None`` when no seed entity could be linked."""
        config: GraphRetrieverConfig = self.graph_config
        tri_graph: TriGraph = self.tri_graph

        query_entities = self.entity_extractor.extract_query_entities(query)
        if not query_entities:
            return None

        seeds = link_seed_entities(
            tri_graph,
            _normalize(np.asarray(self.dense_model.embed_documents(query_entities))),
            config.seed_min_similarity,
        )
        if not seeds:
            return None

        question_embedding = _normalize(
            np.asarray([self.dense_model.embed_query(query)])
        )[0]

        entity_weights, activated = activate_entities(
            tri_graph, question_embedding, seeds, config
        )
        passage_weights = score_passages(
            tri_graph,
            activated,
            self._dense_prior(query, collection_name, document_ids, config),
            config,
        )
        scores = run_ppr(tri_graph, entity_weights, passage_weights, config.damping)

        order = np.argsort(scores)[::-1]
        allowed = set(document_ids)
        activated_surfaces = [
            tri_graph.entity_surfaces[i]
            for i in sorted(activated, key=lambda i: -activated[i].score)
        ][:10]

        ranked: List[Dict[str, Any]] = []
        best = float(scores[order[0]]) if order.size else 0.0
        for passage_idx in order:
            if len(ranked) >= k:
                break
            score = float(scores[passage_idx])
            if score <= 0:
                break
            passage_id = tri_graph.passage_ids[int(passage_idx)]
            if allowed and passage_id.split("+")[0] not in allowed:
                continue
            ranked.append(
                {
                    "id": passage_id,
                    "graph_score": score,
                    # The judge's metric thresholds are calibrated on cosine similarity,
                    # while raw PageRank mass sums to 1 over every vertex. Rescaling by
                    # the top score keeps the relative gaps and lands in [0, 1].
                    "similarity": score / best if best > 0 else 0.0,
                    "activated_entities": activated_surfaces,
                }
            )

        return ranked or None

    def _dense_prior(
        self,
        query: str,
        collection_name: str,
        document_ids: List[str],
        config: GraphRetrieverConfig,
    ) -> Dict[int, float]:
        """Dense relevance of the candidate pool, keyed by passage row index."""
        results = self.retrieve(
            query=query,
            collection_name=collection_name,
            k=config.dense_candidates,
            search_type="dense",
            output_fields=["text"],
            document_ids=document_ids,
        )

        prior: Dict[int, float] = {}
        unknown = 0
        for result in results:
            passage_idx = self.tri_graph.passage_id_to_idx.get(result["id"])
            if passage_idx is None:
                unknown += 1
                continue
            prior[passage_idx] = float(result["distance"])

        if unknown:
            logger.warning(
                f"{unknown}/{len(results)} dense candidates are absent from the Tri-Graph. "
                "The collection has been indexed since the graph was built; "
                "re-run `mmore graph-index` to include them."
            )
        return prior

    def _materialize(
        self, ranked: List[Dict[str, Any]], collection_name: str
    ) -> List[Document]:
        """Fetch passage text and metadata from Milvus, preserving the graph ranking."""
        ids_str = ",".join(f'"{item["id"]}"' for item in ranked)
        rows = self.client.query(
            collection_name=collection_name,
            filter=f"id in [{ids_str}]",
            output_fields=["id", "text", "paragraph_positions", "file_path"],
        )
        by_id = {row["id"]: row for row in rows}

        docs: List[Document] = []
        for rank, item in enumerate(ranked, start=1):
            row = by_id.get(item["id"])
            if row is None:
                logger.warning(
                    f"Passage {item['id']} is in the Tri-Graph but not in collection "
                    f"{collection_name}; skipping."
                )
                continue
            docs.append(
                Document(
                    page_content=row["text"],
                    metadata={
                        "id": item["id"],
                        "rank": rank,
                        "similarity": item["similarity"],
                        "graph_score": item["graph_score"],
                        "activated_entities": item["activated_entities"],
                        "paragraph_positions": row.get("paragraph_positions", []),
                        "file_path": row.get("file_path", ""),
                    },
                )
            )

        # Ranks must stay contiguous: RAGPipeline.format_docs prints them as citations.
        for rank, doc in enumerate(docs, start=1):
            doc.metadata["rank"] = rank
        return docs


def _query_entity_extractor(
    graph_config: GraphRetrieverConfig, manifest: TriGraphManifest
) -> SpacyEntityExtractor:
    """The extractor for query entities, defaulting to the one that built the graph.

    Query entities are matched against corpus entities by embedding similarity, so the
    two sides have to be produced the same way; the manifest is what keeps them aligned
    when the retriever config says nothing.
    """
    backend = graph_config.ner_backend or manifest.ner_backend
    medspacy = graph_config.medspacy
    if medspacy is None and backend == "medspacy":
        medspacy = MedspacyConfig(**manifest.ner_options)

    return build_entity_extractor(
        backend=backend,
        model_name=graph_config.spacy_model or manifest.spacy_model,
        medspacy=medspacy,
        excluded_labels=graph_config.excluded_entity_labels,
    )


def _check_embedding_space(
    tri_graph: TriGraph, collection_model: DenseModelConfig, directory: str
) -> None:
    """Entity, sentence and passage vectors must come from the same model."""
    built_with = tri_graph.manifest.dense_model.get("model_name")
    if built_with != collection_model.model_name:
        raise ValueError(
            f"The Tri-Graph in {directory} was built with dense model '{built_with}', but "
            f"collection '{tri_graph.manifest.collection_name}' is indexed with "
            f"'{collection_model.model_name}'. Entity and passage vectors would not be "
            "comparable. Re-run `mmore graph-index` with the current collection."
        )


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return matrix / norms
