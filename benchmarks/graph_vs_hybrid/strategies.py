"""Retrieval strategies beyond the three the benchmark started with.

Two families, both built out of retrievers that already exist rather than new ones:

* **Single-field baselines.** ``hybrid`` mixes dense and SPLADE, which hides which half is
  doing the work. ``dense`` and ``sparse`` pin ``search_type`` so each field can be read on
  its own, next to ``lexical`` (BM25). Four columns instead of one, at no indexing cost.

* **Fusions.** The standing finding about the graph is that it *finds* the right passages
  and *orders* them badly — recall@10 at parity with hybrid, hit@1 thirteen points behind.
  If that is true, the graph's candidate set combined with another retriever's ordering
  should beat either alone, and that is what ``graph+dense`` and ``graph+lexical`` test.

Fusion is by reciprocal rank, not by score: PageRank mass, cosine similarity and BM25 all
live on incomparable scales, and RRF only reads positions. Each part is queried to
``depth`` (deeper than the final k, so the fusion has material to reorder) and a passage
scores ``sum(1 / (constant + rank))`` over the parts that returned it.
"""

import logging
from typing import Any, Dict, List, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Strategies served by the Milvus retriever, and the search_type each one pins.
SEARCH_TYPES = {"hybrid": "hybrid", "dense": "dense", "sparse": "sparse"}

# Fusion strategies, and the parts they combine.
FUSIONS: Dict[str, Tuple[str, ...]] = {
    "graph+dense": ("graph", "dense"),
    "graph+lexical": ("graph", "lexical"),
    "graph+hybrid": ("graph", "hybrid"),
}


def uses_graph(strategy: str) -> bool:
    return strategy == "graph" or "graph" in FUSIONS.get(strategy, ())


class PinnedSearchType:
    """A Milvus retriever with ``search_type`` fixed, so a strategy stays one object.

    ``dense`` and ``sparse`` are the same retriever over the same collection as ``hybrid``;
    only the ranker weights differ. Pinning the value here keeps the call site identical
    for every strategy, including inside a fusion.
    """

    def __init__(self, retriever: Any, search_type: str):
        self.retriever = retriever
        self.search_type = search_type

    def invoke(self, query: Any, **kwargs: Any) -> List[Document]:
        payload = dict(query) if isinstance(query, dict) else {"input": query}
        payload["search_type"] = self.search_type
        return self.retriever.invoke(payload, **kwargs)


class ReciprocalRankFusion:
    """Rank-based fusion of several retrievers."""

    def __init__(
        self,
        parts: Sequence[Tuple[str, Any]],
        k: int,
        depth: int,
        constant: float = 60.0,
    ):
        if not parts:
            raise ValueError("A fusion needs at least one part.")
        self.parts = list(parts)
        self.k = k
        self.depth = max(depth, k)
        self.constant = constant

    def invoke(self, query: Any, **kwargs: Any) -> List[Document]:
        scores: Dict[str, float] = {}
        found_by: Dict[str, List[str]] = {}
        documents: Dict[str, Document] = {}

        for name, retriever in self.parts:
            docs = retriever.invoke(query, k=self.depth, **kwargs)
            for rank, doc in enumerate(docs[: self.depth], start=1):
                key = doc.metadata.get("id")
                if not key:
                    continue
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.constant + rank)
                found_by.setdefault(key, []).append(name)
                # Keep the first part's Document: the text is the same either way, and its
                # metadata carries the provenance of the retriever that ranked it best.
                documents.setdefault(key, doc)

        ordered = sorted(scores, key=lambda key: -scores[key])[: self.k]
        fused: List[Document] = []
        for rank, key in enumerate(ordered, start=1):
            doc = documents[key]
            metadata = dict(doc.metadata)
            metadata.update(
                {
                    "rank": rank,
                    "rrf_score": scores[key],
                    "found_by": found_by[key],
                    # `similarity` is what the corrective-RAG judge thresholds on, and the
                    # parts' own scores are not comparable. Rescaling by the best fused
                    # score keeps the relative gaps and lands in [0, 1], as GraphRetriever
                    # already does with PageRank mass.
                    "similarity": scores[key] / scores[ordered[0]],
                }
            )
            fused.append(Document(page_content=doc.page_content, metadata=metadata))
        return fused
