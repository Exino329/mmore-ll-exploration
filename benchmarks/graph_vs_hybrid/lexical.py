"""BM25 over the indexed passages: the reference baseline neither retriever can be read
without.

Why it is here. On MedHop the hybrid retriever scores hit@1 0.015 and recall@10 0.137,
and the graph 0.111 / 0.681 — from which the graph looks like a fivefold improvement. It
is not: exact matching on the accession identifiers the questions are built from reaches
recall@10 0.988 with no model at all. Dense embeddings cannot represent ``DB01171``, and
SPLADE's wordpieces barely do better, so a comparison between two retrievers that both
depend on them measures the embedding of identifiers rather than the retrieval strategy.
A lexical column is what makes that visible.

This is deliberately *not* an mmore retriever. It indexes nothing, persists nothing and
takes no configuration: it exists to put a floor under the other two, and a floor that
needed tuning would not be one. Okapi BM25 with the usual k1=1.5, b=0.75.
"""

import logging
import re
from typing import Any, Dict, List

import numpy as np
from langchain_core.documents import Document

from .config import BenchConfig

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")

K1 = 1.5
B = 0.75


def _tokenize(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Okapi BM25, duck-typing the part of the retriever interface the bench uses."""

    def __init__(self, passages: List[Dict[str, str]], k: int):
        from scipy import sparse

        self.k = k
        self.ids = [p["id"] for p in passages]
        self.texts = [p["text"] for p in passages]

        vocabulary: Dict[str, int] = {}
        rows: List[int] = []
        cols: List[int] = []
        values: List[int] = []
        lengths = np.zeros(len(passages), dtype=np.float32)

        for doc_idx, passage in enumerate(passages):
            counts: Dict[int, int] = {}
            tokens = _tokenize(passage["text"])
            lengths[doc_idx] = len(tokens)
            for token in tokens:
                term = vocabulary.setdefault(token, len(vocabulary))
                counts[term] = counts.get(term, 0) + 1
            rows.extend([doc_idx] * len(counts))
            cols.extend(counts.keys())
            values.extend(counts.values())

        self.vocabulary = vocabulary
        frequencies = sparse.csc_matrix(
            (values, (rows, cols)),
            shape=(len(passages), len(vocabulary)),
            dtype=np.float32,
        )

        # Precompute the per-(document, term) BM25 weight. The query then only has to sum
        # the columns of the terms it contains, which keeps a query at a few milliseconds.
        mean_length = float(lengths.mean()) if len(lengths) else 0.0
        norm = K1 * (1 - B + B * lengths / (mean_length or 1.0))

        weights = frequencies.tocoo()
        tf = weights.data
        denominator = tf + norm[weights.row]
        document_frequency = np.asarray((frequencies > 0).sum(axis=0)).ravel()
        idf = np.log(
            1.0
            + (len(passages) - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        weights.data = (tf * (K1 + 1) / denominator) * idf[weights.col]
        self.weights = weights.tocsc()

        logger.info(
            f"BM25 index: {len(passages)} passages, {len(vocabulary)} terms, "
            f"mean length {mean_length:.0f} tokens"
        )

    def invoke(self, query: Any, **kwargs: Any) -> List[Document]:
        text = query["input"] if isinstance(query, dict) else query
        k = kwargs.get("k", self.k)

        terms = {
            self.vocabulary[token]
            for token in _tokenize(text)
            if token in self.vocabulary
        }
        if not terms:
            return []

        scores = np.asarray(
            self.weights[:, sorted(terms)].sum(axis=1), dtype=np.float32
        ).ravel()
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]

        return [
            Document(
                page_content=self.texts[i],
                metadata={
                    "id": self.ids[i],
                    "rank": rank,
                    "bm25_score": float(scores[i]),
                },
            )
            for rank, i in enumerate(top, start=1)
            if scores[i] > 0
        ]


def build(config: BenchConfig, k: int) -> BM25Retriever:
    from .collection import indexed_passages

    return BM25Retriever(indexed_passages(config), k=k)
