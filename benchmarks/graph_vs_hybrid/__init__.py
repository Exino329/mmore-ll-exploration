"""Benchmark comparing mmore's hybrid dense+sparse RAG against LinearRAG graph retrieval.

Two questions, measured separately:

* **Indexing cost** — how much wall clock, CPU, memory and disk each strategy needs to
  make a corpus searchable (:mod:`.index_bench`).
* **Accuracy** — retrieval quality (:mod:`.retrieval_bench`) and end-to-end answer
  quality (:mod:`.answer_bench`) on the same questions, same collection, same k.

Everything a run produces lands in one directory, so runs are self-contained and
comparable. See ``README.md``.
"""
