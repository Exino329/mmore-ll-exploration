"""Task dispatch.

A task owns three things: how the corpus is built, how the questions are built, and how it
describes itself in the report. Everything downstream — indexing timings, retrieval
metrics, answer scoring, the report — is task-agnostic and works off `corpus.jsonl` and
`qa.jsonl`.
"""

from typing import Dict, Protocol

from .config import BenchConfig
from .corpus import CorpusStats


class Task(Protocol):
    def prepare_corpus(self, config: BenchConfig) -> CorpusStats: ...

    def prepare_questions(self, config: BenchConfig) -> int: ...

    def describe(self, config: BenchConfig) -> Dict[str, str]: ...


def load_task(name: str) -> Task:
    if name == "pubmedqa":
        from . import pubmedqa

        return pubmedqa  # type: ignore[return-value]
    if name == "medhop":
        from . import medhop

        return medhop  # type: ignore[return-value]
    if name == "icd11":
        from . import icd11

        return icd11  # type: ignore[return-value]
    if name == "hotpotqa":
        from . import hotpotqa

        return hotpotqa  # type: ignore[return-value]
    raise ValueError(f"Unknown task: {name}")
