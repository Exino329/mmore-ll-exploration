"""The corpus format shared by every task.

A benchmark document is one retrievable unit with a stable gold key. There is no chunking
step: a PubMed abstract or an ICD-11 description is already the unit a question points at,
and splitting it would make "did retrieval find the right one?" ambiguous.

The gold key travels in ``metadata.file_path`` (``gold://<key>``). ``MultimodalSample``
drops JSONL ``id`` fields when reading them back, so chunk ids only exist once the
collection is built, and ``file_path`` is the only per-document field both retrievers
return.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from mmore.type import DocumentMetadata, MultimodalSample

from .config import GOLD_PREFIX

logger = logging.getLogger(__name__)

TITLE_SEPARATOR = "\n\n"


@dataclass
class CorpusEntry:
    """One indexed document, as the task built it and as the evaluators read it back."""

    key: str
    """Gold key: the ICD-11 code, the PubMed id — whatever a question points at."""

    title: str
    body: str
    extra: Dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return f"{self.title}{TITLE_SEPARATOR}{self.body}".strip()

    def to_sample(self) -> MultimodalSample:
        return MultimodalSample(
            text=self.text,
            modalities=[],
            metadata=DocumentMetadata(
                file_path=f"{GOLD_PREFIX}{self.key}",
                extra={"gold_key": self.key, "title": self.title, **self.extra},
            ),
        )


@dataclass
class CorpusStats:
    documents: int
    gold_documents: int
    distractors: int
    mean_chars: float

    @classmethod
    def of(cls, entries: List[CorpusEntry], gold: int) -> "CorpusStats":
        return cls(
            documents=len(entries),
            gold_documents=gold,
            distractors=len(entries) - gold,
            mean_chars=(
                sum(len(e.text) for e in entries) / len(entries) if entries else 0.0
            ),
        )


def write_corpus(entries: List[CorpusEntry], path: Path, gold: int) -> CorpusStats:
    if not entries:
        raise ValueError("The corpus is empty; check the task's source settings.")

    keys = {entry.key for entry in entries}
    if len(keys) != len(entries):
        raise ValueError(
            f"Gold keys are not unique: {len(entries)} documents, {len(keys)} keys. "
            "Retrieval could not be scored against them."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    # MultimodalSample.to_jsonl opens the file in append mode, so re-running `prepare`
    # would otherwise stack a second copy of the corpus onto the first.
    path.unlink(missing_ok=True)
    MultimodalSample.to_jsonl(str(path), [entry.to_sample() for entry in entries])

    stats = CorpusStats.of(entries, gold)
    logger.info(
        f"Wrote {stats.documents} documents to {path} "
        f"({stats.gold_documents} gold, {stats.distractors} distractors, "
        f"{stats.mean_chars:.0f} chars on average)"
    )
    return stats


def read_corpus(path: Path) -> List[CorpusEntry]:
    """Re-read the JSONL written by :func:`write_corpus`."""
    entries: List[CorpusEntry] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            metadata: Dict[str, str] = record.get("metadata") or {}
            title = metadata.get("title", "")
            text: str = record["text"]
            body = (
                text[len(title) + len(TITLE_SEPARATOR) :]
                if title and text.startswith(title)
                else text
            )
            entries.append(
                CorpusEntry(
                    key=metadata.get("gold_key", ""),
                    title=title,
                    body=body,
                    extra={
                        k: v
                        for k, v in metadata.items()
                        if k not in {"gold_key", "title", "file_path"}
                    },
                )
            )
    return entries
