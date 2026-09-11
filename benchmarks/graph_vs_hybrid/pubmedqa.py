"""Task: PubMedQA with distractors.

The 1,000 expert-annotated instances of `pqa_labeled` supply both golds — the abstract a
question was written from, and its yes/no/maybe decision — so retrieval and generation are
scored without an LLM anywhere in the loop.

On their own those 1,000 abstracts make retrieval trivial. The corpus is therefore padded
with abstracts from a second split that no question points at, which is what turns the
comparison into a search problem. Distractors are streamed, so only the requested number
is downloaded.

The conclusion of each abstract (`long_answer`) is deliberately left out of the indexed
text: it states the answer, and indexing it would measure copying rather than retrieval.
That is PubMedQA's standard reasoning setting.
"""

import logging
import random
from typing import Any, Dict, Iterator, List, Tuple

from .config import BenchConfig
from .corpus import CorpusEntry, CorpusStats, read_corpus, write_corpus
from .qa import Question, write_questions

logger = logging.getLogger(__name__)

DATASET = "qiaojin/PubMedQA"


def _abstract(record: Dict[str, Any], include_mesh: bool) -> str:
    """The abstract as an indexable passage, section labels kept."""
    context = record.get("context") or {}
    contexts: List[str] = list(context.get("contexts") or [])
    labels: List[str] = list(context.get("labels") or [])

    sections = [
        f"{labels[i].title()}: {paragraph}" if i < len(labels) else paragraph
        for i, paragraph in enumerate(contexts)
    ]
    text = "\n\n".join(section.strip() for section in sections if section.strip())

    meshes = context.get("meshes") or []
    if include_mesh and meshes:
        text = f"{text}\n\nMeSH terms: {', '.join(meshes)}"
    return text


def _entry(record: Dict[str, Any], include_mesh: bool, gold: bool) -> CorpusEntry:
    return CorpusEntry(
        key=str(record["pubid"]),
        # No title. PubMedQA's questions are derived from the article titles, and the
        # dataset ships the question but not the title — putting it on the passage would
        # place the query verbatim inside its own gold document and turn retrieval into
        # exact string matching.
        title="",
        body=_abstract(record, include_mesh),
        extra={"pubid": str(record["pubid"]), "role": "gold" if gold else "distractor"},
    )


def _stream_distractors(
    split: str, wanted: int, exclude: set, include_mesh: bool
) -> Iterator[CorpusEntry]:
    from datasets import load_dataset

    stream = load_dataset(DATASET, split, split="train", streaming=True)
    produced = 0
    for record in stream:
        if produced >= wanted:
            return
        key = str(record["pubid"])
        if key in exclude:
            continue
        entry = _entry(record, include_mesh, gold=False)
        if not entry.body.strip():
            continue
        exclude.add(key)
        produced += 1
        if produced % 5000 == 0:
            logger.info(f"  ... {produced}/{wanted} distractors")
        yield entry


def build_corpus(config: BenchConfig) -> Tuple[List[CorpusEntry], int]:
    from datasets import load_dataset

    settings = config.pubmedqa
    gold_records = load_dataset(DATASET, settings.gold_split, split="train")
    logger.info(f"Loaded {len(gold_records)} gold instances from {settings.gold_split}")

    entries = [
        _entry(record, settings.include_mesh, gold=True) for record in gold_records
    ]
    entries = [entry for entry in entries if entry.body.strip()]
    gold_count = len(entries)

    if settings.distractors:
        logger.info(
            f"Streaming {settings.distractors} distractors from "
            f"{settings.distractor_split}"
        )
        entries.extend(
            _stream_distractors(
                settings.distractor_split,
                settings.distractors,
                {entry.key for entry in entries},
                settings.include_mesh,
            )
        )
    return entries, gold_count


def build_questions(config: BenchConfig, entries: List[CorpusEntry]) -> List[Question]:
    from datasets import load_dataset

    settings = config.pubmedqa
    indexed = {entry.key for entry in entries}

    records = [
        record
        for record in load_dataset(DATASET, settings.gold_split, split="train")
        if str(record["pubid"]) in indexed
    ]
    sample = random.Random(settings.seed).sample(
        records, min(settings.questions, len(records))
    )
    if len(sample) < settings.questions:
        logger.warning(
            f"Only {len(sample)} annotated instances are in the corpus; "
            f"pubmedqa.questions={settings.questions} was requested."
        )

    return [
        Question(
            query_id=str(record["pubid"]),
            question=str(record["question"]).strip(),
            gold_key=str(record["pubid"]),
            gold_answer=str(record["final_decision"]).strip().lower(),
            style="pubmedqa",
        )
        for record in sample
    ]


# --- task interface ---------------------------------------------------------------------


def prepare_corpus(config: BenchConfig) -> CorpusStats:
    entries, gold = build_corpus(config)
    return write_corpus(entries, config.corpus_path, gold)


def prepare_questions(config: BenchConfig) -> int:
    return write_questions(
        build_questions(config, read_corpus(config.corpus_path)), config.qa_path
    )


def describe(config: BenchConfig) -> Dict[str, str]:
    settings = config.pubmedqa
    return {
        "task": "PubMedQA (yes/no/maybe)",
        "corpus": f"{settings.gold_split} + {settings.distractors:,} distractors "
        f"from {settings.distractor_split}",
        "questions": f"{settings.questions} expert-annotated",
    }
