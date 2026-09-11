"""Task: MedHop — multi-hop reasoning over Medline abstracts.

The one place in this benchmark where graph retrieval is asked to do what it was designed
for. A MedHop question is a drug-interaction chain: ``interacts_with DB01171?`` with nine
candidate answers, resolvable only by following identifiers across abstracts. Entities in
the corpus are normalised to DrugBank and UniProt accessions (``DB08820``, ``P13569``), so
the chain is spelled out in the text — and scispaCy tags those accessions as entities,
which is precisely what the Tri-Graph links on.

**Per-hop gold, derived.** MedHop ships the whole support set per question without
labelling which documents form the chain, so the labels are reconstructed:

* ``hop1`` — supports naming the question's subject accession (median 3 per question).
* ``hop2`` — supports naming the answer accession (median 1).

340 of the 342 validation questions have the two sets disjoint, which is the bridge
structure the task claims to have. The hop-2 document holds the answer and never mentions
the subject, so nothing in the question's wording points at it: dense retrieval cannot
reach it from the query alone. **Recall of hop2 is the discriminating measurement of this
whole benchmark.**

Loaded from the Hub's parquet conversion — `bigbio/medhop` ships a loading script, which
recent `datasets` releases refuse to run.
"""

import hashlib
import logging
import random
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import BenchConfig
from .corpus import CorpusEntry, CorpusStats, read_corpus, write_corpus
from .qa import Question, write_questions

logger = logging.getLogger(__name__)

PARQUET = (
    "https://huggingface.co/datasets/bigbio/medhop/resolve/"
    "refs%2Fconvert%2Fparquet/medhop_source/{split}/0000.parquet"
)

_ACCESSION = re.compile(r"\b[A-Z]{1,2}\d{4,}\b")


def _key(text: str) -> str:
    """Content hash: MedHop's supports carry no identifier, and repeat across questions."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _subject(query: str) -> str:
    """``interacts_with DB01171?`` → ``DB01171``."""
    return str(query).split()[-1].rstrip("?").strip()


def _load(split: str) -> pd.DataFrame:
    logger.info(f"Loading MedHop {split}")
    return pd.read_parquet(PARQUET.format(split=split))


def _mentions(text: str, accession: str) -> bool:
    # Whole-token match: DB0117 must not match inside DB01171.
    return re.search(rf"\b{re.escape(accession)}\b", text) is not None


# --- corpus -------------------------------------------------------------------------------


def build_corpus(config: BenchConfig) -> Tuple[List[CorpusEntry], int]:
    settings = config.medhop
    frames = [(_load(settings.question_split), True)]
    if settings.distractor_split:
        frames.append((_load(settings.distractor_split), False))

    entries: Dict[str, CorpusEntry] = {}
    gold_keys: set = set()
    for frame, is_question_split in frames:
        for _, row in frame.iterrows():
            for support in row["supports"]:
                text = str(support).strip()
                if not text:
                    continue
                key = _key(text)
                if key not in entries:
                    entries[key] = CorpusEntry(
                        key=key,
                        title="",
                        body=text,
                        extra={
                            "accessions": ",".join(
                                sorted(set(_ACCESSION.findall(text)))[:32]
                            )
                        },
                    )
                if is_question_split:
                    gold_keys.add(key)

    logger.info(
        f"{len(entries)} unique documents "
        f"({len(gold_keys)} reachable by a validation question)"
    )
    return list(entries.values()), len(gold_keys)


# --- questions ----------------------------------------------------------------------------

TEMPLATE = (
    "Which of the following substances interacts with {subject}?\n"
    "Answer with one identifier from this list and nothing else: {choices}"
)

SUBJECT_QUERY = "Which substances interact with {subject}?"
"""Retrieval query for ``retrieval_query: subject_only``.

The full question lists the nine candidates, and the gold answer is always one of them —
so every hop-2 passage, which is defined as a passage naming the answer accession, can be
reached by matching a string copied out of the question. Measured with a lexical baseline
that does nothing else: hop2@10 is 0.111 with the candidates in the query and **0.003**
without. The candidates therefore decide whether MedHop measures bridging or string
matching, and the choice belongs in the config rather than in a footnote."""


def build_questions(config: BenchConfig, entries: List[CorpusEntry]) -> List[Question]:
    settings = config.medhop
    indexed = {entry.key for entry in entries}
    frame = _load(settings.question_split)

    questions: List[Question] = []
    for _, row in frame.iterrows():
        subject = _subject(row["query"])
        answer = str(row["answer"])
        supports = [str(s).strip() for s in row["supports"] if str(s).strip()]

        hop1 = [_key(s) for s in supports if _mentions(s, subject)]
        hop2 = [_key(s) for s in supports if _mentions(s, answer)]
        hop1 = [k for k in hop1 if k in indexed]
        hop2 = [k for k in hop2 if k in indexed]
        if not hop1 or not hop2:
            continue

        questions.append(
            Question(
                query_id=str(row["id"]),
                question=TEMPLATE.format(
                    subject=subject, choices=", ".join(map(str, row["candidates"]))
                ),
                gold_answer=answer,
                style="medhop",
                # Ranking metrics use the union; the hop breakdown is what matters.
                gold_keys=sorted(set(hop1) | set(hop2)),
                gold_groups={"hop1": sorted(set(hop1)), "hop2": sorted(set(hop2))},
                choices=[str(c) for c in row["candidates"]],
                retrieval_query=(
                    SUBJECT_QUERY.format(subject=subject)
                    if settings.retrieval_query == "subject_only"
                    else None
                ),
            )
        )

    if settings.questions and len(questions) > settings.questions:
        questions = random.Random(settings.seed).sample(questions, settings.questions)

    dropped = len(frame) - len(questions)
    if dropped > 0:
        logger.info(f"{dropped} questions dropped (no hop-1 or hop-2 support indexed)")
    return questions


# --- task interface -------------------------------------------------------------------------


def prepare_corpus(config: BenchConfig) -> CorpusStats:
    entries, gold = build_corpus(config)
    return write_corpus(entries, config.corpus_path, gold)


def prepare_questions(config: BenchConfig) -> int:
    return write_questions(
        build_questions(config, read_corpus(config.corpus_path)), config.qa_path
    )


def describe(config: BenchConfig) -> Dict[str, Any]:
    settings = config.medhop
    return {
        "task": "MedHop (multi-hop drug interactions)",
        "corpus": f"supports of {settings.question_split}"
        + (f" + {settings.distractor_split}" if settings.distractor_split else ""),
        "questions": f"{settings.question_split}, 9 candidates each",
    }
