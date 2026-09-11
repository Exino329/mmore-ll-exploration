"""Questions and answer scoring, shared by every task.

A question carries two kinds of gold: the passages that should be retrieved, and the
answer that should be produced. That is what lets retrieval and generation be scored
independently — a wrong answer over the right passage is a generation failure, and the
report should not confuse the two.

Multi-hop tasks need more than one gold passage, and need to distinguish *which* one was
missed: on MedHop the hop-1 document names the subject of the question while the hop-2
document holds the answer and never mentions the subject. Dense retrieval can reach the
first from the question's wording alone; only the graph claims to reach the second.
``gold_groups`` keeps them apart so the report can score them apart.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Question:
    query_id: str
    question: str
    gold_answer: str
    style: str

    gold_keys: List[str] = field(default_factory=list)
    """Every passage that counts as correct. Single-gold tasks hold exactly one."""

    gold_groups: Dict[str, List[str]] = field(default_factory=dict)
    """Named subsets of ``gold_keys``, scored separately — ``{"hop1": [...], "hop2": [...]}``."""

    choices: List[str] = field(default_factory=list)
    """Closed answer set for this question, when it varies per question (MedHop's
    candidate drugs). Empty means the task-wide ``answer.labels`` applies."""

    gold_key: Optional[str] = None
    """Single-gold shorthand. Kept so question files written before multi-hop existed
    still load; mirrored into ``gold_keys``."""

    retrieval_query: Optional[str] = None
    """What the retriever is asked, when that is not the whole question. ``None`` means
    the question itself, which is the normal case.

    MedHop is why this exists: its questions carry the nine candidate answers, one of
    which is the gold accession, so the passage holding the answer is addressable by
    literal string match and no bridging is needed to reach it. Retrieving on the subject
    alone is what turns it back into a multi-hop measurement. The answer stage keeps
    asking the full question — the candidates are needed to answer it."""

    def __post_init__(self):
        if self.gold_key and not self.gold_keys:
            self.gold_keys = [self.gold_key]
        elif self.gold_keys and not self.gold_key:
            self.gold_key = self.gold_keys[0]

    @property
    def query(self) -> str:
        """The text handed to the retriever."""
        return self.retrieval_query or self.question


def write_questions(questions: List[Question], path: Path) -> int:
    if not questions:
        raise ValueError("No question was produced.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(asdict(question), ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(questions)} questions to {path}")
    return len(questions)


def read_questions(path: Path) -> List[Question]:
    with open(path, "r", encoding="utf-8") as handle:
        return [Question(**json.loads(line)) for line in handle]


# --- answer scoring ------------------------------------------------------------------------

# Ported from LinearRAG's src/utils.py, so `contain` accuracy is comparable to theirs.
_PUNCTUATION = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(value: object) -> str:
    text = "" if value is None else str(value)
    text = _PUNCTUATION.sub("", text.lower())
    return " ".join(_ARTICLES.sub(" ", text).split())


def contains_answer(prediction: object, gold: object) -> bool:
    predicted, expected = normalize_answer(prediction), normalize_answer(gold)
    return bool(predicted) and bool(expected) and expected in predicted


_FINAL_ANSWER = "answer:"


def final_answer(prediction: object) -> str:
    """The committed answer of a chain-of-thought response, or the whole of it.

    LinearRAG's QA prompt asks the model to reason after ``"Thought: "`` and conclude after
    ``"Answer: "``, and their ``run.py`` keeps only what follows. Scoring ``contain`` on the
    full response instead would count a gold string that merely appears somewhere in the
    reasoning, which inflates the metric on exactly the questions the model got wrong.
    """
    text = "" if prediction is None else str(prediction)
    position = text.lower().find(_FINAL_ANSWER)
    return (
        text[position + len(_FINAL_ANSWER) :].strip() if position >= 0 else text.strip()
    )


def extract_label(prediction: object, labels: List[str]) -> Optional[str]:
    """The first label the model actually committed to.

    A model asked for one word still writes "Yes, the evidence supports it", and sometimes
    "no" appears later in a sentence that is not the verdict. Taking the earliest label
    occurrence matches how these answers are read.
    """
    text = normalize_answer(prediction)
    if not text:
        return None
    positions = [
        (match.start(), label)
        for label in labels
        if (match := re.search(rf"\b{re.escape(label.lower())}\b", text))
    ]
    return min(positions)[1] if positions else None
