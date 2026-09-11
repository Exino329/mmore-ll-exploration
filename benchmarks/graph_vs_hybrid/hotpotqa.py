"""Task: HotpotQA — the setting LinearRAG itself reports on.

The first task in this benchmark whose numbers are comparable to a published table. The
paper's per-dataset preset for HotpotQA (``scripts/run.sh``: ``MAX_ITERATION=3``,
``THRESHOLD=0.4``, ``PASSAGE_RATIO=0.05``, ``TOP_K_SENTENCE=1``, ``en_core_web_trf``,
``all-mpnet-base-v2``) is what :mod:`mmore.rag.graph.config` already defaults to, so a run
here says whether this port reproduces their result rather than merely whether the graph
helps on a corpus we built ourselves.

**Their question set, not their corpus.** LinearRAG's release ships
``Zly0523/linear-rag/hotpotqa/{questions,chunks}.json``. The questions are usable and are
what this task asks: 1,000 HotpotQA dev instances (811 bridge, 189 comparison), whose ids
join 1000/1000 into the ``distractor`` validation split. The chunks are not:

* 1,311 blocks of ~4,650 characters cut blindly through the whole concatenated corpus.
  Boundaries fall mid-word — chunk 1 opens on ``"##dh"``, a leftover WordPiece
  continuation — so a block holds the tail of one article and the head of another.
* The text is lowercased and detokenized (``"$ 50 million"``, ``"don ' t"``), which is
  most of what a general-domain NER model keys on.
* Each block carries an ``"N:"`` index prefix that goes into the embedding with the rest.
* Nothing labels a gold passage. Their ``evidence`` field is the full 10-paragraph
  distractor context, supporting facts included but not marked, and their
  ``src/evaluate.py`` computes ``contain`` accuracy and an LLM verdict — **no retrieval
  metric at all**. There is no published recall@k to be comparable to.

So the corpus is rebuilt at the granularity a question actually points at: one document
per Wikipedia article, from the dev ``context`` field. Pooling the ten contexts of the
1,000 asked questions gives 9,811 unique articles, which is the corpus size the HotpotQA
retrieval literature reports on.

**Per-hop gold, derived.** HotpotQA labels the two supporting articles but not which one
is the bridge. Two rules, both mirroring :mod:`.medhop`:

* the *far* article (``hop2``) holds the answer string and is reached only through the
  other one — 639/811 bridge questions have the answer in exactly one support;
* failing that, the *near* article (``hop1``) is the one whose title the question names
  outright — 429/811.

Together they split 726/811 bridge questions (90%), and where both fire they agree on 85%.
The remaining 85 keep an aggregate gold and no hop split, so ``hop2_at_k`` is scored over
the questions that define it rather than diluted by the ones that cannot.

**The control group is free.** ``question_type`` partitions the same corpus, the same
pipeline and the same run into 811 bridge questions, where the answer article is reachable
only by bridging, and 189 comparison questions, where the question names both entities and
nothing needs to be bridged. If graph retrieval's advantage is real it shows up on
``bridge``/``hop2`` and vanishes on ``comparison`` — a within-run control no other task in
this benchmark provides.
"""

import hashlib
import json
import logging
import re
import urllib.request
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import BenchConfig
from .corpus import CorpusEntry, CorpusStats, read_corpus, write_corpus
from .qa import Question, write_questions

logger = logging.getLogger(__name__)

DEV_PARQUET = (
    "https://huggingface.co/api/datasets/hotpotqa/hotpot_qa/parquet/"
    "distractor/validation/0.parquet"
)
"""The ``distractor`` validation split: 7,405 questions, each with its 10-paragraph
context and its supporting facts. The parquet conversion, because `hotpot_qa` ships a
loading script that recent `datasets` releases refuse to run."""

LINEARRAG_QUESTIONS = (
    "https://huggingface.co/datasets/Zly0523/linear-rag/resolve/main/"
    "hotpotqa/questions.json"
)
"""LinearRAG's own 1,000-question subsample, taken for its ids alone."""


def _key(title: str) -> str:
    """Gold key of an article. Titles are unique in HotpotQA but carry punctuation and
    non-ASCII, and the key travels through ``metadata.file_path``."""
    return hashlib.sha1(title.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def _load_dev() -> pd.DataFrame:
    logger.info("Loading HotpotQA distractor/validation")
    return pd.read_parquet(DEV_PARQUET)


@lru_cache(maxsize=1)
def _linearrag_ids() -> Tuple[str, ...]:
    logger.info(f"Loading LinearRAG's question subsample from {LINEARRAG_QUESTIONS}")
    with urllib.request.urlopen(LINEARRAG_QUESTIONS) as response:
        payload = json.loads(response.read())
    return tuple(str(record["id"]) for record in payload)


def _paragraphs(context: Any) -> List[Tuple[str, str]]:
    """``(title, body)`` of the ten articles offered with a question."""
    return [
        (str(title), " ".join(str(s) for s in sentences).strip())
        for title, sentences in zip(context["title"], context["sentences"])
    ]


def _selected(config: BenchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(asked, distractors)`` — the questions scored, and the ones pooled in for their
    contexts alone. Deterministic, so `prepare_corpus` and `prepare_questions` agree."""
    settings = config.hotpotqa
    dev = _load_dev()

    if settings.question_source == "linearrag":
        wanted = list(_linearrag_ids())
        asked = dev[dev["id"].isin(set(wanted))]
        missing = len(wanted) - len(asked)
        if missing:
            raise ValueError(
                f"{missing} of LinearRAG's {len(wanted)} question ids are absent from "
                "the HotpotQA distractor validation split; the two releases have drifted."
            )
        # Their file order, so a subsample of theirs is a prefix of theirs.
        asked = asked.set_index("id").loc[wanted].reset_index()
    else:
        asked = dev.sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)

    if settings.questions and len(asked) > settings.questions:
        # `linearrag` is already in their file order and `dev` has just been shuffled by
        # the seed, so a prefix is the right subsample in both cases.
        asked = asked.iloc[: settings.questions]

    rest = dev[~dev["id"].isin(set(asked["id"]))]
    distractors = (
        rest.sample(
            n=min(settings.distractor_questions, len(rest)),
            random_state=settings.seed,
        )
        if settings.distractor_questions
        else rest.iloc[:0]
    )
    return asked, distractors


# --- corpus -------------------------------------------------------------------------------


def build_corpus(config: BenchConfig) -> Tuple[List[CorpusEntry], int]:
    settings = config.hotpotqa
    asked, distractors = _selected(config)

    entries: Dict[str, CorpusEntry] = {}
    gold_keys: set = set()

    for frame in (asked, distractors):
        for _, row in frame.iterrows():
            for title, body in _paragraphs(row["context"]):
                if not body:
                    continue
                key = _key(title)
                if key not in entries:
                    entries[key] = CorpusEntry(
                        key=key,
                        # The title is the article's entity name, and indexing it is what
                        # HippoRAG and LinearRAG do. It also makes hop-1 addressable by
                        # literal string match, which is why this is a setting.
                        title=title if settings.include_title else "",
                        body=body,
                        extra={"page_title": title},
                    )

    for _, row in asked.iterrows():
        for title in row["supporting_facts"]["title"]:
            gold_keys.add(_key(str(title)))

    logger.info(
        f"{len(entries)} unique articles from {len(asked)} asked questions"
        + (f" + {len(distractors)} distractor questions" if len(distractors) else "")
        + f" ({len(gold_keys)} of them supporting a question)"
    )
    return list(entries.values()), len(gold_keys)


# --- questions ----------------------------------------------------------------------------


_MIN_TITLE_CHARS = 3
"""Below this a title carries no discriminating signal: HotpotQA has articles called
"It", "Us" and "Up", and asking whether a question "names" one of those is meaningless."""


def _names(question: str, title: str) -> bool:
    """Whether the question names the article outright.

    On word boundaries, not as a bare substring: the title "A" is inside "came", and a
    spurious hit here silently assigns the hop direction backwards.
    """
    title = title.lower().strip()
    if len(title) < _MIN_TITLE_CHARS:
        return False
    return re.search(rf"(?<!\w){re.escape(title)}(?!\w)", question) is not None


def _hop_groups(row: Any) -> Dict[str, List[str]]:
    """Named gold subsets: ``bridge``/``comparison`` always, ``hop1``/``hop2`` when the
    bridge direction is derivable. See the module docstring for the two rules and their
    measured coverage."""
    supports = list(dict.fromkeys(str(t) for t in row["supporting_facts"]["title"]))
    keys = [_key(t) for t in supports]

    if str(row["type"]) != "bridge" or len(supports) != 2:
        return {str(row["type"]): keys}

    bodies = {t: b.lower() for t, b in _paragraphs(row["context"]) if t in supports}
    answer = str(row["answer"]).lower().strip()
    question = str(row["question"]).lower()

    holds_answer = [t for t in supports if answer and answer in bodies.get(t, "")]
    named_in_question = [t for t in supports if _names(question, t)]

    if len(holds_answer) == 1:
        far = holds_answer[0]
    elif len(named_in_question) == 1:
        far = next(t for t in supports if t != named_in_question[0])
    else:
        return {"bridge": keys}

    near = next(t for t in supports if t != far)
    return {"bridge": keys, "hop1": [_key(near)], "hop2": [_key(far)]}


def build_questions(config: BenchConfig, entries: List[CorpusEntry]) -> List[Question]:
    indexed = {entry.key for entry in entries}
    asked, _ = _selected(config)

    questions: List[Question] = []
    for _, row in asked.iterrows():
        groups = _hop_groups(row)
        gold_keys = sorted(
            {_key(str(t)) for t in row["supporting_facts"]["title"]} & indexed
        )
        if len(gold_keys) < 2:
            # Both supports are always in the context they came from; a short set means an
            # empty article body was skipped, and the question is no longer answerable.
            continue

        questions.append(
            Question(
                query_id=str(row["id"]),
                question=str(row["question"]),
                gold_answer=str(row["answer"]),
                style=f"hotpotqa-{row['type']}",
                gold_keys=gold_keys,
                gold_groups={
                    name: [k for k in members if k in indexed]
                    for name, members in groups.items()
                },
            )
        )

    dropped = len(asked) - len(questions)
    if dropped > 0:
        logger.info(
            f"{dropped} questions dropped (a supporting article was not indexed)"
        )
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
    settings = config.hotpotqa
    source = (
        "LinearRAG's 1,000-question subsample"
        if settings.question_source == "linearrag"
        else "HotpotQA distractor/validation"
    )
    return {
        "task": "HotpotQA (multi-hop Wikipedia, bridge + comparison)",
        "corpus": "articles pooled from the asked questions' 10-paragraph contexts"
        + (
            f" + {settings.distractor_questions} distractor questions"
            if settings.distractor_questions
            else ""
        ),
        "questions": f"{source}, open-ended answers",
    }
