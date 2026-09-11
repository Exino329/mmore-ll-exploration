"""Task: ICD-11, reproducing the earlier GraphRAG-vs-standard-RAG study's setup.

Join the label→description and label→title tables, map each disease to its category, keep
the entries that carry a description, and emit one document per disease holding title,
category and description.

The dataset itself is not shipped: point ``icd11.source`` at your copy. See the README —
the tables behind the earlier study were most likely built from the WHO ICD-11 API rather
than downloaded, which is why the loader is source-agnostic.
"""

import logging
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from mmore.rag.llm import LLM

from .config import BenchConfig, Icd11Config, SourceConfig
from .corpus import CorpusEntry, CorpusStats, read_corpus, write_corpus
from .qa import Question, write_questions

logger = logging.getLogger(__name__)

MASK = "[redacted]"

_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]+")

# WHO chapter letters. I, O and U are unused, so a plain alphabet offset does not work.
_ICD11_CHAPTER_LETTERS = {
    letter: str(10 + position) for position, letter in enumerate("ABCDEFGHJKLMNPQRS")
}


# --- source adapters ---------------------------------------------------------------------


def _read_table(spec: str, kind: str) -> pd.DataFrame:
    if kind == "parquet":
        return pd.read_parquet(spec)
    if kind == "hf":
        from datasets import load_dataset

        parts = spec.split(":")
        dataset_id = parts[0]
        name = parts[1] if len(parts) > 2 else None
        split = parts[-1] if len(parts) > 1 else "train"
        return load_dataset(dataset_id, name, split=split).to_pandas()
    raise ValueError(f"Unknown source kind: {kind}")


def _column(frame: pd.DataFrame, name: str, spec: str) -> pd.Series:
    if name not in frame.columns:
        raise KeyError(
            f"Column '{name}' not found in {spec}. Available: {list(frame.columns)}. "
            "Set icd11.source.columns in the bench config."
        )
    return frame[name].astype("string").fillna("")


def _load_pair(text_spec: str, title_spec: str, source: SourceConfig) -> pd.DataFrame:
    """Outer-join a label→text table with a label→title table."""
    columns = source.columns
    text_frame = _read_table(text_spec, source.kind)
    texts = pd.DataFrame(
        {
            "label": _column(text_frame, columns["label"], text_spec),
            "text": _column(text_frame, columns["text"], text_spec),
        }
    )

    if title_spec == text_spec:
        titles = pd.DataFrame(
            {
                "label": texts["label"],
                "title": _column(text_frame, columns["title"], text_spec),
            }
        )
    else:
        title_frame = _read_table(title_spec, source.kind)
        titles = pd.DataFrame(
            {
                "label": _column(title_frame, columns["label"], title_spec),
                "title": _column(title_frame, columns["title"], title_spec),
            }
        )

    merged = texts.merge(titles, on="label", how="outer")
    merged["label"] = merged["label"].str.strip()
    for field in ("text", "title"):
        merged[field] = merged[field].fillna("").str.strip()
    return merged.drop_duplicates(subset="label")


def _synthetic(size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """A stand-in corpus, for smoke-testing the harness without the ICD-11 tables."""
    systems = ["respiratory", "digestive", "circulatory", "nervous", "endocrine"]
    diseases = [
        {
            "label": f"{i % len(systems) + 1}A{i:03d}",
            "title": f"Synthetic condition {i:03d}",
            "text": (
                f"Synthetic condition {i:03d} is a disorder of the "
                f"{systems[i % len(systems)]} system, characterised by marker {i:03d} "
                f"and by a reaction to agent {i % 37}. It typically follows exposure "
                f"to factor {i % 11} and responds to treatment {i % 7}. Onset is "
                f"{'acute' if i % 2 else 'gradual'} and it mainly affects "
                f"{'adults' if i % 3 else 'children'}."
            ),
        }
        for i in range(size)
    ]
    categories = [
        {
            "label": str(chapter + 1),
            "title": f"Diseases of the {system} system",
            "text": "",
        }
        for chapter, system in enumerate(systems)
    ]
    return pd.DataFrame(diseases), pd.DataFrame(categories)


# --- category mapping ----------------------------------------------------------------------


def category_of(code: str, mapping: str) -> Optional[str]:
    """The category label a disease code belongs to.

    ``study`` is the rule the earlier benchmark used: numeric first character → that
    category, alphabetic → ``13 + alphabet_index``. ``icd11`` is the real WHO chapter
    assignment, which the formula only matches for the numeric chapters.
    """
    if not code:
        return None
    head = code[0].upper()
    if head.isdigit():
        return str(int(head))
    if mapping == "study":
        if not head.isalpha():
            return None
        return str(13 + (ord(head) - ord("A")))
    if head in ("V", "X"):
        return head
    return _ICD11_CHAPTER_LETTERS.get(head)


# --- corpus ---------------------------------------------------------------------------------


def build_corpus(settings: Icd11Config) -> Tuple[List[CorpusEntry], int]:
    source = settings.source
    if source.kind == "synthetic":
        diseases, categories = _synthetic(source.synthetic_size)
    else:
        diseases = _load_pair(source.diseases_text, source.diseases_title, source)
        categories = _load_pair(source.categories_text, source.categories_title, source)

    category_titles = dict(zip(categories["label"], categories["title"]))

    described = diseases[
        diseases["text"].str.len() >= settings.min_description_chars
        if settings.require_description
        else diseases["title"].str.len() > 0
    ]
    logger.info(
        f"{len(described)}/{len(diseases)} entries carry a description of at least "
        f"{settings.min_description_chars} characters"
    )
    if settings.limit is not None:
        described = described.head(settings.limit)

    entries = []
    for row in described.itertuples(index=False):
        code = str(row.label)
        category = category_of(code, settings.category_mapping)
        category_title = category_titles.get(category or "", "")
        entries.append(
            CorpusEntry(
                key=code,
                title=str(row.title) or code,
                body=f"Category: {category_title or 'unspecified'}\n\n{row.text}",
                extra={
                    "icd_code": code,
                    "icd_category": category or "",
                    "icd_category_title": category_title,
                },
            )
        )

    missing = sum(1 for e in entries if not e.extra["icd_category_title"])
    if missing:
        logger.warning(f"{missing} entries have no resolved category")
    # Every ICD-11 document is a gold passage: any of them can be asked about.
    return entries, len(entries)


# --- questions --------------------------------------------------------------------------------

VIGNETTE_SYSTEM = (
    "You write short clinical vignettes used to benchmark medical retrieval systems."
)

VIGNETTE_USER = """Rewrite the description below as a clinical presentation of two or three sentences.

Rules:
- Never name the condition, and never use a synonym, abbreviation or eponym for it.
- Do not name the ICD chapter or category.
- Keep the clinically discriminating details: findings, causative agent, affected system, typical course.
- Paraphrase; do not copy sentences verbatim.
- Return the vignette only, with no preamble.

Condition (do not mention it): {title}
Description:
{description}"""


def _distinctive_terms(entries: List[CorpusEntry], max_document_frequency: int = 3):
    """Title words rare enough across the corpus to give the answer away."""
    frequency: Counter = Counter()
    for entry in entries:
        frequency.update({token.lower() for token in _TOKEN.findall(entry.title)})
    return {
        term for term, count in frequency.items() if count <= max_document_frequency
    }


def _redact(text: str, title: str, distinctive: set) -> str:
    redacted = re.sub(re.escape(title), MASK, text, flags=re.IGNORECASE)
    for token in {t.lower() for t in _TOKEN.findall(title)} & distinctive:
        redacted = re.sub(
            rf"\b{re.escape(token)}\w*", MASK, redacted, flags=re.IGNORECASE
        )
    return re.sub(rf"(?:{re.escape(MASK)}[\s,]*)+", f"{MASK} ", redacted).strip()


def _vignettes(
    entries: List[CorpusEntry], settings: Icd11Config, distinctive: set
) -> List[Optional[str]]:
    llm = LLM.from_config(settings.llm)

    def one(entry: CorpusEntry) -> Optional[str]:
        try:
            response = llm.invoke(
                [
                    ("system", VIGNETTE_SYSTEM),
                    (
                        "human",
                        VIGNETTE_USER.format(title=entry.title, description=entry.body),
                    ),
                ]
            )
        except Exception as error:  # one bad generation must not sink the run
            logger.warning(f"Vignette generation failed for {entry.key}: {error}")
            return None
        vignette = str(getattr(response, "content", response)).strip()
        if not vignette:
            return None
        # The model leaks the name often enough to be worth catching; redacting is
        # cheaper and more reliable than retrying, and keeps the question answerable.
        return (
            "A patient presents as follows.\n\n"
            f"{_redact(vignette, entry.title, distinctive)}\n\n"
            "Which condition is this?"
        )

    with ThreadPoolExecutor(max_workers=settings.max_workers) as pool:
        return list(
            tqdm(
                pool.map(one, entries),
                total=len(entries),
                desc="Generating vignettes",
                unit="q",
            )
        )


def build_questions(
    settings: Icd11Config, entries: List[CorpusEntry]
) -> List[Question]:
    usable = [e for e in entries if e.body.strip() and e.title.strip()]
    if not usable:
        raise ValueError("No corpus entry has both a title and a description.")

    sample = random.Random(settings.seed).sample(
        usable, min(settings.questions, len(usable))
    )
    distinctive = _distinctive_terms(entries)

    if settings.style == "masked":
        texts: List[Optional[str]] = [
            "Which condition matches the following description?\n\n"
            + _redact(entry.body, entry.title, distinctive)
            for entry in sample
        ]
    elif settings.style == "vignette":
        texts = _vignettes(sample, settings, distinctive)
    else:
        raise ValueError(f"Unknown icd11.style: {settings.style}")

    questions = [
        Question(
            query_id=entry.key,
            question=text,
            gold_key=entry.key,
            gold_answer=entry.title,
            style=settings.style,
        )
        for entry, text in zip(sample, texts)
        if text
    ]
    if len(questions) < len(sample):
        logger.warning(f"Dropped {len(sample) - len(questions)} questions that failed.")
    return questions


# --- task interface -----------------------------------------------------------------------------


def prepare_corpus(config: BenchConfig) -> CorpusStats:
    entries, gold = build_corpus(config.icd11)
    return write_corpus(entries, config.corpus_path, gold)


def prepare_questions(config: BenchConfig) -> int:
    return write_questions(
        build_questions(config.icd11, read_corpus(config.corpus_path)), config.qa_path
    )


def describe(config: BenchConfig) -> Dict[str, str]:
    return {
        "task": "ICD-11 disease identification",
        "corpus": f"{config.icd11.source.kind} source, "
        f"category mapping '{config.icd11.category_mapping}'",
        "questions": f"{config.icd11.questions} ({config.icd11.style})",
    }
