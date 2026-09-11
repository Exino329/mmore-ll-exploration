"""End-to-end answer accuracy.

Two scoring modes, chosen by ``answer.scoring``:

``exact_label``
    The answer is one of a closed set (PubMedQA's yes/no/maybe). Deterministic, free, and
    reported next to the majority-class baseline — 55% of PubMedQA's labelled decisions
    are "yes", so an accuracy figure means nothing on its own.

``contain``
    The answer is open-ended (a HotpotQA span) and scored by LinearRAG's ``contain`` rule
    alone: the normalised gold appears in the model's committed answer. Deterministic and
    free, and directly comparable to the ``contain`` column of their published table.

``judge``
    The answer is open-ended (a disease name). Scored the way LinearRAG's
    ``src/evaluate.py`` does it: ``contain`` accuracy plus an LLM ruling correct/incorrect,
    with its prompt.

Generation runs sequentially so the per-question latency is real and so the retriever —
which holds a spaCy pipeline and a Milvus client — is never driven from two threads; only
judging is parallel, since that is pure API latency.
"""

import json
import logging
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from mmore.rag.llm import LLM
from mmore.rag.pipeline import RAGConfig, RAGPipeline
from mmore.utils import load_config

from .collection import key_by_chunk_id
from .config import BenchConfig, write_stage_configs
from .qa import Question, contains_answer, extract_label, final_answer, read_questions

logger = logging.getLogger(__name__)

STRATEGIES = ("hybrid", "graph")

JUDGE_SYSTEM = "You are an expert evaluator."

JUDGE_USER = """Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {prediction}
Gold answer: {gold}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:"""


def _doc_metadata(doc: Any) -> Dict[str, Any]:
    return doc.get("metadata", {}) if isinstance(doc, dict) else doc.metadata


def _generate(
    strategy: str,
    config: BenchConfig,
    questions: List[Question],
    keys: Dict[str, str],
    config_path: Path,
) -> List[Dict[str, Any]]:
    pipeline = RAGPipeline.from_config(load_config(str(config_path), RAGConfig))

    records = []
    for question in tqdm(questions, desc=f"Answering [{strategy}]", unit="q"):
        started = time.perf_counter()
        result = pipeline(
            queries={
                "input": question.question,
                "collection_name": config.index.collection_name,
            },
            return_dict=True,
        )[0]
        elapsed = time.perf_counter() - started

        retrieved = [
            keys.get(_doc_metadata(doc).get("id", "")) for doc in result.get("docs", [])
        ]
        records.append(
            {
                "query_id": question.query_id,
                "question": question.question,
                "gold_answer": question.gold_answer,
                "gold_keys": question.gold_keys,
                "choices": question.choices,
                "pred_answer": result.get("answer", ""),
                "retrieved_keys": [key for key in retrieved if key],
                "gold_retrieved": any(key in retrieved for key in question.gold_keys),
                **{
                    f"{group}_retrieved": any(key in retrieved for key in members)
                    for group, members in question.gold_groups.items()
                },
                "seconds": elapsed,
            }
        )
    return records


def _score_exact_label(records: List[Dict[str, Any]], config: BenchConfig) -> None:
    for record in records:
        # MedHop's nine candidates differ per question; yes/no/maybe does not.
        labels = record.get("choices") or config.answer.labels
        predicted = extract_label(record["pred_answer"], labels)
        record["pred_label"] = predicted
        record["accuracy"] = float(predicted == record["gold_answer"].strip().lower())
        record["contain_accuracy"] = float(
            contains_answer(record["pred_answer"], record["gold_answer"])
        )


def _score_contain(records: List[Dict[str, Any]], config: BenchConfig) -> None:
    """LinearRAG's ``contain`` accuracy, and nothing else.

    Their ``src/evaluate.py`` reports two figures, ``contain`` and an LLM verdict; only the
    first is deterministic and free, and it is the one their published table can be checked
    against without buying a second opinion from a judge. ``judge`` remains available and
    computes both.
    """
    for record in records:
        prediction = final_answer(record["pred_answer"])
        record["pred_final"] = prediction
        record["accuracy"] = float(contains_answer(prediction, record["gold_answer"]))
        record["contain_accuracy"] = record["accuracy"]


def _score_judge(records: List[Dict[str, Any]], config: BenchConfig) -> None:
    llm = LLM.from_config(config.answer.judge_llm)

    def verdict(record: Dict[str, Any]) -> float:
        try:
            response = llm.invoke(
                [
                    ("system", JUDGE_SYSTEM),
                    (
                        "human",
                        JUDGE_USER.format(
                            prediction=record["pred_answer"], gold=record["gold_answer"]
                        ),
                    ),
                ]
            )
        except Exception as error:
            logger.warning(f"Judge failed on {record['query_id']}: {error}")
            return float("nan")
        content = str(getattr(response, "content", response)).strip().lower()
        return 1.0 if content.startswith("correct") else 0.0

    with ThreadPoolExecutor(max_workers=config.answer.max_workers) as pool:
        verdicts = list(
            tqdm(
                pool.map(verdict, records), total=len(records), desc="Judging", unit="q"
            )
        )
    for record, score in zip(records, verdicts):
        record["accuracy"] = score
        record["contain_accuracy"] = float(
            contains_answer(record["pred_answer"], record["gold_answer"])
        )


def _majority_baseline(records: List[Dict[str, Any]]) -> float:
    """Accuracy of always answering the most frequent gold label.

    PubMedQA's labelled split is 55% "yes"; without this the accuracy column reads as
    better than it is.
    """
    if not records:
        return 0.0
    # With per-question candidate sets the majority class is meaningless; the informative
    # floor is picking uniformly at random among the candidates.
    if records[0].get("choices"):
        return sum(1 / len(r["choices"]) for r in records if r["choices"]) / len(
            records
        )
    golds = Counter(r["gold_answer"].strip().lower() for r in records)
    return golds.most_common(1)[0][1] / len(records)


def evaluate_strategy(
    strategy: str,
    config: BenchConfig,
    questions: List[Question],
    keys: Dict[str, str],
    config_path: Path,
) -> Dict[str, Any]:
    records = _generate(strategy, config, questions, keys, config_path)

    if config.answer.scoring == "exact_label":
        _score_exact_label(records, config)
    elif config.answer.scoring == "contain":
        _score_contain(records, config)
    elif config.answer.scoring == "judge":
        _score_judge(records, config)
    else:
        raise ValueError(f"Unknown answer.scoring: {config.answer.scoring}")

    scored = [r["accuracy"] for r in records if not np.isnan(r["accuracy"])]
    total = len(records) or 1
    summary = {
        "strategy": strategy,
        "scoring": config.answer.scoring,
        "questions": len(records),
        "k": config.answer.k,
        "accuracy": float(np.mean(scored)) if scored else 0.0,
        "scored": len(scored),
        "majority_baseline": _majority_baseline(records),
        "contain_accuracy": sum(r["contain_accuracy"] for r in records) / total,
        f"gold_retrieved_at_{config.answer.k}": sum(
            1 for r in records if r["gold_retrieved"]
        )
        / total,
        **{
            f"{key}_at_{config.answer.k}": sum(1 for r in records if r.get(key)) / total
            for key in sorted(
                {k for r in records for k in r if k.endswith("_retrieved")}
            )
            if key != "gold_retrieved"
        },
        "unparsed_answers": sum(1 for r in records if r.get("pred_label") is None)
        if config.answer.scoring == "exact_label"
        else 0,
        "latency_mean_s": float(np.mean([r["seconds"] for r in records]))
        if records
        else 0.0,
    }

    path = config.results_path(f"answers_{strategy}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"summary": summary, "answers": records}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"[{strategy}] accuracy={summary['accuracy']:.3f} "
        f"(majority baseline {summary['majority_baseline']:.3f}) → {path}"
    )
    return summary


def run(
    config: BenchConfig, strategies: List[str] = list(STRATEGIES)
) -> Dict[str, Any]:
    if not config.qa_path.exists():
        raise FileNotFoundError(
            f"{config.qa_path} not found. Run the `prepare` stage first."
        )

    paths = write_stage_configs(config)
    questions = read_questions(config.qa_path)
    keys = key_by_chunk_id(config)

    summaries = {
        strategy: evaluate_strategy(
            strategy, config, questions, keys, paths[f"rag_{strategy}"]
        )
        for strategy in strategies
    }

    path = config.results_path("answers.json")
    path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries
