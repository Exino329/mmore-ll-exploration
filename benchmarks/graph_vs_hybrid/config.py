"""Benchmark configuration, and the mmore configs derived from it.

One YAML drives the whole run. The mmore stage configs (`index`, `graph-index`,
retriever, RAG) are *generated* from it into the run directory rather than maintained by
hand: the two strategies must agree on the collection, the dense model and k, or the
comparison measures the configs instead of the retrievers.

``task`` selects what is being benchmarked on — the corpus, the questions and how answers
are scored. Everything else (timing, retrieval metrics, report) is task-agnostic.
"""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, get_args, get_origin

import yaml

from mmore.rag.graph.config import MedspacyConfig
from mmore.rag.llm import LLMConfig
from mmore.utils import load_config

GOLD_PREFIX = "gold://"
"""Written into ``metadata.file_path``, the only per-document field both retrievers
return. It is how a retrieved passage is traced back to the entry it came from."""

TASKS = ("pubmedqa", "medhop", "icd11", "hotpotqa")

_FALSE = {"false", "no", "0", "off"}
_TRUE = {"true", "yes", "1", "on"}


def _declared_type(owner: Any, name: str) -> Any:
    """The annotated type of a dataclass field, with ``Optional`` unwrapped.

    The value currently held is not enough to go on: a field defaulting to ``None`` — any
    ``Optional`` one — would otherwise take the override as a bare string.
    """
    for field_info in fields(owner):
        if field_info.name != name:
            continue
        annotation = field_info.type
        if get_origin(annotation) is Union:
            candidates = [a for a in get_args(annotation) if a is not type(None)]
            return candidates[0] if len(candidates) == 1 else None
        return annotation
    return None


def _coerce(raw: str, current: Any, declared: Any = None) -> Any:
    """Cast a command-line string to the type of the config field it targets."""
    if raw.lower() in {"none", "null"}:
        return None

    kind = type(current) if current is not None else declared
    origin = get_origin(kind)
    if origin is not None:  # List[str] and friends
        kind = origin

    if kind is bool:
        if raw.lower() not in _TRUE | _FALSE:
            raise ValueError(f"Expected a boolean, got {raw!r}")
        return raw.lower() in _TRUE
    if kind is int:
        return int(raw)
    if kind is float:
        return float(raw)
    if kind is list:
        items = [item.strip() for item in raw.split(",") if item.strip()]
        # The element type matters: `--set retrieval.ks=1,5,50` on a List[int] silently
        # produced strings, and every `predicted[:k]` downstream then raised on them.
        element = (
            type(current[0])
            if isinstance(current, list) and current
            else next(iter(get_args(declared) or ()), None)
        )
        if element in (bool, int, float):
            return [_coerce(item, None, element) for item in items]
        return items
    return raw


# --- task: PubMedQA -----------------------------------------------------------------------


@dataclass
class PubmedqaConfig:
    """Expert-annotated PubMed abstracts, hidden in a pool of distractors.

    Each question carries two golds: the abstract it was written from (retrieval) and a
    yes/no/maybe decision (generation). Both are scored without an LLM.
    """

    gold_split: str = "pqa_labeled"
    """The 1,000 expert-annotated instances. Their abstracts are the gold passages."""

    distractor_split: str = "pqa_artificial"
    """Abstracts added to the corpus that no question points at. Streamed, so only the
    requested number is downloaded. ``pqa_unlabeled`` (61k) is the smaller alternative."""

    distractors: int = 19_000
    """Corpus size is ``1,000 + distractors``. Retrieval is near-trivial on a corpus of
    1,000; this is the knob that makes it a search problem."""

    questions: int = 500
    seed: int = 13

    include_mesh: bool = True
    """Append the MeSH terms to the indexed passage. They are the closest thing the corpus
    has to curated entity annotations, and the graph builder extracts entities from text."""


# --- task: MedHop -------------------------------------------------------------------------


@dataclass
class MedhopConfig:
    """Multi-hop drug-interaction questions over Medline abstracts."""

    question_split: str = "validation"
    """The split the questions come from. Its supports are the reachable gold passages."""

    distractor_split: str = "train"
    """Supports pooled into the corpus that no asked question points at. Without them the
    corpus is 6k documents of which ~22 are relevant per question, which is not a search
    problem. Set to an empty string to index only the question split."""

    questions: Optional[int] = None
    """Subsample the questions. ``null`` asks all of them (342 in validation)."""

    seed: int = 13

    retrieval_query: str = "full"
    """What the retriever is asked. ``full`` sends the question as the generator sees it,
    nine candidate accessions included — and since the gold answer is always among them,
    the hop-2 passage is reachable by copying a string out of the question. ``subject_only``
    sends the subject accession alone, which is what makes hop-2 require bridging. The
    answer stage always sees the full question. Changing this needs `prepare --qa-only`,
    not a re-index."""

    def __post_init__(self):
        allowed = ("full", "subject_only")
        if self.retrieval_query not in allowed:
            raise ValueError(
                f"Unknown medhop.retrieval_query: {self.retrieval_query}. "
                f"Expected one of {allowed}."
            )


# --- task: HotpotQA -----------------------------------------------------------------------


@dataclass
class HotpotqaConfig:
    """Multi-hop Wikipedia questions, on the setting LinearRAG publishes results for.

    See :mod:`.hotpotqa` for why the corpus is rebuilt from the HotpotQA dev contexts
    instead of taken from LinearRAG's ``chunks.json``.
    """

    question_source: str = "linearrag"
    """``linearrag`` asks their exact 1,000-question subsample, which is what makes the
    answer numbers comparable to their table. ``dev`` samples from the 7,405 validation
    questions instead, for a run that is bigger than theirs."""

    questions: Optional[int] = None
    """Subsample the questions. ``null`` asks all of them (1,000 under ``linearrag``)."""

    seed: int = 13

    distractor_questions: int = 0
    """Contexts of that many further dev questions, pooled into the corpus without their
    questions being asked. 0 keeps the 9,811 articles the asked questions bring, which is
    the size the HotpotQA retrieval literature reports on; raising it is how the corpus
    becomes a harder search problem without changing what is scored."""

    include_title: bool = True
    """Prepend the article title to the indexed text, as HippoRAG and LinearRAG do. It is
    also the entity a bridge question names, so with it on, hop-1 is partly addressable by
    literal string match — the same trade-off `medhop.retrieval_query` documents."""

    def __post_init__(self):
        allowed = ("linearrag", "dev")
        if self.question_source not in allowed:
            raise ValueError(
                f"Unknown hotpotqa.question_source: {self.question_source}. "
                f"Expected one of {allowed}."
            )


# --- task: ICD-11 -------------------------------------------------------------------------


@dataclass
class SourceConfig:
    """Where the ICD-11 tables come from.

    ``kind: parquet`` expects the layout of the original study: one table mapping labels
    to descriptive texts and another mapping labels to titles, for diseases and again for
    categories. ``kind: hf`` takes ``"dataset_id[:config][:split]"`` in the same four
    slots. When the same path/spec is given for the text and title tables, it is read once
    and both columns are taken from it.

    ``kind: synthetic`` fabricates a small corpus; it exists so the harness can be smoke
    tested without the dataset and produces meaningless accuracy numbers.
    """

    kind: str = "parquet"
    diseases_text: str = ""
    diseases_title: str = ""
    categories_text: str = ""
    categories_title: str = ""
    columns: Dict[str, str] = field(
        default_factory=lambda: {"label": "label", "text": "text", "title": "title"}
    )
    synthetic_size: int = 200


@dataclass
class Icd11Config:
    source: SourceConfig = field(default_factory=SourceConfig)

    category_mapping: str = "study"
    """``study`` reproduces the previous benchmark's rule (numeric label → that category,
    alphabetic label → ``13 + alphabet_index``). ``icd11`` uses the real WHO chapter
    letters, which disagree with the formula from chapter 10 on. The category title ends
    up in the indexed text, so the choice is not cosmetic."""

    require_description: bool = True
    """Keep only entries that carry a description (6,641 of 34,663 in the original run).
    Entries without one give the graph builder no text to extract entities from."""

    min_description_chars: int = 80
    limit: Optional[int] = None
    """Truncate the corpus, for quick runs. ``null`` keeps everything."""

    style: str = "vignette"
    """How a question is made from an entry. ``vignette`` asks an LLM to rewrite the
    description as a clinical presentation with the disease unnamed. ``masked`` blanks the
    title out of the description with no LLM — free and deterministic, but the question
    then shares most of its wording with the indexed passage, which flatters dense
    retrieval."""

    questions: int = 300
    seed: int = 13
    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            llm_name="gpt-4o-mini", max_new_tokens=400, temperature=0.0
        )
    )
    max_workers: int = 8


# --- shared stages ---------------------------------------------------------------------------


@dataclass
class IndexBenchConfig:
    collection_name: str = "bench"
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    sparse_model: str = "splade"
    is_multimodal: bool = False


@dataclass
class GraphBenchConfig:
    """Index- and query-time graph settings. Defaults are the paper's."""

    spacy_model: str = "en_core_web_sm"
    normalize_entities: bool = True
    batch_size: int = 128
    n_process: int = 1

    ner_backend: str = "spacy"
    """``spacy`` runs ``spacy_model`` alone; ``medspacy`` wraps it in medspaCy's clinical
    components. The two are orthogonal — the NER model is still ``spacy_model``, since
    medspaCy contributes assertion and sentence splitting, not a model."""

    medspacy: MedspacyConfig = field(default_factory=MedspacyConfig)
    """Only read when ``ner_backend: medspacy``. ``medspacy.drop_asserted`` is the knob
    worth sweeping: it decides whether ConText-negated entities stay in the graph."""

    use_gpu: Optional[bool] = None
    """Run spaCy's NER on the GPU. Only transformer pipelines gain much, and it needs
    cupy installed alongside spaCy."""

    artifacts_name: str = "graph_index"
    """Subdirectory of the run holding this graph. Give a second graph its own name to
    compare index-time settings — a different NER model, say — against the *same* Milvus
    collection, instead of rebuilding the collection or overwriting the first graph."""

    max_iterations: int = 3
    top_k_sentence: int = 1
    iteration_threshold: float = 0.4
    passage_ratio: float = 0.05
    passage_node_weight: float = 0.05
    damping: float = 0.5
    seed_min_similarity: float = 0.5
    dense_candidates: int = 200

    activation_merge: str = "overwrite"
    """``overwrite`` reproduces the reference implementation, where a later hop replaces
    what was known about an entity it re-reaches; ``best`` keeps the max score and min tier.
    Defaults to the reference behaviour so existing results stay reproducible."""


@dataclass
class RetrievalBenchConfig:
    ks: List[int] = field(default_factory=lambda: [1, 3, 5, 10])

    reranker_model_name: Optional[str] = None
    """Off by default: the reranker reorders the top-k of *both* strategies with the same
    cross-encoder, which hides the difference the benchmark is trying to measure. Set it
    to compare the deployed configuration instead of the retrieval strategies."""

    warmup_queries: int = 3

    fusion_depth: int = 100
    """How deep each part of an ``a+b`` strategy is queried before reciprocal-rank fusion.
    Deeper than the final k on purpose: a passage the graph ranks 40th and dense ranks 3rd
    is exactly the case fusion exists to catch, and a depth of k would never see it."""

    fusion_constant: float = 60.0
    """The ``k`` of reciprocal-rank fusion, ``1 / (constant + rank)``. 60 is the value from
    the original RRF paper and is left alone unless a sweep says otherwise: it flattens the
    difference between the top ranks, so no single part can dominate on its first hit."""
    """Excluded from the latency statistics: the first queries pay for lazy model loads."""


@dataclass
class AnswerBenchConfig:
    scoring: str = "exact_label"
    """``exact_label`` matches the prediction against ``labels`` — free, deterministic, and
    what PubMedQA's yes/no/maybe decision calls for. ``contain`` checks that the gold span
    appears in the answer the model committed to, which is LinearRAG's own metric and the
    one their HotpotQA numbers are reported in — also free. ``judge`` asks an LLM to rule
    correct/incorrect, as LinearRAG's ``src/evaluate.py`` does, for open-ended answers."""

    labels: List[str] = field(default_factory=lambda: ["yes", "no", "maybe"])

    llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            llm_name="gpt-4o-mini", max_new_tokens=256, temperature=0.0
        )
    )
    judge_llm: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            llm_name="gpt-4o-mini", max_new_tokens=16, temperature=0.0
        )
    )
    k: int = 5
    max_workers: int = 8
    system_prompt: str = (
        "You are answering a biomedical research question using the retrieved abstracts "
        "below. Reply with exactly one word: yes, no, or maybe.\n\nContext:\n{context}"
    )


@dataclass
class BenchConfig:
    task: str = "pubmedqa"
    run_dir: str = "benchmarks/.runs/pubmedqa"

    pubmedqa: PubmedqaConfig = field(default_factory=PubmedqaConfig)
    medhop: MedhopConfig = field(default_factory=MedhopConfig)
    icd11: Icd11Config = field(default_factory=Icd11Config)
    hotpotqa: HotpotqaConfig = field(default_factory=HotpotqaConfig)

    index: IndexBenchConfig = field(default_factory=IndexBenchConfig)
    graph: GraphBenchConfig = field(default_factory=GraphBenchConfig)
    retrieval: RetrievalBenchConfig = field(default_factory=RetrievalBenchConfig)
    answer: AnswerBenchConfig = field(default_factory=AnswerBenchConfig)

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"Unknown task: {self.task}. Expected one of {TASKS}.")

    @classmethod
    def load(cls, path: str) -> "BenchConfig":
        return load_config(path, cls)

    def override(self, assignments: List[str]) -> "BenchConfig":
        """Apply ``dotted.path=value`` overrides in place.

        Sweeping a query-time graph parameter needs no re-indexing, so this is how a
        parameter study is run: same collection, same graph, one `retrieval` pass per
        setting. The value is cast to the type the field already holds.
        """
        for assignment in assignments:
            if "=" not in assignment:
                raise ValueError(f"Expected 'path.to.field=value', got {assignment!r}")
            path, raw = assignment.split("=", 1)

            target: Any = self
            parts = path.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise AttributeError(f"No config field named '{path}'")

            setattr(
                target,
                leaf,
                _coerce(raw, getattr(target, leaf), _declared_type(target, leaf)),
            )
        return self

    # --- run layout --------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return Path(self.run_dir)

    @property
    def corpus_path(self) -> Path:
        return self.root / "corpus.jsonl"

    @property
    def qa_path(self) -> Path:
        return self.root / "qa.jsonl"

    @property
    def milvus_uri(self) -> str:
        return str(self.root / "milvus.db")

    @property
    def graph_artifacts_dir(self) -> str:
        return str(self.root / self.graph.artifacts_name)

    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    def results_path(self, name: str) -> Path:
        return self.root / "results" / name


# --- generated mmore configs -------------------------------------------------------------

_DB = "db"


def _db(config: BenchConfig) -> Dict[str, Any]:
    return {"uri": config.milvus_uri, "name": "bench"}


def _indexer(config: BenchConfig) -> Dict[str, Any]:
    return {
        "dense_model": {
            "model_name": config.index.dense_model,
            "is_multimodal": config.index.is_multimodal,
        },
        "sparse_model": {
            "model_name": config.index.sparse_model,
            "is_multimodal": False,
        },
        _DB: _db(config),
    }


def _ner_settings(config: BenchConfig) -> Dict[str, Any]:
    """The entity-extraction half of the graph config, shared by both stages.

    Index and query time must agree on it: query entities are linked to corpus entities by
    embedding similarity, so extracting them differently compares two vocabularies.
    """
    settings: Dict[str, Any] = {
        "spacy_model": config.graph.spacy_model,
        "ner_backend": config.graph.ner_backend,
    }
    if config.graph.ner_backend == "medspacy":
        settings["medspacy"] = asdict(config.graph.medspacy)
    return settings


def _retriever(config: BenchConfig, strategy: str, k: int) -> Dict[str, Any]:
    retriever: Dict[str, Any] = {
        "type": strategy,
        _DB: _db(config),
        "collection_name": config.index.collection_name,
        "hybrid_search_weight": 0.5,
        "k": k,
        "use_web": False,
        "reranker_model_name": config.retrieval.reranker_model_name,
    }
    if strategy == "graph":
        retriever["graph"] = {
            "artifacts_dir": config.graph_artifacts_dir,
            # Written out rather than left to default from the manifest, so that query-time
            # entity extraction is sweepable with --set like every other graph knob.
            **_ner_settings(config),
            "max_iterations": config.graph.max_iterations,
            "top_k_sentence": config.graph.top_k_sentence,
            "iteration_threshold": config.graph.iteration_threshold,
            "passage_ratio": config.graph.passage_ratio,
            "passage_node_weight": config.graph.passage_node_weight,
            "damping": config.graph.damping,
            "seed_min_similarity": config.graph.seed_min_similarity,
            "dense_candidates": config.graph.dense_candidates,
            "activation_merge": config.graph.activation_merge,
        }
    return retriever


def write_stage_configs(config: BenchConfig) -> Dict[str, Path]:
    """Materialize the mmore configs for this run and return their paths."""
    config.configs_dir.mkdir(parents=True, exist_ok=True)

    max_k = max(config.retrieval.ks)
    documents: Dict[str, Dict[str, Any]] = {
        "index": {
            "indexer": _indexer(config),
            "collection_name": config.index.collection_name,
            "documents_path": str(config.corpus_path),
        },
        "graph_index": {
            "indexer": _indexer(config),
            "collection_name": config.index.collection_name,
            "graph": {
                "artifacts_dir": config.graph_artifacts_dir,
                **_ner_settings(config),
                "normalize_entities": config.graph.normalize_entities,
                "batch_size": config.graph.batch_size,
                "n_process": config.graph.n_process,
                "use_gpu": config.graph.use_gpu,
                "rebuild": True,
            },
        },
        "retriever_hybrid": _retriever(config, "hybrid", max_k),
        "retriever_graph": _retriever(config, "graph", max_k),
    }

    for strategy in ("hybrid", "graph"):
        documents[f"rag_{strategy}"] = {
            "retriever": _retriever(config, strategy, config.answer.k),
            "llm": asdict(config.answer.llm),
            "system_prompt": config.answer.system_prompt,
        }

    paths = {}
    for name, document in documents.items():
        path = config.configs_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        paths[name] = path
    return paths
