"""Configuration objects for the LinearRAG-style graph retrieval strategy.

This module deliberately imports nothing from ``mmore.rag.retriever``: ``RetrieverConfig``
needs to reference the retriever-side config, while ``graph.retriever`` needs to subclass
``Retriever``. Keeping the configs isolated here breaks that cycle.

Reference: *LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora*
(arXiv 2510.10114). Defaults follow the per-dataset presets published in the authors'
``scripts/run.sh``; the two fields absent from the paper are marked below.
"""

from dataclasses import dataclass, field
from typing import List, Optional

# spaCy entity labels dropped at both index and query time, as in the reference
# implementation: pure numbers carry no linking signal and blow up the entity vocabulary.
DEFAULT_EXCLUDED_LABELS = ["ORDINAL", "CARDINAL"]

NER_BACKENDS = ("spacy", "medspacy")

# How a re-reached entity is recorded during semantic bridging. See
# ``GraphRetrieverConfig.activation_merge``.
ACTIVATION_MERGES = ("overwrite", "best")

# medspaCy components added on top of the spaCy pipeline. ConText is the reason to reach
# for medspaCy at all; PyRuSH replaces the parser's sentence segmentation.
DEFAULT_MEDSPACY_COMPONENTS = ["medspacy_pyrush", "medspacy_context"]

# ConText assertion attributes, set by ``medspacy_context`` on every entity.
CONTEXT_ATTRIBUTES = (
    "is_negated",
    "is_uncertain",
    "is_historical",
    "is_hypothetical",
    "is_family",
)


@dataclass
class MedspacyConfig:
    """Options for ``ner_backend: medspacy``. Ignored by the plain spaCy backend.

    medspaCy is not an NER model: on a blank pipeline its ``TargetMatcher`` starts with no
    rules and extracts nothing. It is a set of *clinical* components layered on top of a
    spaCy pipeline, so ``spacy_model`` still chooses what recognizes entities (scispaCy's
    ``en_core_sci_md``, say) and this only decides what happens to them afterwards.
    """

    components: List[str] = field(
        default_factory=lambda: list(DEFAULT_MEDSPACY_COMPONENTS)
    )
    """medspaCy components to enable, from ``medspacy.util.ALL_PIPE_NAMES``:
    ``medspacy_pyrush`` (clinical sentence splitter), ``medspacy_context`` (assertion),
    ``medspacy_target_matcher`` (rule-based entities), ``medspacy_sectionizer``,
    ``medspacy_quickumls`` (UMLS concept matching), ``medspacy_tokenizer`` (a more
    aggressive tokenizer — note it replaces the one the base model's NER was trained
    with), ``medspacy_preprocessor``, ``medspacy_postprocessor``,
    ``medspacy_doc_consumer``."""

    drop_asserted: List[str] = field(default_factory=list)
    """ConText attributes that disqualify an entity, from :data:`CONTEXT_ATTRIBUTES`.
    ``["is_negated"]`` keeps "no evidence of macular edema" from putting *macular edema*
    in the graph as if the passage were about it. Empty by default: ConText's scope rules
    were written for clinical notes and over-extend on academic prose, so dropping is a
    setting to sweep rather than a free win. Requires ``medspacy_context``."""

    target_rules_path: Optional[str] = None
    """JSON file of medspaCy ``TargetRule`` entries loaded into
    ``medspacy_target_matcher``. This is how medspaCy extracts entities *itself*, from a
    lexicon, instead of only annotating the base model's."""

    quickumls_path: Optional[str] = None
    """QuickUMLS resource directory for ``medspacy_quickumls``. ``None`` falls back to the
    sample dictionary medspaCy ships, which is a demo and matches almost nothing; a real
    one has to be built from a UMLS subscription."""

    load_rules: bool = True
    """Load medspaCy's default ConText and section rule sets."""

    def __post_init__(self):
        # A typo here is otherwise silent: an attribute that does not exist reads as False
        # on every entity, so nothing is ever dropped and the run looks like a baseline.
        unknown = [a for a in self.drop_asserted if a not in CONTEXT_ATTRIBUTES]
        if unknown:
            raise ValueError(
                f"Unknown ConText attributes in drop_asserted: {unknown}. "
                f"Expected any of {list(CONTEXT_ATTRIBUTES)}."
            )
        if self.drop_asserted and "medspacy_context" not in self.components:
            raise ValueError(
                "drop_asserted needs the 'medspacy_context' component to set the "
                "attributes it filters on."
            )


@dataclass
class GraphBuildConfig:
    """Index-time configuration for building the Tri-Graph."""

    artifacts_dir: str = "./graph_index"
    """Root directory of the graph artifacts. Each collection gets a subdirectory."""

    spacy_model: str = "en_core_web_sm"
    """spaCy pipeline used for entity extraction. ``en_core_web_trf`` is more accurate;
    ``en_core_sci_scibert`` (scispaCy) targets biomedical corpora. The model must provide
    both an NER component and sentence boundaries."""

    ner_backend: str = "spacy"
    """``spacy`` runs ``spacy_model`` on its own. ``medspacy`` wraps that same pipeline in
    medspaCy's clinical components — see :class:`MedspacyConfig`. The model choice is
    independent of the backend."""

    medspacy: MedspacyConfig = field(default_factory=MedspacyConfig)

    normalize_entities: bool = True
    """Merge entity surface forms that differ only by case/whitespace. Deviation from the
    reference implementation, which is case-sensitive at index time but lowercases query
    entities, silently losing matches."""

    min_entity_length: int = 2
    """Drop entity surface forms shorter than this (after stripping)."""

    excluded_entity_labels: List[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_LABELS)
    )

    batch_size: int = 128
    """Batch size for both ``nlp.pipe`` and the embedding model."""

    n_process: int = 1
    """Number of spaCy worker processes. Keep at 1 for transformer pipelines on GPU."""

    use_gpu: Optional[bool] = None
    """Run spaCy on the GPU. ``None`` takes it when available, ``True`` fails if it is
    not, ``False`` forces CPU. Only transformer pipelines gain much — measured on 20k
    PubMed abstracts, ``en_core_sci_scibert`` runs at 2.9 docs/s on six CPU cores. spaCy
    reaches the GPU through cupy, which `--extra graph` does not install: add the build
    matching your CUDA version, e.g. ``cupy-cuda12x``."""

    max_doc_length: int = 1_000_000
    """Upper bound on characters handed to spaCy for a single chunk (``nlp.max_length``)."""

    rebuild: bool = False
    """Ignore existing artifacts and rebuild the graph from scratch."""

    def __post_init__(self):
        if self.ner_backend not in NER_BACKENDS:
            raise ValueError(
                f"Unknown ner_backend: {self.ner_backend}. Expected one of {NER_BACKENDS}."
            )


@dataclass
class GraphRetrieverConfig:
    """Query-time configuration for graph retrieval."""

    artifacts_dir: str = "./graph_index"
    """Must match the ``artifacts_dir`` used by ``mmore graph-index``."""

    spacy_model: Optional[str] = None
    """Entity extractor for the query. Defaults to the model recorded in the manifest, so
    query-time and index-time entity extraction stay consistent."""

    ner_backend: Optional[str] = None
    """Defaults to the backend recorded in the manifest, for the same reason: query
    entities are matched against corpus entities, so both sides must be extracted the
    same way."""

    medspacy: Optional[MedspacyConfig] = None
    """Defaults to the medspaCy options recorded in the manifest. Overriding it is mostly
    useful to assert *query* entities differently from corpus ones — dropping negated
    entities from a question is not the same decision as dropping them from a passage."""

    excluded_entity_labels: List[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_LABELS)
    )

    max_iterations: int = 3
    """Number of semantic-bridging hops. The paper uses 3 (5 for MuSiQue)."""

    top_k_sentence: int = 1
    """Sentences kept per activated entity at each hop."""

    iteration_threshold: float = 0.4
    """Entities scoring below this are not expanded and do not join the frontier."""

    activation_merge: str = "overwrite"
    """How an entity reached more than once during bridging is recorded.

    ``overwrite`` is the reference implementation: the last hop to reach an entity replaces
    what was known about it, so a seed re-reached at hop 2 keeps the weaker propagated score
    and the higher tier. ``score_passages`` then divides its contribution by that tier,
    penalising twice over the entity the question is actually about — on 40 MedHop questions,
    91% of seeds end up demoted this way.

    ``best`` keeps the maximum score and the minimum tier instead. It changes bookkeeping
    only: the frontier still expands on the freshly propagated scores, so the walk reaches
    exactly the same entities. Not in the reference implementation."""

    passage_ratio: float = 0.05
    """Weight of the dense-retrieval term in the passage prior."""

    passage_node_weight: float = 0.05
    """Global scaling of passage nodes in the personalized-PageRank reset vector."""

    damping: float = 0.5
    """PageRank damping factor: the probability of following an edge rather than
    teleporting back to the reset vector, so a higher value diffuses further from the
    query's seeds.

    0.5 is the reference implementation's default (``src/config.py``), which is what this
    port follows. The paper's text quotes the conventional 0.85; the two disagree, and
    which one produced the published numbers is not recoverable from the release."""

    seed_min_similarity: float = 0.5
    """Not in the reference implementation: minimum cosine similarity for a query entity
    to be linked to a corpus entity. Upstream takes an unconditional argmax, so an
    unrelated query entity still injects a high-weight seed into the reset vector."""

    dense_candidates: int = 200
    """Not in the reference implementation: size of the dense-retrieval candidate pool
    used to build the passage prior. Upstream scores every passage in the corpus on every
    query. Passages outside the pool still receive PageRank mass through their entity
    edges, so this bounds the reset vector, not the result set."""

    def __post_init__(self):
        if self.ner_backend is not None and self.ner_backend not in NER_BACKENDS:
            raise ValueError(
                f"Unknown ner_backend: {self.ner_backend}. Expected one of {NER_BACKENDS}."
            )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.top_k_sentence < 1:
            raise ValueError("top_k_sentence must be >= 1")
        if self.activation_merge not in ACTIVATION_MERGES:
            raise ValueError(
                f"Unknown activation_merge: {self.activation_merge}. "
                f"Expected one of {list(ACTIVATION_MERGES)}."
            )
        if not 0.0 < self.damping < 1.0:
            raise ValueError("damping must lie in (0, 1)")
