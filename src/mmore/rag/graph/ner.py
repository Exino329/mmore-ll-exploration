"""Entity extraction for the Tri-Graph.

Port of the reference implementation's ``src/ner.py``. This is the only model that runs at
index time: LinearRAG's whole point is that graph construction is relation-free and
therefore consumes zero LLM tokens.

Two backends produce the same :class:`DocumentEntities`, selected by
``ner_backend``. :class:`SpacyEntityExtractor` runs a spaCy pipeline on its own;
:class:`MedspacyEntityExtractor` wraps that same pipeline in medspaCy's clinical
components. The spaCy model is chosen independently of the backend, so a graph can be
rebuilt with a different NER without touching anything else.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .config import (
    DEFAULT_EXCLUDED_LABELS,
    NER_BACKENDS,
    GraphBuildConfig,
    MedspacyConfig,
)

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def normalize_entity(surface: str) -> str:
    """Canonical key under which two surface forms are considered the same entity."""
    return _WHITESPACE.sub(" ", surface).strip().lower()


@dataclass
class DocumentEntities:
    """Entities extracted from a single chunk."""

    entities: List[str] = field(default_factory=list)
    """Distinct entity surface forms occurring anywhere in the chunk."""

    sentences: List[Tuple[str, List[str]]] = field(default_factory=list)
    """``(sentence_text, entity_surface_forms)`` for every sentence carrying at least one
    entity. Sentences without entities are dropped: they can never bridge two entities."""


class SpacyEntityExtractor:
    """Batched spaCy NER producing chunk-level and sentence-level entity sets."""

    def __init__(
        self,
        model_name: str,
        excluded_labels: Optional[Sequence[str]] = None,
        batch_size: int = 128,
        n_process: int = 1,
        max_doc_length: int = 1_000_000,
        min_entity_length: int = 2,
        use_gpu: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.excluded_labels = set(
            excluded_labels if excluded_labels is not None else DEFAULT_EXCLUDED_LABELS
        )
        self.batch_size = batch_size
        self.n_process = n_process
        self.min_entity_length = min_entity_length
        self.nlp = self._build_pipeline(model_name, use_gpu)
        self.nlp.max_length = max_doc_length

    @staticmethod
    def _activate_gpu(use_gpu: Optional[bool]) -> bool:
        """Move spaCy onto the GPU, before any pipeline is loaded.

        Only worth it for transformer pipelines (``en_core_sci_scibert`` and friends),
        where it is the difference between minutes and hours; the CNN pipelines are
        CPU-bound by design and see little benefit. spaCy reaches the GPU through cupy,
        which is a separate install (``cupy-cuda12x``), so this is a no-op on a plain
        ``--extra graph`` environment.
        """
        if use_gpu is False:
            return False

        import spacy

        activated = spacy.prefer_gpu()  # False when cupy is missing or no device
        if not activated:
            if use_gpu:
                raise RuntimeError(
                    "spacy_use_gpu is set but spaCy cannot reach a GPU. It needs cupy "
                    "matching your CUDA version, e.g. `uv pip install cupy-cuda12x`."
                )
            return False

        # Transformer pipelines run their weights under PyTorch. Handing cupy the same
        # allocator stops the two from carving up VRAM separately, which matters on
        # small cards.
        try:
            from thinc.api import set_gpu_allocator

            set_gpu_allocator("pytorch")
        except Exception as e:
            logger.debug(f"Could not share the PyTorch GPU allocator with cupy: {e}")
        return True

    def _load_spacy(
        self, model_name: str, use_gpu: Optional[bool], exclude: Sequence[str]
    ) -> Any:
        """``spacy.load`` with the GPU decision and the actionable error messages."""
        try:
            import spacy
        except ImportError as e:
            raise ImportError(
                "Graph retrieval needs spaCy. Install it with: uv sync --extra graph"
            ) from e

        on_gpu = self._activate_gpu(use_gpu)
        logger.info(
            f"Loading spaCy model '{model_name}' on {'GPU' if on_gpu else 'CPU'}"
        )

        try:
            return spacy.load(model_name, exclude=list(exclude))
        except OSError as e:
            raise OSError(
                f"spaCy model '{model_name}' is not installed. "
                f"Install it with: python -m spacy download {model_name}"
            ) from e

    @staticmethod
    def _ensure_sentence_boundaries(nlp: Any, model_name: str) -> None:
        if any(nlp.has_pipe(pipe) for pipe in ("parser", "senter", "medspacy_pyrush")):
            return
        # Sentence boundaries drive the whole semantic-bridging step; without them every
        # chunk collapses into a single sentence node.
        logger.warning(
            f"spaCy model '{model_name}' provides no sentence segmentation; "
            "adding a rule-based sentencizer."
        )
        nlp.add_pipe("sentencizer", first=True)

    def _build_pipeline(self, model_name: str, use_gpu: Optional[bool]) -> Any:
        # The lemmatizer is never read here and is the most expensive optional component
        # in the small/medium pipelines.
        nlp = self._load_spacy(model_name, use_gpu, exclude=["lemmatizer"])
        self._ensure_sentence_boundaries(nlp, model_name)
        return nlp

    def extraction_stats(self) -> Dict[str, int]:
        """Counters worth logging once extraction is done. Empty for plain spaCy."""
        return {}

    def _keep(self, ent: Any) -> bool:
        return (
            ent.label_ not in self.excluded_labels
            and len(ent.text.strip()) >= self.min_entity_length
        )

    def extract_documents(self, texts: Sequence[str]) -> Iterator[DocumentEntities]:
        """Yield one ``DocumentEntities`` per input text, in input order."""
        docs = self.nlp.pipe(
            texts,
            batch_size=self.batch_size,
            n_process=self.n_process,
        )
        for doc in docs:
            yield self._from_doc(doc)

    def _from_doc(self, doc: Any) -> DocumentEntities:
        entities: List[str] = []
        seen_in_doc = set()
        # dict preserves insertion order, so sentence nodes stay deterministic
        per_sentence: dict[str, List[str]] = {}

        for ent in doc.ents:
            if not self._keep(ent):
                continue
            surface = ent.text.strip()

            if surface not in seen_in_doc:
                seen_in_doc.add(surface)
                entities.append(surface)

            try:
                sentence = ent.sent.text.strip()
            except ValueError:
                # No sentence boundaries available for this doc; treat it as one sentence.
                sentence = doc.text.strip()
            if not sentence:
                continue

            bucket = per_sentence.setdefault(sentence, [])
            if surface not in bucket:
                bucket.append(surface)

        return DocumentEntities(entities=entities, sentences=list(per_sentence.items()))

    def extract_query_entities(self, query: str) -> List[str]:
        """Entity surface forms mentioned in a query, deduplicated, in order."""
        doc = self.nlp(query)
        found: List[str] = []
        seen = set()
        for ent in doc.ents:
            if not self._keep(ent):
                continue
            surface = ent.text.strip()
            if surface.lower() in seen:
                continue
            seen.add(surface.lower())
            found.append(surface)
        return found


def _silence_pyrush() -> None:
    """Stop PyRuSH from logging once per sentence-boundary decision.

    Two separate channels, and both matter at corpus scale. PyRuSH itself logs through
    loguru at DEBUG. Its rule engine, PyFastNER, logs ``"\\." is not a eligible syntax.``
    through the **root** logger for every unsupported escape it re-reads — measured on this
    benchmark, 2.1 GB of log file in 44 minutes, all of it formatted and written by the
    process doing the extraction.

    Raising the root level stops those at ``isEnabledFor``, which is the cheapest place to
    stop them. mmore's own loggers are named, and a named logger keeps the level set on it,
    so moving mmore's verbosity onto the ``mmore`` logger first leaves this benchmark's
    output untouched. Records that other libraries propagate up to the root *handlers* are
    unaffected either way — propagation does not re-check the root logger's level.
    """
    try:
        from loguru import logger as loguru_logger

        loguru_logger.disable("PyRuSH")
    except ImportError:  # pragma: no cover - loguru ships with PyRuSH
        pass

    root = logging.getLogger()
    effective = root.getEffectiveLevel()
    if effective < logging.WARNING:
        logging.getLogger("mmore").setLevel(effective)
        root.setLevel(logging.WARNING)


class MedspacyEntityExtractor(SpacyEntityExtractor):
    """The same extraction, with medspaCy's clinical components layered on the pipeline.

    medspaCy does not replace the NER model — ``medspacy.load()`` on a blank pipeline
    yields an empty ``TargetMatcher`` and extracts nothing. ``model_name`` still decides
    what recognizes entities; medspaCy contributes ConText, which marks every entity
    negated / uncertain / historical / hypothetical / about a family member, and PyRuSH,
    a sentence splitter built for clinical text.

    ConText is what makes this interesting for a graph: *macular edema* in "there was no
    evidence of macular edema" is an entity the passage argues **against**, and linking
    the passage to it is a false edge. ``medspacy.drop_asserted`` decides whether those
    entities are dropped; it is empty by default because ConText's scope rules were
    written for clinical notes and over-extend on academic prose.
    """

    def __init__(
        self,
        model_name: str,
        medspacy: Optional[MedspacyConfig] = None,
        **kwargs: Any,
    ):
        self.medspacy = medspacy or MedspacyConfig()
        self.drop_asserted = tuple(self.medspacy.drop_asserted)
        self._assertions: Counter = Counter()
        # Sets self.nlp through the overridden _build_pipeline below, so everything this
        # reads must already be assigned.
        super().__init__(model_name, **kwargs)

    def _build_pipeline(self, model_name: str, use_gpu: Optional[bool]) -> Any:
        try:
            import medspacy
        except ImportError as e:
            raise ImportError(
                "ner_backend 'medspacy' needs medspaCy. Install it with: "
                "uv pip install --python .venv/bin/python medspacy"
            ) from e

        components = list(self.medspacy.components)
        exclude = ["lemmatizer"]
        if "medspacy_pyrush" in components:
            _silence_pyrush()
            # PyRuSH writes token.is_sent_start, which spaCy refuses on a parsed doc
            # ([E043]), so the two sentence splitters cannot coexist. Nothing here reads
            # the dependency parse, and dropping it is also the faster pipeline.
            exclude.append("parser")

        nlp = self._load_spacy(model_name, use_gpu, exclude=exclude)
        base_tokenizer = nlp.tokenizer

        logger.info(f"Adding medspaCy components: {', '.join(components)}")
        nlp = medspacy.load(
            nlp,
            medspacy_enable=components,
            load_rules=self.medspacy.load_rules,
            quickumls_path=self.medspacy.quickumls_path,
        )
        if "medspacy_tokenizer" not in components:
            # medspacy.load() only swaps the tokenizer when asked, but be explicit: the
            # base model's NER was trained against its own tokenization.
            nlp.tokenizer = base_tokenizer

        self._load_target_rules(nlp)
        self._ensure_sentence_boundaries(nlp, model_name)
        return nlp

    def _load_target_rules(self, nlp: Any) -> None:
        path = self.medspacy.target_rules_path
        if not path:
            return
        if not nlp.has_pipe("medspacy_target_matcher"):
            raise ValueError(
                "medspacy.target_rules_path is set but 'medspacy_target_matcher' is not "
                "in medspacy.components, so the rules would never be applied."
            )
        from medspacy.target_matcher import TargetRule

        rules = TargetRule.from_json(path)
        nlp.get_pipe("medspacy_target_matcher").add(rules)
        logger.info(f"Loaded {len(rules)} medspaCy target rules from {path}")

    def _keep(self, ent: Any) -> bool:
        if not super()._keep(ent):
            return False
        asserted = [
            attribute
            for attribute in self.drop_asserted
            if getattr(ent._, attribute, False)
        ]
        self._assertions.update(asserted)
        if asserted:
            self._assertions["dropped"] += 1
            return False
        self._assertions["kept"] += 1
        return True

    def extraction_stats(self) -> Dict[str, int]:
        """How many entity mentions each ConText attribute cost, over this extractor's
        lifetime. Only counted for the attributes being filtered on."""
        return dict(self._assertions)


def build_entity_extractor(
    backend: str,
    model_name: str,
    medspacy: Optional[MedspacyConfig] = None,
    **kwargs: Any,
) -> SpacyEntityExtractor:
    """Instantiate the extractor named by ``backend``."""
    if backend == "spacy":
        return SpacyEntityExtractor(model_name=model_name, **kwargs)
    if backend == "medspacy":
        return MedspacyEntityExtractor(
            model_name=model_name, medspacy=medspacy, **kwargs
        )
    raise ValueError(f"Unknown ner_backend: {backend}. Expected one of {NER_BACKENDS}.")


def extractor_from_build_config(config: GraphBuildConfig) -> SpacyEntityExtractor:
    return build_entity_extractor(
        backend=config.ner_backend,
        model_name=config.spacy_model,
        medspacy=config.medspacy,
        excluded_labels=config.excluded_entity_labels,
        batch_size=config.batch_size,
        n_process=config.n_process,
        max_doc_length=config.max_doc_length,
        min_entity_length=config.min_entity_length,
        use_gpu=config.use_gpu,
    )
