# Thesis idea 2 (fallback): profiling, observability and benchmarking for RAG

## Title

Working title:

> **Making a RAG pipeline measurable: end-to-end profiling, cost accounting and a
> reproducible benchmark suite for mmore**

This is the *parallel* idea, kept as a rebound option if the graph-RAG topic
([thesis-ideas.md](thesis-ideas.md)) turns out not to hold. The two are not exclusive: the
graph thesis needs exactly the harness described here, and this thesis would use
graph-vs-hybrid as its flagship case study.


## Context

mmore is a four-stage pipeline — `process` → `postprocess` → `index` → `rag` — where each
stage has a very different cost profile: PDF/OCR extraction is CPU- and GPU-bound and runs
distributed over Dask, indexing is embedding-model-bound and I/O-bound on Milvus, retrieval
is latency-bound, generation is token- and money-bound. A user who wants to know *why a run
took four hours*, *which stage will blow up at 100× the corpus*, or *whether a config change
made things worse* has, today, no way to find out other than reading the console.

Fabrice's framing when the idea came up: this is interesting not only as tooling but because
it **produces data that can be used for research** — memory profiling, generated metrics,
benchmarks over available datasets.

The first job, therefore, is to establish what mmore already has. That inventory is below,
and it changes the shape of the topic: mmore is not un-instrumented, it is instrumented
*shallowly and non-persistently*.


## What already exists in mmore (audit of `EPFLiGHT/mmore`, v2.0.0)

### 1. A profiler module — real, documented, tested, but coarse

`src/mmore/profiler.py` (338 lines) wraps Python's `cProfile`:

| Entry point | What it does |
|---|---|
| `@profile_function()` | profiles one function, dumps a `.prof`, prints a `pstats` table |
| `profile_context(name)` | same, for a block |
| `Profiler` class | manual `start()` / `stop()`, also a context manager |
| `time_function()` / `time_context()` | wall clock only, logged as `⏱️ X took N s` |
| `enable_profiling_from_env()` | reads `MMORE_PROFILING_ENABLED`, `MMORE_PROFILING_OUTPUT_DIR`, `MMORE_PROFILING_SORT_BY`, `MMORE_PROFILING_MAX_RESULTS` |

It is wired in: `cli.py:16` calls `enable_profiling_from_env()` for every command, and
`@profile_function()` decorates the **top-level entry point of each stage** —
`run_process.py:87`, `run_postprocess.py:27`, `run_index.py:34`, `run_rag.py:215`,
`run_retriever.py:54,291`, `run_live_retrieval.py:12`, `run_ragcli.py:457`, plus the four
`colvision/run_*.py`. It is documented (`docs/source/advanced_usage/profiler.md`) and covered
by 363 lines of tests (`tests/test_profiler.py`).

So the granularity is exactly one `cProfile` dump per *whole stage*, per *driver process*.

### 2. Ad-hoc wall clocks scattered in the pipeline

Finer timings already exist, but each was added for one display and is thrown away after
printing:

- `rag/retriever.py:376,417` accumulates `_retrieve_seconds` / `_rerank_seconds`, drained
  once by `pop_timings()`.
- `ragcli_helper.py:30` — `TimingHandler`, a LangChain `BaseCallbackHandler` collecting
  retrieval time, generation time and completion tokens (from `usage_metadata`, output
  tokens only), used to print `… @ 42 tok/s`.
- `run_rag.py:43` — `BatchGenerationTimer`, same callback trick for batch mode.
- `ux.py:312` — `model_loading_seconds()`, subtracted so warm-up does not count as pipeline
  time.

### 3. A presentation layer with per-stage summary stats

`src/mmore/ux.py` (647 lines) centralises logging setup, third-party log silencing, progress
bars and `step_summary()` — the closing Rich panel each stage prints. The stats it carries
are already the right *kind* of numbers, but they are strings in a terminal panel:

- `process`: files dispatched vs reused, samples out, size, `files/s` throughput.
- `rag`: `retrieve s/query`, `rerank s/query`, `generate s/query`, privacy-pipeline s/query.

### 4. Quality evaluation — RAGAS, config-driven, not on the CLI

`src/mmore/rag/evaluator.py` (199 lines) builds an index from a HF dataset, runs the RAG
pipeline over its queries and scores with RAGAS: `LLMContextRecall`,
`LLMContextPrecisionWithReference`, `ContextEntityRecall`, `NoiseSensitivity`,
`ResponseRelevancy`, `Faithfulness`, `FactualCorrectness`, `SemanticSimilarity`. Documented
in `core_features/evaluation.md`, example configs in `examples/rag/evaluation/`. It returns a
pandas frame and stops there — no `mmore eval` command exists (the CLI exposes process,
postprocess, index, retrieve, live_retrieval, rag, index_api, websearch, ragcli, tui,
colvision).

### 5. Retrieval-quality signals, used for control rather than reporting

`rag/judge/metrics.py:14` computes `num_docs`, `mean/max_similarity`, `mean/max_rerank_score`
and checks them against thresholds — but to drive the corrective-RAG decision
(`RE_RETRIEVE` / `ADD_QUESTIONS` / `ADD_CONTEXT`), not to report on a run.

### 6. A benchmark script that does not run as shipped

`scripts/lm_eval_rag.sh` (126 lines) integrates the RAG-evaluation-harnesses fork of `lm_eval`
over `pubmedqa,medmcqa,medqa_4options,mmlu_college_medicine,afrimedqa` against
`OpenMeditron/Meditron3-8B`. It needs an external checkout at
`$LIBS_PATH/RAG-evaluation-harnesses`, **and the four Python helpers it calls
(`examples/rag/evaluation/lm_eval_harness/*.py`) are absent from the repository** — that
directory does not exist. The historical outputs of those runs are still there as loose
artifacts (`examples/who/scott_ds_*.jsonl`, `examples/pubmedqa/*.yaml`), with no harness able
to regenerate them.


## Gaps — the actual room for a thesis

| Gap | Evidence |
|---|---|
| **Memory is not profiled at all** | `ProfilingConfig.profile_memory` exists (`profiler.py:27,40,244`) and is **read nowhere**. No RSS, no peak, no GPU memory. `psutil` is not even a core dependency (only in the `privacy` extra). |
| **Nothing is machine-readable** | The profiler emits `.prof` + a printed `pstats` table; `step_summary` prints a Rich panel. No JSON, no CSV, no run record. Two runs cannot be compared without rerunning them side by side and reading two terminals. |
| **No run identity or provenance** | No capture of CPU/GPU/RAM, package versions, commit, config hash. Numbers from two machines — or two branches — are not comparable, which is precisely what a research dataset requires. |
| **Wrong granularity in both directions** | `cProfile` over a whole stage is too coarse to answer "where does *this query* spend its time" and too heavy to leave on; it also silently disables itself when another profiler is active (`profiler.py:92`). There is no per-document or per-query span. |
| **The distributed stage is a blind spot** | `process` dispatches over Dask (`process/dispatcher.py`); the decorator only wraps the driver, so worker time — the bulk of the cost — is invisible. |
| **No cost accounting** | Token counting is display-only, output tokens only (`ragcli_helper.py:_output_tokens`). No input tokens, no per-provider pricing, no €/query or €/corpus. |
| **No telemetry surface** | Nothing OpenTelemetry, Prometheus, W&B or MLflow anywhere in `src/`. The hook point already exists (LangChain callbacks are used by `TimingHandler`) but is used for one print. |
| **No benchmark harness, no baselines, no perf CI** | Workflows are `publish`, `push-to-registry`, `pyright`, `ruff`, `sphinx-docs`, `tests` — no performance job. No committed baseline results, no statistical testing, no dataset wiring that runs end to end. |
| **Quality and cost are measured by separate, disconnected tools** | RAGAS scores quality with no notion of latency or cost; the profiler measures cost with no notion of quality. The interesting question — *quality per second, quality per euro* — cannot be asked. |


## Prior art inside this fork

`benchmarks/graph_vs_hybrid/` (≈3 700 lines, written for the graph-RAG idea) is effectively a
prototype of what this thesis would generalise, which de-risks the topic considerably:

- `timing.py` — each stage in its own subprocess, measured for wall time, child CPU time and
  **peak RSS of the whole process tree** (sampled at 100 ms), because peak memory is only
  meaningful per stage and models must not stay warm across stages.
- `retrieval_bench.py` — `hit@1`, `recall@k`, `MRR`, per-hop breakdowns, latency percentiles.
- `answer_bench.py`, `qa.py` — answer scoring; `mcnemar.py` — paired significance testing.
- `report.py` — one markdown comparison per run, with environment capture (platform, python,
  physical cores, RAM, GPU).
- Four task adapters with corpus construction and distractor padding: PubMedQA, HotpotQA,
  MedHop, ICD-11.

It is task-specific (graph vs hybrid, retrieval only) and lives outside `src/`. The thesis
would be to turn that shape into a first-class, pipeline-wide capability.


## Problem statement

mmore can be run but not *characterised*. Its cost is observable only as text on a terminal,
at whole-stage granularity, on the driver process, with no memory dimension, no persistence
and no provenance — so no run can be compared to another, no regression can be caught, and no
configuration choice can be justified with evidence. Meanwhile quality is measured by a
separate tool that knows nothing of cost.

The thesis is to give mmore a single measurement substrate covering time, memory, tokens and
money alongside retrieval and answer quality, and then to *use* it: to produce a
characterisation of where a multimodal RAG pipeline actually spends its resources, and to turn
that characterisation into decisions.

**Stated plainly:** the plumbing alone is engineering, not research. What makes this a thesis
is the measurement *study* (RQ2) and the *decision layer* (RQ3) built on top of it. That
framing has to be defended from the start, and it is the main risk of this topic compared to
the graph one.


## Research questions

### RQ1. What is the right measurement model for a multi-stage, multimodal, distributed RAG pipeline?

A single span/event model spanning `process → postprocess → index → retrieve → rerank →
generate`, at document and query granularity, that survives Dask distribution and that can be
left enabled in production.

Sub-questions:

- What is the minimum set of dimensions? (wall, CPU, peak/steady RSS, GPU memory and
  utilisation, I/O, input+output tokens, per-provider cost, plus the identity of what was
  measured: modality, processor, model, `k`, chunker.)
- What is the observer effect? `cProfile` at query granularity is unusable in production —
  what is the overhead budget of a sampling or span-based alternative, and how is it measured?
- How are worker-side events collected and joined back to a run without turning the pipeline
  into a distributed-tracing project?
- Does the record adopt an existing schema (OpenTelemetry spans) or a purpose-built one, and
  what does that choice cost in dependencies for an offline deployment?

### RQ2. Where does a RAG pipeline actually spend its resources, and how does that scale?

This is the measurement study, and the part that produces reusable research data.

Sub-questions:

- Per stage and per modality: what dominates, and does the ranking change with corpus size,
  document type, embedding model, hardware?
- Which costs are linear in corpus size and which are not? What can be predicted — can the
  cost of indexing a corpus be estimated from a sample before committing to the full run?
- Memory specifically: what is the peak-RSS profile of each stage, what drives it (batch size,
  model, document length), and which stage sets the hardware requirement for a deployment?
- Where does the money go per answered question, and how does that decompose across retrieval
  breadth `k`, reranking, and generation?

### RQ3. Can the measurements be turned into decisions rather than dashboards?

Sub-questions:

- **Regression detection.** What does a performance CI job need in order to be trustworthy on
  noisy runners — which metrics are stable enough, at what sample size, with which statistical
  test? (`mcnemar.py` is the existing precedent for paired testing.)
- **Config recommendation.** Given a latency or cost budget, can the harness pick `k`,
  reranker on/off, batch size, chunking strategy and embedding model from measured
  quality-vs-cost curves instead of from folklore?
- **Joint reporting.** What is the right way to present quality *and* cost together — quality
  per second, quality per euro — so that two pipeline configurations can be ranked honestly?

### RQ4. What does a reproducible RAG benchmark suite require to be reusable by others?

Sub-questions:

- Which datasets are wired in, and how is the "available dataset" problem handled (PubMedQA
  and HotpotQA download cleanly; ICD-11 definitions are only reachable through the WHO API —
  see `benchmarks/graph_vs_hybrid/README.md`)?
- What must be recorded for a published number to be reproducible on another machine?
- Can the released measurement records themselves be a contribution — a public dataset of
  RAG pipeline runs across configs, hardware and corpora?


## Plan

1. **Audit and consolidate** — fold the scattered wall clocks (`retriever.pop_timings`,
   `TimingHandler`, `BatchGenerationTimer`, `step_summary` stats) onto one record; make
   `profile_memory` mean something.
2. **Instrument** — the span/event model of RQ1, persisted as JSONL per run with full
   environment and config provenance; keep the Rich summary as one renderer of that record,
   not as the record.
3. **Cover the distributed stage** — worker-side collection for the Dask `process` pipeline.
4. **Cost and tokens** — input+output tokens through the LangChain callbacks, per-provider
   pricing, €/query and €/corpus.
5. **Generalise the benchmark harness** — lift `benchmarks/graph_vs_hybrid/` out of its one
   comparison into a task/strategy-parameterised suite covering all four stages, and repair or
   replace the `lm_eval` path (its helper scripts are missing upstream).
6. **Join quality to cost** — the RAGAS evaluator and the retrieval metrics writing into the
   same run record.
7. **Run the study (RQ2)** — scaling curves across corpus size, modality, model, hardware.
8. **Decision layer (RQ3)** — perf CI job with a defensible significance test, and
   budget-constrained config recommendation.
9. **Release** — baselines, the measurement dataset, documentation.

Steps 1–4 are the substrate, 5–6 the harness, 7–8 the thesis contribution. If time runs
short, 8 is the part to cut, not 7.


## Use cases

### Deployment sizing in constrained settings

The humanitarian and clinical deployments mmore targets are exactly where "how much hardware
does this corpus need, and what will it cost per question" is a blocking question, and where
the answer cannot be "run it and see". A memory-and-cost characterisation per stage answers it
before the deployment, and the peak-RSS-per-stage number is what sets the machine.

### Making pipeline choices defensible

Every mmore config choice — hybrid weight, `k`, reranker, chunker, embedding model — is
currently made without a measured quality/cost trade-off. The joint report makes those
choices arguable, and it is the same instrument the graph-RAG thesis needs to state its own
result under equal conditions.

### A shared measurement substrate for the lab

Several theses touch the same pipeline (graph retrieval, privacy, ColVision, websearch). Each
currently re-measures ad hoc. One run record with provenance means their numbers become
comparable, and the accumulated records become a dataset in their own right.


## Relation to idea 1

- **Complementary, not competing.** The graph thesis measures one comparison well; this one
  makes the measurement itself the object.
- **Shared code.** `benchmarks/graph_vs_hybrid/` is the seed of the harness either way, so
  work done now is not lost if the topic switches.
- **The honest caveat.** Idea 1 has a clear scientific question with a published baseline to
  reproduce; idea 2 has to manufacture its research question out of a study, and risks
  reading as infrastructure work. If it is chosen, RQ2 and RQ3 should lead the framing and
  RQ1 should be presented as means, not as the contribution.
