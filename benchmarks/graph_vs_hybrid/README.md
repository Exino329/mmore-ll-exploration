# Standard RAG vs LinearRAG graph retrieval

Two questions, measured separately: **what does indexing cost**, and **which retrieves and
answers better**.

The two strategies differ in exactly one config key. `retriever.type: hybrid` is mmore's
dense+sparse Milvus search; `retriever.type: graph` is entity activation plus personalized
PageRank over the Tri-Graph built by `mmore graph-index`. Same collection, same embedding
model, same k, same generator — so the numbers are about retrieval, not about setup. The
mmore stage configs are generated from the bench config into the run directory rather than
maintained by hand, precisely so they cannot drift apart.

## Tasks

`task:` selects the corpus, the questions and how answers are scored. Everything
downstream — timings, retrieval metrics, the report — is task-agnostic.

### `pubmedqa` (default) — `bench_pubmedqa.yaml`

The 1,000 expert-annotated PubMedQA instances supply both golds: the abstract a question
was written from (retrieval) and its yes/no/maybe decision (generation). Neither needs an
LLM to score.

On their own, 1,000 abstracts make retrieval trivial. The corpus is padded with abstracts
from `pqa_artificial` that no question points at — `pubmedqa.distractors`, 19,000 by
default, for a 20,000-document corpus. Distractors are streamed, so only that many rows
are downloaded.

Two leaks are deliberately closed. The abstract's conclusion (`long_answer`) is not
indexed: it states the answer, and indexing it would measure copying. And passages carry
**no title** — PubMedQA's questions are derived from the article titles, which the dataset
does not ship, so putting the question on its own gold passage would turn retrieval into
exact string matching.

Read the accuracy against the majority-class baseline the report prints: 55% of the
labelled decisions are "yes".

### `icd11` — `bench_icd11.yaml`

Reproduces the earlier GraphRAG-vs-standard-RAG study: join the label→description and
label→title tables, map each disease to its category, keep the entries carrying a
description, one document per disease.

**The dataset is not shipped and is not downloadable as such.** 34,663 entities of which
6,641 have definitions is the signature of the WHO ICD-11 MMS linearization, and the public
tabular exports carry only code and title — the definitions live in the WHO API. The tables
behind the earlier study were therefore almost certainly built from that API rather than
downloaded, which is why `icd11.source` is source-agnostic: `kind: parquet` takes four file
paths, `kind: hf` takes `"dataset_id[:config][:split]"` in the same slots, and column names
are configurable. The most reliable route to comparable numbers is to ask the earlier
study's author for the parquet files.

`category_mapping: study` reproduces their rule (numeric first character → that category,
alphabetic → `13 + alphabet_index`). That formula does not match the real WHO chapters,
which run `A`→10 … `S`→26 with `I`, `O`, `U` unused; `category_mapping: icd11` uses those.
The category title is part of the indexed text, so the choice changes what is retrieved.

Questions come from `icd11.style`: `vignette` has an LLM rewrite each description as a
clinical presentation with the disease unnamed (one call per question), `masked` blanks the
title out of its own description (free, but the question then shares most of its wording
with the passage, and dense retrieval scores near ceiling).

### `hotpotqa` — `bench_hotpotqa.yaml`

The one task here whose numbers are comparable to a published table. LinearRAG's own
`scripts/run.sh` carries a HotpotQA preset — `MAX_ITERATION=3`, `THRESHOLD=0.4`,
`PASSAGE_RATIO=0.05`, `TOP_K_SENTENCE=1`, `en_core_web_trf`, `all-mpnet-base-v2` — and it
is what `mmore.rag.graph.config` already defaults to, so a run here says whether this port
reproduces their result rather than only whether the graph helps on a corpus we built.

**Their questions, not their corpus.** The release ships
`Zly0523/linear-rag/hotpotqa/{questions,chunks}.json`. The questions are used as-is: 1,000
HotpotQA dev instances, 811 bridge and 189 comparison, whose ids join 1000/1000 into the
`distractor` validation split. The chunks are not usable, for four independent reasons:

- 1,311 blocks of ~4,650 characters cut blindly through the whole concatenated corpus.
  Boundaries fall mid-word — chunk 1 opens on `##dh`, a leftover WordPiece continuation —
  so a block holds the tail of one article and the head of another.
- The text is lowercased and detokenized (`"$ 50 million"`, `"don ' t"`), which is most of
  what a general-domain NER model keys on.
- Each block carries an `"N:"` index prefix that goes into the embedding with the rest.
- **Nothing labels a gold passage.** Their `evidence` field is the full 10-paragraph
  distractor context, supporting facts included but unmarked, and `src/evaluate.py`
  computes `contain` accuracy and an LLM verdict and nothing else. LinearRAG publishes no
  retrieval metric at all, so there is no recall@k to be comparable to — only answers.

So the corpus is rebuilt at the granularity a question points at: one document per
Wikipedia article, from the dev `context` field. Pooling the ten contexts of the 1,000
asked questions gives **9,811 articles**, the size the HotpotQA retrieval literature
reports on. `hotpotqa.distractor_questions` pools in the contexts of further dev questions
that are never asked, for a harder corpus without changing what is scored.

**Per-hop gold, derived.** HotpotQA labels the two supporting articles but not which is the
bridge. Two rules, both mirroring `medhop`: the far article (`hop2`) holds the answer string
(639/811 bridge questions have it in exactly one support); failing that, the near article
(`hop1`) is the one whose title the question names outright, on word boundaries (429/811).
Together they split **726/811** bridge questions, and where both fire they agree on 85%.
The other 85 keep an aggregate gold and no hop split — group metrics are scored over the
questions that define them, so `hop2_at_k` is not diluted by the ones that cannot.

**The control group is free.** `question_type` partitions the same corpus, the same
pipeline and the same run: 811 `bridge` questions where the answer article is reachable
only by bridging, and 189 `comparison` questions that name both entities and need no
bridging at all. If graph retrieval's advantage is real it shows up on `bridge`/`hop2` and
vanishes on `comparison`. An advantage visible on both is not bridging — it is something
else, and no other task here can tell the difference within a single run.

Two metrics are new with this task and apply to any multi-gold one:

- `all_at_k` — *every* gold in the top k, not just the first. There are exactly two
  supporting articles and an answer needs both, which is what HippoRAG reports as
  Recall@2/@5. `recall_at_k` is satisfied by the easy one.
- `bridge_at_k` / `comparison_at_k` — the control split above.

`answer.scoring: contain` is LinearRAG's own metric: the gold span appears in the answer
the model committed to. Deterministic and free. It reads only what follows `"Answer:"`,
as their `run.py` does — scoring the whole chain-of-thought would credit a gold string that
merely appears in the reasoning.

## Install

```bash
uv sync --extra index --extra llm --extra graph --extra cpu   # or --extra gpu
export OPENAI_API_KEY=...            # only for the `answer` stage and vignette questions
```

### Biomedical entity extraction is not optional

`en_core_web_sm` is general-domain. Measured on 1,300 PubMedQA abstracts, it extracts no
biomedical entity from the questions — "winter" tagged `DATE` is a typical hit — so the
graph retriever finds no seed and **falls back to hybrid on 75% of queries**. The benchmark
would then be comparing hybrid against itself. scispaCy's `en_core_sci_md` brings that to
**0%** and triples the entity vocabulary (10,072 → 26,892 entities on the same corpus).
LinearRAG uses scispaCy for its medical corpus too.

```bash
uv pip install --python .venv/bin/python scispacy==0.6.2
uv pip install --python .venv/bin/python \
  https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_md-0.5.4.tar.gz
uv pip install --python .venv/bin/python spacy==3.7.5
```

The last line is the catch: the scispaCy models are built for spaCy 3.7 and fail to load
under 3.8 with a config validation error, while `pyproject.toml` asks for `spacy>=3.8`. The
venv therefore has to sit at 3.7.5, and **`uv sync` will undo it** — reinstall the pin
afterwards. `en_core_web_sm` still loads at 3.7.5 with a `[W095]` warning, and the repo's
graph tests pass there.

Measured on 500 questions: `en_core_sci_md` and `en_core_sci_scibert` are
indistinguishable (hit@1 0.798 vs 0.808, p=0.62). What matters is *being* a biomedical
model, not which one. `en_core_sci_md` is CPU-friendly and is the sane default.

### General-domain extraction, for HotpotQA

`hotpotqa` is Wikipedia, so the biomedical models are the wrong tool and scispaCy is not
needed. LinearRAG's preset for it is `en_core_web_trf` — general-domain, but the
transformer pipeline, not `sm`. That distinction is the one the PubMedQA run already paid
for: `en_core_web_sm` is what left 75% of queries with no seed entity. On OntoNotes text it
does far better than it did on abstracts, but `trf` is what makes the run comparable.

```bash
uv pip install --python .venv/bin/python \
  https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl
```

The 3.7.3 wheel is the one to take: it resolves against the `spacy==3.7.5` pin the
scispaCy models force, adding only `spacy-curated-transformers` and its two dependencies,
and moves neither spacy nor the biomedical models. Verified — the other tasks keep working
after it. `en_core_web_lg-3.7.1` is the CPU-friendly substitute if no GPU is reachable; it
is not the paper's setting, so say so when reporting.

The corpus is small (9,811 short articles), so this is a cheap run compared to the 20,000
PubMed abstracts — but `trf` still wants the GPU, which spaCy reaches through cupy and the
`LD_LIBRARY_PATH` export documented below.

### Trying another entity extractor

`graph.ner_backend` selects *how* entities are extracted, `graph.spacy_model` selects *what
model* does it. The two are independent, so going back to yesterday's configuration is one
key — every existing bench config keeps working untouched, and graphs built before the
setting existed load as `ner_backend: spacy`.

| `ner_backend` | pipeline | install |
|---|---|---|
| `spacy` (default) | `spacy_model` alone | `--extra graph` |
| `medspacy` | `spacy_model` wrapped in [medspaCy](https://github.com/medspacy/medspacy)'s clinical components | `uv pip install --python .venv/bin/python medspacy` |

**medspaCy is not an NER model.** `medspacy.load()` on a blank pipeline gives an empty
`TargetMatcher` and extracts nothing, so `spacy_model` still decides what finds entities.
What medspaCy adds is *assertion*: ConText marks every entity negated, uncertain,
historical, hypothetical or about a family member. That is the reason to want it here — a
passage saying "there was no evidence of macular edema" currently links to *macular edema*
as if it were about it, and `graph.medspacy.drop_asserted: [is_negated]` is what stops it.
It also swaps in PyRuSH for sentence splitting, which forces the parser out of the pipeline
(spaCy refuses two sentence splitters: `[E043]`).

It installs cleanly next to scispaCy — it asks for `spacy<3.8` on Python 3.11, the same
place the scispaCy models pin the venv to. That constraint is why it is not in
`pyproject.toml`: the `graph` extra requires `spacy>=3.8` and the two cannot be resolved
together.

`bench_pubmedqa_medspacy.yaml` reuses the PubMedQA run directory, so only the graph is
rebuilt — no `prepare`, no `index`:

```bash
BENCH=benchmarks/graph_vs_hybrid/bench_pubmedqa_medspacy.yaml
python -m benchmarks.graph_vs_hybrid index --graph-only -c $BENCH --tag medspacy
python -m benchmarks.graph_vs_hybrid retrieval --strategies graph -c $BENCH --tag medspacy

# the run with a hypothesis behind it: drop the entities ConText says are negated
python -m benchmarks.graph_vs_hybrid index --graph-only -c $BENCH --tag medspacy_neg \
  --set graph.medspacy.drop_asserted=is_negated \
  --set graph.artifacts_name=graph_index_medspacy_neg
python -m benchmarks.graph_vs_hybrid retrieval --strategies graph -c $BENCH --tag medspacy_neg \
  --set graph.medspacy.drop_asserted=is_negated \
  --set graph.artifacts_name=graph_index_medspacy_neg
```

With `drop_asserted` empty, medspaCy changes the entity set **not at all** — 240,147
entities against plain scispaCy's 240,137 on the full corpus, since those are still
`en_core_sci_md`'s — and moves only the sentence split. It is the control, not the
experiment. `drop_asserted: [is_negated]` is the experiment: it cost 7,793 entities (−3.2%)
and 40,856 edges (−3.2%) on the 20k corpus. Extraction runs at roughly half the speed of
plain scispaCy — 23.5 min against ~12 — so budget ~28 min per graph, embeddings included.

`graph.medspacy.components` takes any of `medspacy.util.ALL_PIPE_NAMES`. Two are worth
knowing about: `medspacy_target_matcher` with `target_rules_path` makes medspaCy extract
entities *itself* from a lexicon, and `medspacy_quickumls` maps them to UMLS concepts —
though the dictionary medspaCy ships is a demo that matches almost nothing, and a real one
has to be built from a UMLS subscription.

### GPU, for the transformer pipeline

`en_core_sci_scibert` runs at 2.9 abstracts/s on six CPU cores — nearly two hours for
20,000 — and 51.9/s on an RTX 3060 Ti. spaCy does not take the GPU on its own: it reaches
it through cupy, and `graph.use_gpu` (added to `GraphBuildConfig`) calls `prefer_gpu()`
before the pipeline loads. The default `None` uses the GPU when one is reachable and falls
back to CPU silently; the stage logs which it picked.

```bash
uv pip install --python .venv/bin/python "cupy-cuda12x<14" "numpy==1.26.4"
```

Two traps, both silent:

- **cupy 14 is built against numpy 2**, spaCy 3.7.5 against numpy 1. Installing plain
  `cupy-cuda12x` pulls numpy 2 and breaks every compiled spaCy extension. Hence `<14`.
- **cupy cannot find `libcublas.so.12`.** The CUDA libraries are present — PyTorch ships
  them under `site-packages/nvidia/*/lib` — but cupy does not look there, so
  `spacy.prefer_gpu()` returns False and everything quietly runs on CPU. Export the path
  before running:

```bash
export LD_LIBRARY_PATH="$(ls -d $PWD/.venv/lib/python3.11/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH"
```

Check it took: the `graph-index` log prints `Loading spaCy model '…' on GPU`.

The dense and sparse models never had this problem — sentence-transformers and splade
detect CUDA by themselves.

## Run

```bash
python -m benchmarks.graph_vs_hybrid prepare    # corpus.jsonl + qa.jsonl
python -m benchmarks.graph_vs_hybrid index      # times `mmore index`, then `mmore graph-index`
python -m benchmarks.graph_vs_hybrid retrieval  # recall@k / MRR / latency — no LLM, free to repeat
python -m benchmarks.graph_vs_hybrid answer     # answer accuracy — needs an API key
python -m benchmarks.graph_vs_hybrid report     # one markdown table set
```

`all` chains them (`--skip-answer` to stop before the paid part). Every stage takes
`-c <bench.yaml>` and `--run-dir` on either side of the stage name.

Everything a run produces — corpus, questions, Milvus database, graph artifacts, generated
configs, logs, per-query results — lands under `run_dir`, so runs are self-contained and
two of them can be diffed. `index` wipes that run's database and graph before measuring,
and touches nothing outside it.

### Sweeping graph parameters

Every graph knob except the entity-extraction ones (`spacy_model`, `ner_backend`,
`medspacy`, `normalize_entities`) is applied at *query* time,
so a parameter study needs no re-indexing — it is one `retrieval` pass per setting, a few
minutes each. `--set` overrides any config field, `--tag` keeps the results from
overwriting each other:

```bash
for pr in 0.05 0.5 1.5 3.0; do
  python -m benchmarks.graph_vs_hybrid retrieval --strategies graph \
    --tag "pr$pr" --set graph.passage_ratio=$pr --set graph.iteration_threshold=0.5
done
```

Results land in `results/retrieval_graph_<tag>.json`, each carrying the `graph_settings`
that produced it. `passage_ratio` is the one to look at first: it is the weight of the
dense-retrieval prior in the PageRank reset vector, and the paper's presets differ by a
factor of 30 between datasets — 0.05 for HotpotQA and 2WikiMultihop, **1.5 for their
medical corpus**. On a task where dense retrieval alone is strong, 0.05 throws away the
signal that works.

Rough cost of the default PubMedQA run on 20,000 documents, extrapolated from 1,300
(6 cores, RTX 3060 Ti): ~14 min for `index`, ~20 min for `graph-index`, and about 1 GB of
graph artifacts. Retrieval evaluation is seconds.

Check the harness after changing it, in about a minute, with no API key and no scispaCy:

```bash
python -m benchmarks.graph_vs_hybrid all -c benchmarks/graph_vs_hybrid/bench_smoke.yaml --skip-answer
```

It runs on a fabricated corpus; its accuracy numbers mean nothing.

## Reading the results

**Indexing.** Graph RAG is *additive*, not alternative: `mmore graph-index` reads its
passages back out of the Milvus collection, so its column is `index` + `graph-index`. The
difference column is what the graph actually costs on top. Wall clock includes model
loading and Milvus startup — a fixed ten-odd seconds that dominates small corpora — so the
report also shows the compute time each stage reports for itself, which excludes it. Peak
RSS is sampled every 100 ms over the process tree, so a shorter spike can be missed.

**Retrieval.** `recall@k` is document-level: the gold key appears among the first k
distinct retrieved documents. The reranker is off by default — the same cross-encoder
reordering both strategies' top-k hides the difference being measured. Set
`retrieval.reranker_model_name` to compare deployed configurations instead.

Always read `hybrid_fallback_rate` first. A graph query that links no query entity into the
graph falls back to hybrid retrieval, and those results are hybrid results sitting in the
graph column. A non-zero rate means part of the comparison is hybrid against itself, and
points at the NER model or at `seed_min_similarity`.

**Answers.** `scoring: exact_label` matches the prediction against a closed label set and
reports the majority-class baseline next to it. `scoring: judge` is for open-ended answers
and uses LinearRAG's `src/evaluate.py` prompt and its `contain` accuracy, so those figures
stay comparable to theirs. Generation runs sequentially — real per-question latency, and
the retriever holds a spaCy pipeline and a Milvus client that should not be driven from two
threads; only judging is parallel.

## What the PubMedQA run found

500 questions, 20,000 documents, 0.2% graph fallback — so the graph really was exercised.

| | hit@1 | MRR | recall@10 | median latency |
|---|---|---|---|---|
| hybrid | **0.936** | **0.957** | 0.988 | **32 ms** |
| graph, paper defaults | 0.714 | 0.804 | 0.952 | 456 ms |
| graph, best of ten configurations | 0.808 | 0.885 | 0.988 | 447 ms |

Indexing cost the graph 1.45× the wall clock, +3.7 GB peak RSS and +720 MB on disk.

Three things came out of the sweep, each checked with a paired McNemar test on hit@1:

- **The graph finds the right documents and ranks them worse.** recall@10 reaches parity
  (0.988) while hit@1 stays 13 points behind. The failure is in the ranking, not the
  candidate set.
- **Every setting that helps, helps by suppressing the graph.** Raising `passage_ratio`
  (0.05 → 10) and lowering `damping` (0.85 → 0.15) are two routes to the same ceiling and
  are statistically indistinguishable from each other. The best configuration is the one
  that leans hardest on the dense prior and diffuses least. `damping=0.85`, the value the
  paper's text quotes, is the worst tested.
- **The NER model matters only up to a point.** General-domain `en_core_web_sm` left 75% of
  queries with no seed entity, silently falling back to hybrid; any biomedical model fixes
  that. But `en_core_sci_scibert` over `en_core_sci_md` is noise (p=0.62) — the two build
  graphs of near-identical size.

Read as: on single-hop biomedical retrieval, LinearRAG costs a lot and returns nothing.
Which is the expected place for it to lose — see below.

## What the medspaCy runs found

Same 500 questions, same collection, same query-time parameters — only the entity
extraction differs. Both medspaCy graphs were built against the graph the table above
measures, so the three columns are paired question by question.

| | hit@1 | MRR | recall@10 | median latency |
|---|---|---|---|---|
| scispaCy `en_core_sci_md` | 0.714 | 0.804 | 0.952 | 456 ms |
| + medspaCy, drops nothing (control) | **0.718** | 0.805 | 0.952 | 533 ms |
| + medspaCy, `drop_asserted: [is_negated]` | 0.702 | 0.795 | 0.946 | 523 ms |

Paired McNemar on hit@1, discordant pairs and exact two-sided p:

| | discordants | p |
|---|---|---|
| scispaCy vs control | 3 / 5 | 0.727 |
| scispaCy vs drop-negated | 12 / 6 | 0.238 |
| **control vs drop-negated** | **11 / 3** | **0.057** |

- **PyRuSH segmentation is a non-event** (p=0.73). The clinical sentence splitter on
  academic prose changes neither the entity set (240,147 vs 240,137) nor the ranking.
- **Dropping ConText-negated entities hurts**, and the control is what shows it. Against
  the scispaCy baseline the drop looks like noise (p=0.24) because two changes are mixed
  in; against the control, which differs by that filter alone, it is 11 questions lost
  against 3 gained (p=0.057). Borderline, one run, 14 discordant questions out of 500 —
  but the direction is consistent across hit@1, MRR and recall@10.
- **Why it hurts is the useful part**: in retrieval, a negated entity is still what the
  passage is *about*. An abstract concluding "no association between X and Y" is squarely
  about X and Y, and cutting those edges removes real topical signal. Negation is decisive
  for *assertion* — deciding whether a fact holds, which is what building a knowledge base
  needs — and counterproductive for *topical retrieval*. Do not carry a ConText filter
  from one task to the other.

The run also produced a lesson about the harness rather than the method: PyRuSH's rule
engine logs through the root logger, and those lines cost **36 of the first run's 59
minutes** of extraction (2.6 GB of log). `_silence_pyrush` in `mmore.rag.graph.ner` raises
the root level for exactly that reason. Any timing measured before that fix — including
the "5× slower than scispaCy" first reading — is an artifact of it; the real cost of
medspaCy is about 2×.

## What the strategy matrix found

`hybrid` mixes dense and SPLADE, which hides which half does the work, and comparing the
graph against that mixture alone cannot say whether the graph adds anything to a retriever
that already works. Eight strategies, same collection, same 500 questions, same k:

| | hit@1 | MRR | recall@10 | median latency |
|---|---|---|---|---|
| BM25 | 0.890 | 0.927 | 0.980 | **1 ms** |
| dense | 0.926 | 0.946 | 0.980 | 30 ms |
| SPLADE | **0.946** | **0.964** | **0.988** | 32 ms |
| hybrid (dense+SPLADE) | 0.936 | 0.957 | **0.988** | 32 ms |
| graph | 0.714 | 0.804 | 0.950 | 420 ms |
| graph + dense | 0.844 | 0.903 | 0.980 | 489 ms |
| graph + BM25 | 0.838 | 0.897 | 0.972 | 423 ms |
| graph + hybrid | 0.836 | 0.898 | 0.978 | 497 ms |

Paired McNemar on hit@1:

| comparison | Δ hit@1 | discordants | p |
|---|---|---|---|
| SPLADE vs hybrid | +0.010 | 17 / 12 | 0.46 |
| graph+dense vs dense | −0.082 | 13 / 54 | 4×10⁻⁷ |
| graph+BM25 vs BM25 | −0.052 | 18 / 44 | 1×10⁻³ |
| graph+hybrid vs hybrid | −0.100 | 10 / 60 | 8×10⁻¹⁰ |
| graph+dense vs graph | +0.130 | 66 / 1 | 9×10⁻¹⁹ |

- **Fusing the graph into a working retriever makes it worse, every time.** All three
  fusions land significantly below the partner they were fused with. Rank fusion cannot be
  blamed: the same RRF lifts the bare graph by 13 points (66 questions gained, 1 lost).
- **The premise behind the fusion does not hold here.** "The graph finds the right passages
  and orders them badly" predicts that its candidate set brings something dense lacks. It
  brings nothing: `graph + dense` reaches recall@10 0.980, *exactly* dense's own, and the
  graph alone is at 0.950 — below dense, not at parity. There is nothing to harvest.
- **SPLADE alone is not significantly better than hybrid** (p=0.46), despite topping the
  table. The mixture costs nothing; it also gains nothing on this task.
- **BM25 is within 5.6 points of the best strategy at 1 ms**, against 420 ms for the graph.

Read with the caveat that the graph runs at the paper's defaults here. Its best swept
configuration reached recall@10 0.988, and a fusion with *that* graph is the open question
— though that configuration is also the one that leans hardest on the dense prior, so it is
partly fusing dense with dense.

## What the `activation_merge` ablation found

`graph.activation_merge` decides what happens when semantic bridging reaches the same entity
twice. `overwrite` is the reference implementation: the later hop replaces what was known,
so a seed re-reached at hop 2 keeps the weaker propagated score *and* a tier of 2, which
`score_passages` divides by. Instrumenting the cascade on 40 MedHop questions, **91% of
seeds end up demoted that way** — the entity the question is about is levelled with the ones
reached by ricochet. `best` keeps the max score and the min tier; the walk is untouched and
reaches exactly the same entities.

Paired, same collection and same query-time parameters:

| | Δ recall@10 | McNemar recall@10 | Δ hit@1 | Δ hop2@10 |
|---|---|---|---|---|
| MedHop, paper defaults | **+0.032** | **11 gained / 0 lost, p=0.001** | +0.003 | 0.000 |
| MedHop, it5/tk10 | +0.012 | 4 gained / 0 lost, p=0.125 | −0.006 | 0.000 |
| PubMedQA (single-hop control) | 0.000 | 1 gained / 1 lost, p=1.000 | 0.000 | — |

- **The gain is one-directional and free.** On the paper's own settings, eleven questions
  gain a gold passage in the top 10 and none loses one; on single-hop PubMedQA the change is
  inert in both directions. The direction is consistent across both MedHop configurations.
- **Ranking does not move.** hit@1 and MRR stay within noise, which matches the standing
  finding that the graph's weakness is ordering, not the candidate set.
- **It does not touch the multi-hop failure.** `hop2@10` is bit-identical in both
  configurations — zero discordant questions. Seed demotion was a real defect and is not
  what keeps hop-2 passages unreachable.

Related: `damping` was swept for the first time here and is a **non-event** on MedHop —
0.5, 0.95 and 0.99 give identical recall, MRR and `hop2@10` to four decimals. The reach of
the PageRank diffusion is not what bounds this task.

## What the HotpotQA run found

1,000 questions, 9,811 articles, `en_core_web_trf`, `all-mpnet-base-v2`, paper defaults,
4.1% graph fallback. Indexing: 281 s dense+sparse, 322 s graph-index (68,564 entities,
36,685 sentences, 121,090 edges, +321 MB). `all@k` is the metric the task defines — both
supporting articles in the top k — and `recall@k`, satisfied by the easier one, is near
ceiling for everything and says nothing.

| | hit@1 | all@2 | all@10 | hop2@10 | comparison@10 | median |
|---|---|---|---|---|---|---|
| BM25 | 0.773 | 0.244 | 0.734 | 0.711 | 1.000 | **0 ms** |
| dense | 0.789 | 0.290 | 0.676 | 0.669 | 0.984 | 36 ms |
| SPLADE | **0.884** | 0.384 | 0.803 | 0.771 | 1.000 | 36 ms |
| hybrid | 0.828 | 0.354 | 0.803 | 0.771 | 1.000 | 36 ms |
| graph | 0.711 | 0.326 | 0.763 | 0.787 | **0.963** | 197 ms |
| graph + dense | 0.795 | 0.378 | 0.833 | 0.842 | 1.000 | 244 ms |
| graph + BM25 | 0.789 | 0.339 | 0.854 | 0.848 | 1.000 | 201 ms |
| graph + hybrid | 0.800 | **0.396** | **0.870** | **0.875** | 1.000 | 252 ms |

Paired McNemar, exact two-sided, gained/lost:

| comparison | metric | Δ | discordants | p |
|---|---|---|---|---|
| graph+dense vs dense | all@10 | +0.157 | 185 / 28 | 1×10⁻²⁹ |
| graph+BM25 vs BM25 | all@10 | +0.120 | 159 / 39 | 2×10⁻¹⁸ |
| graph+hybrid vs hybrid | all@10 | +0.067 | 110 / 43 | 6×10⁻⁸ |
| graph+hybrid vs SPLADE | all@10 | +0.067 | 110 / 43 | 6×10⁻⁸ |
| graph+hybrid vs hybrid | hop2@10 | +0.103 | 95 / 20 | 7×10⁻¹³ |
| graph+hybrid vs SPLADE | hit@1 | **−0.084** | 50 / 134 | 5×10⁻¹⁰ |
| graph vs hybrid | hop2@10 | +0.015 | 98 / 87 | **0.46** |
| graph vs hybrid | bridge@10 | −0.023 | 5 / 24 | 5×10⁻⁴ |
| graph vs hybrid | comparison@10 | −0.037 | 0 / 7 | 0.016 |

- **Fusing the graph in helps here, and the same code on PubMedQA made things worse.** All
  three fusions beat their partner on `all@10`, and `graph+hybrid` beats SPLADE, the best
  single strategy. On PubMedQA every fusion landed *significantly below* its partner. Same
  RRF, same retrievers, opposite sign — which is the strongest evidence in this benchmark
  that the PubMedQA verdict was about the task and not about the method.
- **But the bare graph beats nothing.** It is last on hit@1, and significantly below hybrid
  on both `bridge@10` (p=5×10⁻⁴) and `comparison@10` (p=0.016). Its `hop2@10` of 0.787 is
  the highest of the single strategies and is **not** significantly above hybrid's 0.771
  (98 gained / 87 lost, p=0.46). Read the raw column and you would claim the graph is the
  best hop-2 retriever; the paired test says that is noise.
- **What the graph contributes is complementarity, not accuracy.** Fused into hybrid it
  gains 95 hop-2 passages and loses 20. It fails on different questions than the lexical
  and dense retrievers do, which is exactly what a fusion can exploit and what a
  head-to-head column cannot show.
- **The control behaves as predicted, with a caveat.** The graph alone is the only strategy
  below 1.000 on `comparison@10`, where nothing needs bridging — it costs where it cannot
  help. But it also loses on `bridge@10` in aggregate, so "the graph is good at bridge
  questions" is false as stated. What survives is narrower: it adds reach on the hop-2 leg
  when something else does the ranking.
- **The ranking weakness reproduces, and the two metrics disagree because of it.**
  `graph+hybrid` is 6.7 points *above* SPLADE on `all@10` and 8.4 points *below* it on
  hit@1, both highly significant. HotpotQA needs two passages and the graph pays for the
  second with the first one's rank. On `all@2` the advantage over SPLADE disappears
  entirely (+0.012, p=0.52): the contribution only shows at depth.
- **hybrid and SPLADE return the same top 10**, differing only in order — every `all@10`,
  `hop1@10` and `hop2@10` figure is identical. At depth 10 on this task the dense half of
  the mixture adds nothing.

Read as: on multi-hop retrieval the graph earns its place as a *component*, not as a
retriever. Which is the mirror image of the PubMedQA finding, and the reason both runs had
to exist.

## Known limits

- The graph is fused by reciprocal rank, which discards magnitudes. On PubMedQA that was
  exonerated (RRF lifted the bare graph by 13 points while every fusion lost); here it is
  untested in the other direction — a score-level blend might do better still.
- Nothing here is comparable to LinearRAG's published retrieval numbers, because there are
  none: **their release reports answer accuracy only**, and their `chunks.json` has no
  notion of a gold passage. Only the HotpotQA `answer` stage could ever be compared to
  them, and it has not been run.
- The graph was tuned across ten configurations; the hybrid baseline was left at its
  defaults. `retrieval.hybrid_search_weight` deserves the same sweep before the comparison
  is presented as even-handed.
- `max_iterations` has been swept on MedHop only (3 / 5 / 8, where 5 and 8 are identical to
  four decimals). It is untouched on PubMedQA and HotpotQA.
- PubMedQA questions were written from the article titles and share vocabulary with their
  abstracts, so dense retrieval starts from a high floor. That compresses the range in
  which a difference can show up.
- Timings are single-shot. For anything going in a report, run `index` a few times and take
  the median; the first run of all also pays for model downloads.
- The `answer` stage has not been exercised against a live API — there was no key on the
  machine when this was written. Everything else is run end to end.
