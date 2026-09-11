# Thesis idea: scalable graph RAG

## Title

Working title:

> **Extending RAG retrieval for multi-hop answering**


## Context

The standard RAG pipeline is a flat one. Documents are split into chunks and then each chunk is turned into a dense embedding
vector for semantic similarity or a sparse lexical one for term matching (BM25, or a learned
variant such as SPLADE). When the user asks a question to the LLM, the top-k chunks are scored against it, and those chunks are pasted into the
LLM's context as the evidence it must answer from. Generally, both semantic and lexical search are used to retrieve the chunks which
is known as hybrid retrieval.

Its assumption is that the answer lives in a chunk that *looks like the question*. That
holds for lookup questions and breaks as soon as the evidence is spread over several
chunks: no individual chunk resembles the question closely enough to be retrieved, so the
right passages never reach the LLM and each chunk is scored in isolation, with no notion
that two of them talk about the same entity.

Graph RAG retrieval tries to address this problem. Instead of indexing chunks as independent
points, it builds a structure over the corpus entities, relations and summaries that
records how passages connect, and retrieves by traversing that graph structure.

 

## Problem statement

Graph RAG is proposed as a fix for multi-hop and knowledge-intensive retrieval, but its
standard construction cost puts it out of reach at corpus scale, and its benefit over a
well-tuned hybrid baseline is rarely measured under equal conditions. This thesis asks
whether the construction cost can be removed without removing the benefit, and whether,
once it is removed, a benefit is still there at all.

Several kinds of graph RAG have been presented in the literature : **GraphRAG** (Microsoft) , **RAPTOR**, **HippoRAG**,**E²GraphRAG**.  They differ in *what
structure* they build over the corpus. However every of theses implementations relies on an LLM to extract the entities
and build the graph, which makes creating the graph expensive. A previous attempt to build a graph rag using 
the GraphRag library from microsoft was taking 20h to index with less accurate results than using hybrid retrieval.
Also note that retrieval time is also bigger since (find why when searching the paper here)

After research whether is it possible to build a graph rag without using an llm to construct the graph, 
the following paper has been found : https://arxiv.org/abs/2510.10114]**LinearRAG. It proposes a graph using **NER only** using Spacy, no LLM
extraction and no relation labels, then get the multi-hop behaviour back at query time
through activation and propagation over that graph.

### Main idea of how LinearRAG works

**The Tri-Graph.** Three levels of nodes (passages, sentences, entities) and only two
kinds of edges, both of them membership links: *sentence contains entity* and *passage
contains entity*. They are stored as two binary sparse matrices, `M` (sentences × entities)
and `C` (passages × entities), which are simply the adjacency matrices of that bipartite
graph. Crucially there is **no entity–entity edge** : The graph records *where* entities appear, never *how* they relate.

![The Tri-Graph: sentence–entity mentions (M, solid) and entity–passage containment (C, dashed)](img.png)


**Retrieval, stage 1: activation by semantic bridging** :

NER extracts the entities
contained in the query, and each of them is matched to its single closest graph entity by
embedding similarity. Their activation in the graph is the sparse vector `a⁰`: one
non-zero entry per query entity in the question `q`.

The second ingredient is the query-sentence relevance distribution `σ_q`. This vector represents the cosine similarity
between each sentence `s` in the sentence set `S` from the graph and the query `q`.

Activation is then propagated iteratively t times:

```
aᵗ = MAX( Mᵀ(σ_q ⊙ (M aᵗ⁻¹)), aᵗ⁻¹ )
```

aᵗ represents the activation vector of the entities in the t-th iteration of semantic propagation. allowing to 
identify a set of contextually relevant entities that anchors a subgraph in the corpus
that aligns with the reasoning structure of the query

**Retrieval, stage 2: global importance aggregation.** Stage 1 ends with `a_q`, the
activated entities and their scores. Stage 2 turns that into a ranking of passages, and it
changes graph to do so: propagation ran on the sentence-entity graph `M`, ranking runs on
the passage-entity graph `C`, a bipartite graph whose nodes are the passages `V_p` and the
entities `V_e`, with an edge wherever a passage contains an entity.

Every node of that graph is given a score.  Entity nodes take `a_q` directly, so an entity weighs what the propagation gave it.
Passage nodes are set by:

```
I(v) = ( λ · sim(q, v) + ln( 1 + Σ_{eᵢ ∈ E_a} a_q⁽ⁱ⁾ · ln(1 + N_eᵢ) / L_eᵢ ) ) · W_p
```

The first term is the passage's own dense similarity to the query. The second sums, over
the activated entities `E_a` the passage contains, their activation score `a_q⁽ⁱ⁾` weighted
by how often the entity occurs in the passage (`N_eᵢ`). A passage
concentrating a few strongly activated entities therefore starts higher than one mentioning
many weak ones. `λ` and `W_p` are constants.

These scores form the *teleport
distribution* of the random walk when running pageRank algorithm. at each step the walker follows an edge with probability
`d`, and with probability `1 - d` it lands back on a node drawn from that distribution. A node scoring zero there is never a landing
point and can only receive importance through edges. This is what makes the ranking specific
to the query.


A personalized PageRank is then run over the bipartite graph with those values, and the
passages are ranked by their converged score. The top-k among `V_p` are the retrieved
passages, and they go into the LLM prompt.


Multi-hop reasoning therefore happens over *sentences*, not over relation edges. The path
`Beatrice I → s₁ → Barbarossa → s₂ → Germany` is structurally a graph path; its edges just
happen to be whole sentences whose meaning was never compressed into a predicate.

## Research questions

### RQ1. Does graph structure recover the multi-hop evidence flat retrieval misses — as a replacement for the hybrid pipeline, or as an extension of it?

Dense is not a rival system here but a **special case** of the graph scorer: the passage
prior is `passage_ratio · dense + entity bonus`, so zeroing the bonus and the damping gives
back the dense ranking. Graph vs dense is therefore a nested ablation, and its delta
isolates what the structure contributes. The candidate pool gives the matching measurement:
the dense top-N *plus* every passage incident to an activated entity — the share of returned
passages coming from outside that pool is what the graph finds and dense does not.

Sub-questions:

- Where does the graph win over dense and over lexical retrieval, on which metric, and does
  the win survive at the *answer* level (LLM-as-judge) or only at recall@k?
- Are the graph and lexical signals complementary in a way dense and lexical are not?
- Is the graph better used in place of an existing signal, or fused on top of it? Under
  fusion, does it ever *degrade* a question flat retrieval already answered?

### RQ2. How far does the extension carry — what is the hop horizon of sentence-level bridging, and can it be pushed?

Semantic bridging propagates activation through sentences, not through relation edges, so
its reach is bounded by how the activation decays against the acceptance threshold rather
than by the nominal number of iterations. This RQ characterises that bound and asks whether
it can be moved without destroying the selectivity of the reset vector.

Sub-questions:

- What is the empirical distribution of query→answer distance in the Tri-Graph, per dataset,
  and where does activation stop firing?
- Is the binding constraint depth, NER coverage, or the multiplicative decay against a fixed
  threshold?
- Does a decay-aware threshold (per-tier, or normalised by tier depth) recover the second hop
  without flooding the activated set — and does the passage scorer's tier balance have to
  change with it, given that hop-1 passages saturate the top-k?

### RQ3. What does the extension cost, and what does an LLM-free construction give up?

Sub-questions:

- What does an LLM-free construction gain: indexing cost, determinism, no extraction
  hallucination? How does it scale with corpus size against an LLM extractor on the same
  corpus?
- What is the query-time cost of the extension, and is it proportionate to the gain?
- What can an LLM-built graph materialize that NER cannot? The Tri-Graph carries **no
  entity-entity edge**, so a taxonomy (the ICD chapter hierarchy) and a typed relation
  (*treats*, *contraindicated in*, *revised by*) are exactly what it cannot represent. How
  far does semantic bridging through sentences substitute for them in practice?

### RQ4. Which medical and humanitarian question shapes benefit from the extension?

There are spaCy models capable of processing biomedical/clinical text:
https://allenai.github.io/scispacy/

Sub-questions:

- Which question shapes separate the graph from flat retrieval: multi-hop chains, or
  aggregative one-to-many questions whose gold is a *set* of passages?
- Does domain NER (scispaCy) change the answer to RQ1 on a biomedical corpus?
- How can the graph's retrieval result be interpreted — and audited — in a medical context?



## Plan 

- Implement the graph rag
- Measure its retrieval quality (recall@k,hit@1,MRR,...)
- Answer quality (LLM as a judge)
- Measure retrieval and answer quality in a medical context and identify


Note that if time allows, there is a related LLM-free system worth looking at: **EHRAG**
(arXiv `2604.17458`, https://arxiv.org/pdf/2604.17458,
https://github.com/yfsong00/EHRAG). It is a separate implementation inspired by LinearRAG,
which it uses as a baseline rather than as a foundation, and it stays LLM-free at indexing
time (spaCy NER) and linear. Its addition is clustering: entity embeddings are grouped with
BIRCH into concept nodes, each linked to the entities nearest its centroid, which let the
activation jump between entities that are related but never co-occur in a sentence.
It reports outperforming LinearRAG.



## Use cases: medical and humanitarian



### Medical

Question shapes where the answer is split across documents:

- **Guideline question answering over a fragmented corpus.** WHO guidelines, national
  protocols and formularies, where answering needs the combination of two or several chunks. No single chunk contains a direct answer to the question. Examples of multi-hop
  questions :

  - *"Can the drug used for asthma attacks make the heart race?"* 
  - *"Can you drink grapefruit juice while on a cholesterol treatment?"* 
  - *"Is the usual treatment for severe acne safe during pregnancy?"
  - *"The patient is allergic to [...] — can they be given this antibiotic?"*

- **Aggregative, one-to-many questions.** A second shape, distinct from multi-hop: the
  corpus is indexed one way (a disease and its description) and the question runs the other
  way (a sign, and every condition that presents it), so the gold is a **set** of passages
  rather than one. Flat retrieval ranks by question-passage resemblance and returns the
  best-worded match; the graph activates the sign as an entity and, through `C`, reaches
  every passage containing it, ranking highest those that concentrate several activated
  entities. ICD-11 is the instrumented corpus for this shape
  (`benchmarks/graph_vs_hybrid/bench_icd11.yaml`). Examples:

  - *"Which diseases present with fever together with a skin rash in a child?"*
  - *"Which conditions cause progressive, painless loss of vision?"*

- **Auditability** Since graph construction require no LLM, there is no risk of hallucination during the graph construction
and the construction of the graph from the corpus is deterministic. the index records only where entities are mentioned, and every retrieved passage stays traceable to those mentions. Both matter when the corpus is patient data or when the answer feeds a clinical or operational decision.


### Humanitarian

- **Deployment cost, on the indexing side.** The graph is built with NER alone, so a
  corpus can be indexed on the hardware available on site, with no per-document API call
  and nothing sent to a provider.


