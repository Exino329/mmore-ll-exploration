# Thesis idea: scalable graph RAG

## Title

Working title:

> **Graph RAG: LLM-free graph construction for efficient retrieval**


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

### RQ1. What are the trade-offs of an LLM-free graph construction against an LLM-constructed one?


Sub-questions:

- What does an LLM-free construction gain: cost, determinism, no extraction hallucination?
- How does construction cost scale with corpus size, and what would the same corpus cost
  with an LLM extractor?
- What can an LLM-built graph materialize that NER cannot?


### RQ2. How does LLM-free graph retrieval compare with lexical and dense retrieval, and how do the three signals combine?

Dense is not a rival system here but a **special case** of the graph scorer: the passage
prior is `passage_ratio · dense + entity bonus`, so zeroing the bonus and the damping gives
back the dense ranking. Graph vs dense is therefore a nested ablation, and its delta
isolates what the structure contributes. The candidate pool gives the matching measurement,
being the dense top-N *plus* every passage incident to an activated entity: the share of
returned passages coming from outside that pool is what the graph finds and dense does not.

Sub-questions:

- Where does the graph win over dense and over lexical retrieval, and on which metric?
- Are the graph and lexical signals complementary in a way dense and lexical are not?
- Is the graph better used in place of an existing signal, or fused on top of it?


### RQ3. Which medical retrieval tasks can benefit using a graphRag?


There are spaCy models capable of processing biomedical/clinical text:
https://allenai.github.io/scispacy/

Sub-questions:

- Which medical question shapes separate the graph from flat retrieval?
- How can we interpret graph's retrieval results in a medical context ?




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
