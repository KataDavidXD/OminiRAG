# OminiRAG -- Modular RAG Component Standardization

OminiRAG defines a **5-dimension RAG configuration space** where every
dimension is independently searchable. Three real-world RAG systems --
**LongRAG** (extractive QA), **LightRAG** (knowledge-graph-augmented
retrieval), and **Self-RAG** (LLM-based retrieval with evidence scoring) --
serve as **default presets**, and their internal components can be freely
swapped through a shared contract layer.

```text
Chunking -> Query -> Retrieval -> Post-Retrieval -> Generation
  (offline)          (corpus search)  (reranking/      (answer
                                       compression/     generation)
                                       critique)
```

Components from any framework can be injected into any pipeline via dependency
injection. All three pipelines share an **identical 4-node LangGraph topology**:

```text
query_processing -> retrieval -> post_retrieval -> generation -> END
```

An **AG-UCT search engine** explores the 5-dimension configuration space across
3 benchmark datasets to find optimal component combinations.

---

## Overall Architecture

```mermaid
graph TB
    subgraph Offline["Offline Layer (Chunking -> Index)"]
        C_SP["standard_passage<br/>(fixed-size chunks)"]
        C_LR["longrag_4k<br/>(graph-merged 4K units)"]
        C_KG["kg_extraction<br/>(entities + relations + chunks)"]
    end

    subgraph Retrieval_Methods["Real Retrieval Methods"]
        R_BM25["BM25Retrieval<br/>(rank_bm25)"]
        R_Dense["DenseRetrieval<br/>(sentence-transformers E5)"]
        R_Hybrid["HybridRetrieval<br/>(BM25 + Dense RRF)"]
        R_LightRAG["LightRAGRetrieval<br/>(vector + KG hybrid)"]
    end

    subgraph PostRetrieval["Post-Retrieval Methods"]
        PR_CE["CrossEncoderReranking<br/>(ms-marco MiniLM)"]
        PR_LC["LightRAGReranking<br/>(LLM context compression)"]
        PR_SC["SelfRAGReranking<br/>(generate+critique scoring)"]
        PR_ID["IdentityReranking<br/>(passthrough)"]
    end

    subgraph Protocols["rag_contracts (Protocol Layer)"]
        P_Q["Query protocol<br/>process(query, ctx) -> list[str]"]
        P_R["Retrieval protocol<br/>retrieve(queries, top_k) -> list[RR]"]
        P_RK["Reranking protocol<br/>rerank(query, results, top_k) -> list[RR]"]
        P_G["Generation protocol<br/>generate(query, context, inst) -> GR"]
    end

    subgraph Pipelines["LangGraph Pipelines (identical topology)"]
        longPipe["LongRAG build_graph()"]
        lightPipe["LightRAG build_query_graph()"]
        selfPipe["SelfRAG build_selfrag_modular_graph()"]
    end

    subgraph Adapters["Framework Adapters"]
        longA["LongRAGGeneration"]
        lightA["LightRAGQuery<br/>LightRAGRetrieval<br/>LightRAGReranking<br/>LightRAGGeneration"]
        selfA["SelfRAGReranking<br/>SelfRAGGeneration"]
        commonA["SimpleLLMGeneration"]
    end

    subgraph Utility["Utility / Benchmark Scaffolding"]
        utilA["LLMRetrieval<br/>DuckDuckGoRetrieval<br/>HFDatasetRetrieval<br/>ALCEDocRetrieval"]
    end

    subgraph Benchmarks["Benchmark Adapters"]
        BH["HotpotQA<br/>EM/F1"]
        BU["UltraDomain<br/>LLM-judge"]
        BA["ALCE<br/>F1/STR-EM"]
    end

    subgraph Search["AG-UCT Search"]
        UCT["RAGPipelineSearchState<br/>5 dims x N options<br/>build_pipeline_from_config()"]
    end

    Offline --> Retrieval_Methods
    Retrieval_Methods --> Protocols
    PostRetrieval --> Protocols
    Protocols --> Pipelines
    Adapters --> Protocols
    Pipelines --> Benchmarks
    UCT --> Adapters
    UCT --> Retrieval_Methods
    UCT --> PostRetrieval
    UCT --> Benchmarks
```



### Data Types (`rag_contracts/types.py`)


| Type               | Fields                                               |
| ------------------ | ---------------------------------------------------- |
| `RetrievalResult`  | `source_id`, `content`, `score`, `title`, `metadata` |
| `GenerationResult` | `output`, `citations`, `metadata`                    |
| `QueryContext`     | `topic`, `history`, `metadata`                       |
| `Document`         | `doc_id`, `content`, `metadata`                      |
| `Chunk`            | `chunk_id`, `doc_id`, `content`, `metadata`          |


---

## Core Idea

Each stage is a Python `Protocol` (structural interface). Any class that
implements the right method signature is automatically compatible -- no
inheritance required. Three projects with completely different architectures can
exchange components at any stage.

```text
                    rag_contracts (shared protocols)
                   +------------------------------+
                   |  Chunking   .chunk()          |
                   |  Embedding  .embed()          |
                   |  Query      .process()        |
                   |  Retrieval  .retrieve()       |
                   |  Reranking  .rerank()         |
                   |  Generation .generate()       |
                   +--+---------------+--------+---+
                      |               |        |
          +-----------v--+   +-------v------+ +v-----------+
          |   LongRAG    |   |   LightRAG   | |  Self-RAG  |
          |  (adapters)  |   |  (adapters)  | | (adapters) |
          +--------------+   +--------------+ +------------+
```

---

## Framework Presets and Benchmark Matrix

The three frameworks are now **default configuration presets** within the
5-dimension search space, not rigid pipelines:

| Preset | Chunking | Query | Retrieval | Post-Retrieval | Generation |
| --- | --- | --- | --- | --- | --- |
| **LongRAG** | `longrag_4k` | `identity` | `bm25` / `dense_e5` | `identity` | `longrag_reader` |
| **LightRAG** | `kg_extraction` | `lightrag_keywords` | `lightrag_hybrid` | `lightrag_compress` | `lightrag_answer` |
| **Self-RAG** | `standard_passage` | `identity` | `dense_e5` | `selfrag_critique` | `selfrag_generator` |

### Benchmarks

**HotpotQA**: Multi-hop QA over Wikipedia. Evaluation: EM, F1. Challenge: connecting facts across multiple passages.

**UltraDomain**: Domain-specific QA (agriculture, CS, legal). Evaluation: LLM-as-judge (Comprehensiveness, Diversity, Empowerment). Challenge: deep domain knowledge, entity relationships.

**ALCE**: Per-document segmented QA (ASQA, QAMPARI, ELI5). Evaluation: F1, STR-EM, citation recall/precision. Challenge: per-document evaluation, citation quality.

### How Each Preset Handles Each Benchmark

| Benchmark | LongRAG Preset | LightRAG Preset | Self-RAG Preset | Key Challenge |
| --- | --- | --- | --- | --- |
| **HotpotQA** | BM25/Dense over 4K chunks -> reader | KG hybrid retrieval over Wikipedia | Dense retrieval -> critique scoring -> best passage | Multi-hop fact connection |
| **UltraDomain** | Dense over 4K chunks | Native (hybrid vector+KG) | Dense retrieval -> per-passage scoring | Deep domain knowledge |
| **ALCE** | BM25/Dense over ALCE docs -> reader | KG index over source corpus | Dense retrieval -> per-passage scoring+selection | Per-document evaluation |

**Critical insight**: SelfRAG generation provides maximum value when retrieval returns MULTIPLE passages (for per-passage scoring+selection). Cross-encoder reranking is the standard precision improvement across all presets.

```mermaid
graph LR
    subgraph inputShapes["Retrieval Output Shapes"]
        SingleLong["Single long context<br/>(HFDatasetRetrieval: 1x 4K)"]
        MultiChunks["Multiple chunks<br/>(LightRAGRetrieval: N x variable)"]
        MultiDocs["Multiple documents<br/>(ALCE pre-retrieved: N x doc)"]
    end

    subgraph genBehavior["Generation Behavior"]
        LongGen["LongRAG/LightRAG Gen<br/>Concatenates all, generates once"]
        SelfGen["SelfRAG Gen<br/>Generates per-passage, scores, selects best"]
    end

    SingleLong -->|"works but<br/>no selection benefit"| SelfGen
    MultiChunks -->|"full benefit"| SelfGen
    MultiDocs -->|"full benefit"| SelfGen
    SingleLong -->|"natural fit"| LongGen
    MultiChunks -->|"concatenates"| LongGen
    MultiDocs -->|"concatenates"| LongGen
```



---

## 5-Dimension Configuration Space

### Available Components by Dimension

| Dimension | Component | Source | Notes |
| --- | --- | --- | --- |
| **Chunking** (offline) | `standard_passage` | -- | Fixed-size passage chunks (e.g. 512 tokens) |
| | `longrag_4k` | LongRAG | Graph-based document merging into 4K-token units |
| | `kg_extraction` | LightRAG | Entity/relation extraction -> KG + chunk stores |
| **Query** | `identity` | `rag_contracts` | Passthrough -- returns `[query]` |
| | `lightrag_keywords` | `lightrag_langgraph/adapters.py` | LLM keyword extraction -> `[query, kw1, kw2, ...]` |
| **Retrieval** (corpus search) | `bm25` | `rag_contracts/retrieval_methods.py` | Lexical matching via BM25Okapi. Fast, no GPU. |
| | `dense_e5` | `rag_contracts/retrieval_methods.py` | Semantic matching via multilingual-e5-small. |
| | `bm25_dense_hybrid` | `rag_contracts/retrieval_methods.py` | Reciprocal Rank Fusion of BM25 + Dense. |
| | `lightrag_hybrid` | `lightrag_langgraph/adapters.py` | Vector + KG hybrid. Requires `kg_extraction` chunking. |
| | `lightrag_graph` | `lightrag_langgraph/adapters.py` | KG-only traversal. Requires `kg_extraction` chunking. |
| **Post-Retrieval** | `identity` | `rag_contracts` | Passthrough -- returns results unchanged |
| | `cross_encoder` | `rag_contracts/reranking_methods.py` | Cross-encoder re-scoring (ms-marco-MiniLM-L-12-v2). Standard reranking. |
| | `lightrag_compress` | `lightrag_langgraph/adapters.py` | LLM context compression into focused evidence brief. |
| | `selfrag_critique` | `selfrag/adapters.py` | Per-passage generate+score (ISREL/ISSUP/ISUSE). Caches predictions. |
| **Generation** | `longrag_reader` | `longRAG_example/.../adapters.py` | LLM reader with NQ/HotpotQA prompts. Short answer extraction. |
| | `lightrag_answer` | `lightrag_langgraph/adapters.py` | LLM answer from structured context + optional compressed notes. |
| | `selfrag_generator` | `selfrag/adapters.py` | Per-passage generate+score+select. Reuses critique cache. |
| | `simple_llm` | `rag_contracts/common_components.py` | Generic LLM-based answer extraction. |

**Utility / benchmark scaffolding** (not in the search space):
`HFDatasetRetrieval`, `ALCEDocRetrieval`, `LLMRetrieval`, `DuckDuckGoRetrieval`, `FallbackRetrieval` -- these are used internally by benchmark adapters or as demo fallbacks, not as comparable retrieval methods.

### Combination Space

- Chunking: 3 options
- Query: 2 options
- Retrieval: 5 options (with constraint: `lightrag_hybrid`/`lightrag_graph` require `kg_extraction`)
- Post-Retrieval: 4 options
- Generation: 4 options

**Theoretical: 3 x 2 x 5 x 4 x 4 = 480 combinations (360 after constraint pruning).**

### Key Configurations (Framework Presets + Cross-Framework)

| # | Chunking | Query | Retrieval | Post-Retrieval | Generation | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | longrag_4k | identity | bm25 | identity | longrag_reader | LongRAG baseline |
| 2 | kg_extraction | lightrag_keywords | lightrag_hybrid | lightrag_compress | lightrag_answer | LightRAG baseline |
| 3 | standard_passage | identity | dense_e5 | selfrag_critique | selfrag_generator | Self-RAG baseline |
| 4 | standard_passage | identity | bm25 | cross_encoder | longrag_reader | Classic IR pipeline |
| 5 | standard_passage | identity | bm25_dense_hybrid | cross_encoder | longrag_reader | Hybrid + cross-encoder |
| 6 | standard_passage | identity | dense_e5 | cross_encoder | simple_llm | Dense + cross-encoder |
| 7 | kg_extraction | lightrag_keywords | lightrag_hybrid | cross_encoder | lightrag_answer | LightRAG retrieval + cross-encoder |
| 8 | standard_passage | identity | bm25_dense_hybrid | selfrag_critique | selfrag_generator | Hybrid + SelfRAG scoring |
| 9 | longrag_4k | identity | dense_e5 | cross_encoder | longrag_reader | LongRAG chunks + dense + CE |
| 10 | standard_passage | identity | bm25_dense_hybrid | lightrag_compress | lightrag_answer | Hybrid + LLM compression |

Integration tests in `tests/test_all_combinations.py` and `tests/test_retrieval_methods.py`.

### Pipeline Topology

All 3 pipeline builders share the same DI signature and produce identical
4-node topologies. The pipeline frame is inferred from the configuration
(not an explicit slot):

| Builder | State Type | DI Signature | Inferred When |
| --- | --- | --- | --- |
| `build_graph()` | `LongRAGGraphState` | `(retrieval, generation, reranking=None, query=None)` | Default (BM25/Dense retrieval) |
| `build_query_graph()` | `LightRAGGraphState` | `(retrieval, generation, reranking=None, query=None)` | `kg_extraction` + `lightrag_*` retrieval |
| `build_selfrag_modular_graph()` | `SelfRAGModularState` | `(retrieval, generation, reranking=None, query=None)` | `selfrag_critique` + `selfrag_generator` |

Identical DI interface means any component combination is structurally possible across all 3 pipeline frames.

---

## The 6 Replaceable Stages

All protocols live in `rag_contracts/protocols.py`. Data types live in
`rag_contracts/types.py`.

### Stage 1 -- Chunking

Split raw documents into retrieval units.

```python
class Chunking(Protocol):
    def chunk(self, documents: list[Document]) -> list[Chunk]: ...
```

**Real implementations:**

- LongRAG groups documents into 4K-token units using graph-based merging
- `SelfRAGChunking` -- MD5-based single-passage chunking with optional DocStore persistence
- LightRAG uses 1200-token chunks with 100-token overlap

### Stage 2 -- Embedding

Convert text chunks into vector representations.

```python
class Embedding(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**Real implementations:**

- LongRAG uses Tevatron/Contriever dense embeddings
- LightRAG embeds into 3 separate VectorStores (chunks, entities, relations)
- `SelfRAGEmbedding` -- Contriever mean-pool encoding with optional VectorStore persistence
- `IdentityEmbedding` returns empty vectors when a pipeline skips this stage

### Stage 3 -- Query

Expand or decompose a user query into retrieval-ready queries.

```python
class Query(Protocol):
    def process(self, query: str, context: QueryContext) -> list[str]: ...
```

**Real implementations:**

- `LightRAGQuery` -- LLM keyword extraction producing high/low-level keywords. Caches query result for downstream retrieval optimization.
- `IdentityQuery` -- returns the original query unchanged

### Stage 4 -- Retrieval

First-stage retrieval of candidate chunks from an indexed corpus.

```python
class Retrieval(Protocol):
    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]: ...
```

**Real retrieval methods** (comparable, used in search space):

- `BM25Retrieval` -- lexical matching via BM25Okapi (`rank_bm25` library). CPU-only, fast, interpretable.
- `DenseRetrieval` -- semantic matching via `intfloat/multilingual-e5-small` (sentence-transformers). 118M params, CPU-friendly.
- `HybridRetrieval` -- reciprocal rank fusion of BM25 + Dense. Surveys consistently show hybrid > either alone.
- `LightRAGRetrieval` -- hybrid vector+KG retrieval across 4 stores (chunks, entities, relations, graph). Requires KG-extracted index.

**Utility / benchmark scaffolding** (not comparable, not in search space):

- `HFDatasetRetrieval` -- pre-joined 4K context lookup from HuggingFace dataset (benchmark scaffolding)
- `ALCEDocRetrieval` -- wraps ALCE pre-retrieved documents (benchmark scaffolding)
- `LLMRetrieval` -- asks an LLM to fabricate background context (no corpus search)
- `DuckDuckGoRetrieval` -- web search via DuckDuckGo (no local corpus)
- `FallbackRetrieval` -- chains primary + fallback retrieval (combinator)

### Stage 5 -- Post-Retrieval

Second-stage processing of retrieval candidates: reranking, compression, or critique scoring.

```python
class Reranking(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], top_k: int = 10) -> list[RetrievalResult]: ...
```

**Implementations:**

- `CrossEncoderReranking` -- cross-encoder re-scoring via `cross-encoder/ms-marco-MiniLM-L-12-v2` (sentence-transformers). The standard reranking baseline.
- `LightRAGReranking` -- LLM context compression into a focused evidence brief. NOTE: This is context compression, not traditional cross-encoder reranking.
- `SelfRAGReranking` -- generates per-passage answers with logprob-based scoring (ISREL, ISSUP, ISUSE). Caches predictions in `metadata["_selfrag_pred"]` for downstream reuse.
- `IdentityReranking` -- returns results unchanged, truncated to `top_k`

### Stage 6 -- Generation

Produce final output from query + reranked context.

```python
class Generation(Protocol):
    def generate(self, query: str, context: list[RetrievalResult], instruction: str = "") -> GenerationResult: ...
```

**Real implementations:**

- `LongRAGGeneration` -- wraps LongRAG's GPT/Claude/Gemini readers with NQ and HotpotQA prompt templates
- `LightRAGGeneration` -- LLM answer generation from structured context + optional compressed notes
- `SelfRAGGeneration` -- generates per-passage candidate answers, scores them, returns the highest-scoring answer. Reuses `SelfRAGReranking` cache when available (0 extra LLM calls).
- `SimpleLLMGeneration` -- generic LLM-based answer extraction

---

## Self-RAG: The Post-Retrieval / Generation Cache

Self-RAG's scoring mechanism is fundamentally different from standard rerankers.
For each retrieved passage, the LLM generates a candidate answer and scores it
using logprob-based signals:

1. **ISREL** (relevance) -- probability of `[Relevant]` token
2. **ISSUP** (grounding) -- probability of `[Fully supported]` / `[Partially supported]`
3. **ISUSE** (utility) -- expected utility across 5 levels

When `SelfRAGReranking` and `SelfRAGGeneration` are used together in the same
pipeline, the reranking step caches its generated text and scores in
`metadata["_selfrag_pred"]`. Generation detects the cache and skips
re-generating, selecting the best answer directly. This reduces LLM calls from
2N to N for N passages:


| Reranking | Generation | Behavior                                                | LLM calls |
| --------- | ---------- | ------------------------------------------------------- | --------- |
| SelfRAG   | SelfRAG    | Rerank caches, gen reuses                               | N         |
| SelfRAG   | LongRAG    | Rerank scores+reorders, LongRAG reads reordered         | N + 1     |
| SelfRAG   | LightRAG   | Rerank scores+reorders, LightRAG compresses+answers     | N + 2     |
| Identity  | SelfRAG    | Gen generates from scratch, scores, selects             | N         |
| LightRAG  | SelfRAG    | Compression reranks, then SelfRAG generates per-passage | 1 + N     |


---

## How Component Swap Works in Real Code

### Step 1: Define a component that satisfies a Protocol

No base class needed -- just implement the right method signature (duck typing).

```python
from rag_contracts import RetrievalResult

class MyCustomRetrieval:
    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        return [RetrievalResult(source_id="...", content="...")]
```

This automatically satisfies `rag_contracts.Retrieval` because it has a
`.retrieve(queries, top_k)` method with the right signature. Verify at runtime:

```python
from rag_contracts import Retrieval
assert isinstance(MyCustomRetrieval(), Retrieval)  # True
```

### Step 2: Inject into any LangGraph pipeline via dependency injection

```python
from longRAG_example.longrag_langgraph.main_pipeline import build_graph
from lightrag_langgraph.main_pipeline import build_query_graph
from selfrag.modular_pipeline import build_selfrag_modular_graph

# Same components, any pipeline frame:
graph = build_graph(retrieval=my_ret, generation=my_gen, reranking=my_rr)
graph = build_query_graph(retrieval=my_ret, generation=my_gen, reranking=my_rr)
graph = build_selfrag_modular_graph(retrieval=my_ret, generation=my_gen, reranking=my_rr)

result = await graph.ainvoke({"query": "What is X?"})
```

### Step 3: Cross-project swap -- mix components from different frameworks

```python
from selfrag.modular_pipeline import build_selfrag_modular_graph
from selfrag.adapters import SelfRAGReranking
from rag_contracts import LLMRetrieval

# Use LLM retrieval + SelfRAG reranking + LongRAG reader inside Self-RAG's pipeline
compiled = build_selfrag_modular_graph(
    retrieval=LLMRetrieval(llm=my_llm),
    generation=LongRAGReaderGeneration(llm=my_llm),
    reranking=SelfRAGReranking(model=my_model, rel_tokens=rel_tokens, ...),
)
result = await compiled.ainvoke({"query": "Who wrote Hamlet?"})
```

---

## Running the Demos

### Prerequisites

```bash
pip install openai python-dotenv ddgs langgraph
```

Create a `.env` file:

```bash
LLM_API_KEY=sk-...
LLM_BASE_URL=https://...   # optional, for custom endpoints
DEFAULT_LLM=gpt-4o-mini
```

### Main Demo: 3-Framework Cross-Project Swap

```bash
python real_selfrag_swap_demo.py           # full demo (3 sections)
python real_selfrag_swap_demo.py --quick   # only section 1 (cross-swaps)
python real_selfrag_swap_demo.py --query "Your custom question here"
```

Runs 3 sections with live LLM calls:

**Section 1: Cross-Project Component Swaps** -- 6 configurations across all 3 pipeline frames:


| Config | Pipeline | Retrieval    | Reranking        | Generation        | What it demonstrates                                          |
| ------ | -------- | ------------ | ---------------- | ----------------- | ------------------------------------------------------------- |
| A      | Self-RAG | LLM context  | Identity         | SelfRAGGeneration | Native Self-RAG scoring                                       |
| B      | Self-RAG | DuckDuckGo   | Identity         | LongRAG reader    | Cross-gen: web retrieval + extractive reader in Self-RAG pipe |
| C      | LongRAG  | LLM context  | Identity         | SelfRAGGeneration | Cross-gen: Self-RAG scoring in LongRAG pipe                   |
| D      | LongRAG  | DDG+Fallback | SelfRAGReranking | LongRAG reader    | Full cross: web + evidence scoring + reader                   |
| E      | LightRAG | LLM context  | Identity         | SelfRAGGeneration | 3rd pipeline: Self-RAG scoring in LightRAG pipe               |
| F      | LightRAG | DDG+Fallback | SelfRAGReranking | LongRAG reader    | Full cross: all 3 frameworks in LightRAG pipe                 |


**Section 2: 3-Pipeline Identity Test** -- Identical components (LLM retrieval + LongRAG reader) through all 3 pipelines, proving they produce consistent results.

**Section 3: ALCE Benchmark Evaluation** -- Real F1/STR-EM scoring on ALCE sample data with 3 different generators + the graph_factory pipeline pattern.

### Sample Output

```text
========================================================================
  CROSS-PROJECT SWAP RESULTS
========================================================================
  Config Pipeline   Style           Chars Cites   Time    Score
  ------ ---------- --------------- ----- ----- ------ --------
  A      Self-RAG   selfrag           351     1  10.8s     2.46
  B      Self-RAG   longrag-reader    145     3   7.2s
  C      LongRAG    selfrag           349     1   9.8s     2.46
  D      LongRAG    longrag-reader    331     3  13.5s
  E      LightRAG   selfrag           361     1   9.8s     2.46
  F      LightRAG   longrag-reader    336     3  13.8s

========================================================================
  3-PIPELINE IDENTITY TEST
========================================================================
  Config Pipeline   Style           Chars Cites   Time    Score
  ------ ---------- --------------- ----- ----- ------ --------
  L      LongRAG    longrag-reader    208     1   7.9s
  R      LightRAG   longrag-reader    232     1   8.1s
  S      Self-RAG   longrag-reader    188     1   8.3s

========================================================================
  ALCE BENCHMARK SUMMARY
========================================================================
  Generator                     F1     EM  STR-EM  Words   Time
  ------------------------- ------ ------ ------- ------ ------
  LongRAG reader             41.2%   0.0%    0.0%    29   7.7s
  SelfRAG gen                37.5%   0.0%    0.0%    22  23.0s
  SimpleLLM gen              41.9%   0.0%    0.0%    30   6.4s
  LongRAG (pipeline)         44.7%   0.0%    0.0%    26   5.5s
```

### AG-UCT Pipeline Search

```bash
python -m uct_engine.examples.rag_pipeline_search          # simulated rewards
python -m uct_engine.examples.rag_pipeline_search --real    # real pipeline evaluation
```

Searches over the 5-dimension configuration space (chunking x query x retrieval x post_retrieval x generation) across 3 benchmark datasets using Monte Carlo tree search with cost-aware scoring. Hard constraint pruning ensures `lightrag_hybrid`/`lightrag_graph` are only paired with `kg_extraction` chunking.

### Real-LLM Benchmark Demo (HotpotQA / ALCE / UltraDomain)

End-to-end demo that loads **full HuggingFace datasets** through the
`Benchmark_Sampling` SDK, performs **stratified sampling**, and runs the
`SimpleLLMGeneration` reader against a real OpenAI-compatible API.

```bash
# Single benchmark
python demos/run_real_llm_benchmark.py --benchmark hotpotqa --budget 20
python demos/run_real_llm_benchmark.py --benchmark alce       --budget 20
python demos/run_real_llm_benchmark.py --benchmark ultradomain --budget 20

# All three (writes demos/results/benchmark_results.json)
python demos/run_real_llm_benchmark.py --all --budget 20
```

What the demo does for each benchmark:

| Stage | HotpotQA | ALCE | UltraDomain |
| --- | --- | --- | --- |
| Loader | `HotpotQAAPI` -> 97k items | `ALCEAPI.load_subset("asqa")` -> 948 items | `UltraDomainAPI.sample(strategy="balanced")` -> 698 items in 13 domains |
| Sampling | Proportional stratified by `(type, level)` | `random.sample` from ASQA | Balanced across `physics / cs / legal` |
| Pipeline | Identity Query -> context-from-distractor-passages -> Identity Reranking -> `SimpleLLMGeneration` | Identity Query -> `ALCEDocRetrieval` -> Identity Reranking -> `SimpleLLMGeneration` | Identity Query -> per-item context chunk -> Identity Reranking -> `SimpleLLMGeneration` |
| Metrics | EM / F1 (HotpotQA standard, per-stratum breakdown) | F1 / STR-EM / EM (ALCE standard, with citation strip) | LLM-as-judge (Comprehensiveness / Diversity / Empowerment, 1-5) |

**Latest run** (`gpt-4o-mini`, `budget=20`, `seed=42`, 80 LLM calls, ~52 k tokens, ~3 min):

| Benchmark | n | Primary metric | Secondary | Tokens | Time |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | 20 (16 bridge_hard + 4 comparison_hard) | **F1 51.2%**  EM 40.0% | bridge_hard F1 51.5% / comparison_hard F1 50.0% | 16,106 | 45.8 s |
| ALCE / asqa | 20 | **F1 6.1%**  STR-EM 37.1% | avg length 2.5 words (concise reader vs long-form gold) | 15,765 | 41.0 s |
| UltraDomain | 20 (balanced) | **C 2.15 / D 1.55 / E 2.10** (out of 5) | per-domain: physics 2.57/1.86/2.57, cs 2.00/1.43/2.00, legal 1.83/1.33/1.67 | 20,655 | 94.1 s |

Reading the numbers:

- HotpotQA's "hard" strata are recoverable with a vanilla reader on the gold distractor passages -- the stratified sampler concentrates budget on actually hard questions instead of the easy/medium population.
- ALCE shows the canonical short-vs-long-form mismatch: the OmniRAG `SimpleLLMGeneration` returns 2-3 word entity answers, so STR-EM (entity in answer) is ~37% but token-F1 against the long gold is only ~6%. Running ALCE with `LongRAGGeneration` or `SelfRAGGeneration` (long-form) closes most of that gap.
- UltraDomain's per-domain ranking (physics > cs > legal) exposes the reader's weakness: legal text needs longer, structured answers; the concise reader scores 1.83/1.33/1.67. The same fix as ALCE -- swap in a long-form generation component -- materially improves the Empowerment dimension.

### Benchmark Sampling SDK

The `Benchmark_Sampling/` package is an **independent, importable SDK** for
budget-aware benchmark evaluation. The OmniRAG demos consume only its public
APIs -- the SDK never reaches into OmniRAG.

| Module | Purpose |
| --- | --- |
| `bsamp.loader.{hotpot_qa,ALCE,UltraDomain,FreshWiki}` | HuggingFace loaders that read the local cache snapshots (`~/.cache/huggingface/hub/datasets--*`). |
| `bsamp.sampling.adapters.{hotpotqa,alce,ultradomain,freshwiki}` | Map raw rows to a canonical `BenchmarkItem(payload, target, metadata)`. |
| `bsamp.sampling.stratification` | `build_hotpotqa_config()`, `build_alce_config()`, `build_ultradomain_config()`. |
| `bsamp.sampling.samplers` | `StratifiedSampler` (proportional + Neyman) and `MetropolisHastingsSampler`. |
| `bsamp.sampling.engine.SamplingEngine` | High-level facade that wires adapter -> stratification -> sampler. |
| `bsamp.sampling.estimator` | Population-mean / variance estimators with stratification correction. |

```python
from bsamp.sampling.adapters.hotpotqa import HotpotQAAdapter
from bsamp.sampling.engine import SamplingEngine

adapter = HotpotQAAdapter(root_dir="~/.cache/huggingface/hub/datasets--hotpotqa--hotpot_qa/snapshots/<sha>")
engine = SamplingEngine(adapter=adapter, method="proportional", budget=200, seed=42)
result = engine.run()                         # -> SamplingResult

result.items            # list[BenchmarkItem] -- sampled items
result.strata_summary   # dict[str, int]      -- stratum -> population count
result.realization      # ItemRealization      -- allocation vector + realized item_ids
result.estimate         # Estimate | None      -- mean, std_error, ci_lower, ci_upper
result.state            # SamplingState        -- full serializable checkpoint
```

The `demos/run_real_llm_benchmark.py` script uses `SamplingEngine` for stratified
sampling and passes the drawn `BenchmarkItem` list through the OmniRAG
`*BenchmarkAdapter` evaluators. It supports `--generator {simple,longrag,selfrag}`
for head-to-head generation comparison.

---

## Running Tests

```bash
# All tests (86 tests)
python -m pytest tests/ -v

# AG-UCT engine tests
python -m pytest AG-UCT/uct_engine/tests/ -v
```

**Total: 86 offline tests + 9 real-LLM demo configurations + 4 ALCE benchmark evaluations.**

Tests verify:

- `rag_contracts` types, identity implementations, and `@runtime_checkable` protocol conformance
- Protocol conformance of all adapter classes (LightRAG, LongRAG, Self-RAG)
- Cross-project component swaps through real LangGraph execution (all 3 pipeline frames)
- All 14 benchmark-specific combinations (`test_all_combinations.py`)
- LightRAG cross-project swaps: LightRAG retrieval in LongRAG pipe, LongRAG retrieval in LightRAG pipe, SelfRAG reranking in LightRAG pipe, 3-way mixes (`test_lightrag_cross_swap.py`)
- Self-RAG reranking/generation cache mechanism (`test_selfrag_cache.py`)
- Benchmark adapters: HotpotQA, UltraDomain, ALCE with KG sample data (`test_benchmark_adapters.py`)
- LongRAG adapter behavior: NQ, HotpotQA, error handling (`test_longrag_adapters.py`)

---

## Project Structure

```text
ominirag/
+-- rag_contracts/                   # Shared protocol layer
|   +-- types.py                     #   Document, Chunk, RetrievalResult, GenerationResult, QueryContext
|   +-- protocols.py                 #   Chunking, Embedding, Query, Retrieval, Reranking, Generation
|   +-- identity.py                  #   Identity* passthrough implementations
|   +-- retrieval_methods.py         #   BM25Retrieval, DenseRetrieval, HybridRetrieval, CorpusIndex
|   +-- reranking_methods.py         #   CrossEncoderReranking
|   +-- common_components.py         #   Utility: LLMRetrieval, DuckDuckGoRetrieval, FallbackRetrieval,
|   |                                #   ALCEDocRetrieval, SimpleLLMGeneration (NOT comparable retrieval)
|   +-- component_registry.py        #   Canonical config-to-component builder (single source of truth)
|   +-- __init__.py
|
+-- longRAG_example/
|   +-- longrag_langgraph/           # LongRAG as modular LangGraph pipeline
|       +-- state.py                 #   LongRAGGraphState (TypedDict)
|       +-- main_pipeline.py         #   build_graph(retrieval, generation, reranking, query)
|       +-- adapters.py              #   HFDatasetRetrieval, LongRAGGeneration
|       +-- nodes/
|           +-- query_node.py
|           +-- retrieval_node.py
|           +-- reranking_node.py
|           +-- generation_node.py
|
+-- lightrag_langgraph/              # LightRAG as modular LangGraph pipeline
|   +-- state.py                     #   LightRAGGraphState (TypedDict, includes query_result)
|   +-- main_pipeline.py             #   build_query_graph(), build_index_graph()
|   +-- adapters.py                  #   LightRAGQuery, LightRAGRetrieval, LightRAGReranking,
|   |                                #   LightRAGGeneration
|   +-- nodes/
|       +-- query_node.py            #   Caches query_result for retrieval optimization
|       +-- retrieval_node.py        #   Passes pre-computed query_result to LightRAGRetrieval
|       +-- reranking_node.py
|       +-- generation_node.py
|
+-- self-rag_langgraph/self-rag-wtb/
|   +-- selfrag/
|       +-- constants.py             #   Control tokens, PROMPT_DICT, load_special_tokens()
|       +-- adapters.py              #   Forward + reverse adapters (Self-RAG <-> canonical)
|       +-- modular_pipeline.py      #   build_selfrag_modular_graph() -- canonical DI graph
|       +-- state.py                 #   SelfRAGModularState
|       +-- nodes/                   #   Canonical query/retrieval/reranking/generation nodes
|       +-- graph_query.py           #   Original Self-RAG query pipeline (vLLM-native)
|       +-- graph_query_longform.py  #   Beam search long-form pipeline
|       +-- graph_index.py           #   Indexing pipeline (chunk -> embed)
|
+-- A-Simplified-Core-Workflow-for-Enhancing-RAG/
|   +-- lightrag_core_simplified/src/
|       +-- config.py                #   LightRAG configuration
|       +-- modules/                 #   query_module, retrieval_module, reranking_module,
|                                    #   generation_module (split from monolithic retrieval)
|
+-- AG-UCT/uct_engine/
|   +-- examples/
|       +-- rag_pipeline_search.py   #   UCT search over 5-dimension RAG config space
|                                    #   build_pipeline_from_config() maps to real adapters
|
+-- ominirag_wtb/                    # WTB integration layer for cache-aware evaluation
|   +-- config_types.py              #   RAGConfig, BenchmarkQuestion, WorkItem, VALID_* sets
|   +-- graph_factories.py           #   config_to_graph_factory(), frame inference
|   +-- batch_runner.py              #   run_batch_with_reuse() orchestration
|   +-- cache_aware_evaluator.py     #   AG-UCT Evaluator ABC bridge to WTB execution
|   +-- reuse_ledger.py              #   ReuseLedger, MaterializedEntry for bipartite cache
|   +-- __init__.py                  #   Public API exports
|
+-- benchmark/
|   +-- base_adapter.py              #   Shared helpers: sample_chunks_to_retrieval_results, invoke_graph_sync
|   +-- hotpotqa_adapter.py          #   HotpotQA evaluation (EM/F1)
|   +-- ultradomain_adapter.py       #   UltraDomain evaluation (LLM-judge)
|   +-- alce_adapter.py              #   ALCE evaluation (F1/STR-EM), supports graph_factory
|   +-- sample_data/                 #   KG-format samples for all 3 benchmarks
|       +-- hotpotqa_kg_sample/      #   chunks, graph, kv, vdb_*, queries
|       +-- ultradomain_kg_sample/   #   chunks, graph, kv, vdb_*, queries
|       +-- alce_kg_sample/          #   alce_docs, graph, kv, queries
|
+-- Benchmark_Sampling/              # Independent SDK for HF dataset loading + budgeted sampling
|   +-- benchmark/
|   |   +-- loader/                  #   HotpotQA / ALCE / UltraDomain / FreshWiki HF loaders
|   |   +-- sampling/
|   |       +-- adapters/            #   Raw-row -> BenchmarkItem adapters per dataset
|   |       +-- samplers/            #   StratifiedSampler, MetropolisHastingsSampler
|   |       +-- engine.py            #   SamplingEngine facade (plan -> sample -> estimate)
|   |       +-- stratification.py    #   build_*_config() helpers
|   |       +-- estimator.py         #   Population-mean/variance with stratification correction
|   |       +-- types.py             #   BenchmarkItem, SamplingPlan, ...
|   +-- tests/                       #   Unit + real-data tests for the SDK
|
+-- demos/
|   +-- run_real_llm_benchmark.py    #   Real-LLM HotpotQA/ALCE/UltraDomain demo (full HF datasets)
|   +-- run_benchmark_demo.py        #   KG sample-data demo (no HF cache required)
|   +-- results/
|       +-- benchmark_results.json   #   Latest cross-benchmark scoring report
|
+-- tests/
|   +-- test_rag_contracts.py        #   Protocol + type tests
|   +-- test_retrieval_methods.py    #   BM25/Dense/Hybrid/CrossEncoder on sample_data
|   +-- test_all_combinations.py     #   Benchmark-specific combinations
|   +-- test_cross_project_swap.py   #   Cross-project swaps via LangGraph
|   +-- test_lightrag_cross_swap.py  #   LightRAG cross-framework swaps
|   +-- test_selfrag_cache.py        #   SelfRAG reranking/generation cache
|   +-- test_benchmark_adapters.py   #   Benchmark adapters + KG sample data
|   +-- test_longrag_adapters.py     #   LongRAG adapter behavior
|
+-- real_selfrag_swap_demo.py        #   Cross-project swap demo: 3-section real LLM test
+-- .env                             #   LLM_API_KEY, LLM_BASE_URL, DEFAULT_LLM
```

---

## Data Reality Map

Where each artifact actually comes from in the running system. Use this table
when reading test results -- "real" vs "sample" is the difference between a
diagnostic signal and a smoke test.

| Surface | Data source | Scale | LLM calls | Used by |
| --- | --- | --- | --- | --- |
| `tests/test_benchmark_adapters.py` | KG sample JSONs in `benchmark/sample_data/` | 3-10 items per benchmark | Mocked or fake `llm.complete` | Unit-level adapter contract tests |
| `tests/test_all_combinations.py` | KG sample data | 3-10 items per benchmark | Mocked LLMs in fixtures | Combination matrix verification |
| `Benchmark_Sampling/tests/test_real_data_sampling.py` | Real HuggingFace caches | 97k HotpotQA / 948 ASQA / 698 UltraDomain (3 domains) | None (sampling only) | SDK behavior on real distributions |
| `demos/run_benchmark_demo.py` | KG sample data | 3-10 items per benchmark | Real LLM (`SimpleLLMGeneration`) | Quick smoke check |
| `demos/run_real_llm_benchmark.py` | Real HuggingFace caches | Configurable budget (default 20/benchmark) | Real LLM, real metrics | End-to-end signal report |
| `real_selfrag_swap_demo.py` | LLM-generated context + DuckDuckGo + ALCE sample docs | 1-3 queries x 6 configs | Real LLM | Cross-framework swap correctness |
| `AG-UCT/.../rag_pipeline_search.py` | KG sample data (default) or HF datasets via `--real --use-hf` | UCT controlled | Optional, depending on flags | RAG configuration search |

Things that are **not** real yet, tracked as gaps:

- BM25 / Dense / Hybrid retrieval methods are implemented and tested on KG sample data (5-10 chunks). Full-corpus indexing against Wikipedia or UltraDomain requires a separate offline step.
- CrossEncoder reranking downloads the model on first use (~50MB). It runs on CPU but is slower than BM25.
- LongRAG's full Wikipedia 4K-chunk corpus is wired through `HFDatasetRetrieval` (benchmark scaffolding) but real retrieval should use `BM25Retrieval` or `DenseRetrieval` over indexed chunks.
- LightRAG's KG indexer (`build_index_graph()`) builds against the small KG sample by default; building against a full UltraDomain corpus requires a separate indexing run.
- Self-RAG's logprob scoring (ISREL/ISSUP/ISUSE) requires a local vLLM model with `selfrag_llama2_7b` weights. Cross-project demos shim `vllm`/`torch` so the *interface* is exercised, but scores are illustrative.
- ALCE's heavy metrics (AutoAIS NLI, MAUVE, QA-pipeline) need GPU models and are intentionally skipped.

---

## Adding a New Component

To add a new Retrieval implementation (for example, a vector database):

```python
from rag_contracts import RetrievalResult

class PineconeRetrieval:
    def __init__(self, index_name: str):
        import pinecone
        self.index = pinecone.Index(index_name)

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        results = []
        for q in queries:
            matches = self.index.query(vector=embed(q), top_k=top_k)
            for m in matches:
                results.append(RetrievalResult(
                    source_id=m.id,
                    content=m.metadata["text"],
                    score=m.score,
                ))
        return results[:top_k]
```

Then plug it into any pipeline:

```python
graph = build_graph(
    retrieval=PineconeRetrieval("my-index"),
    generation=LongRAGReaderGeneration(llm),
)
```

No other code changes required. The protocol contract guarantees compatibility.
