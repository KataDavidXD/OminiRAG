# OminiRAG -- Modular RAG Component Standardization

OminiRAG defines a **canonical 6-stage RAG pipeline** where every stage is a
replaceable component. Three real-world RAG systems -- **LongRAG** (extractive
QA), **STORM** (Wikipedia-style article generation), and **Self-RAG** (LLM-based
retrieval with evidence scoring) -- are refactored so that their internal
components can be freely swapped with each other through a shared contract layer.

```
Chunking -> Embedding -> Query -> Retrieval -> Reranking -> Generation
```

---

## Core Idea

Each stage is a Python `Protocol` (structural interface). Any class that
implements the right method signature is automatically compatible -- no
inheritance required. Three projects with completely different architectures can
exchange components at any stage.

```
                    rag_contracts (shared protocols)
                   ┌─────────────────────────────┐
                   │  Chunking   .chunk()         │
                   │  Embedding  .embed()         │
                   │  Query      .process()       │
                   │  Retrieval  .retrieve()      │
                   │  Reranking  .rerank()        │
                   │  Generation .generate()      │
                   └──┬──────────────┬────────┬───┘
                      │              │        │
          ┌───────────▼──┐   ┌──────▼─────┐ ┌▼──────────┐
          │   LongRAG    │   │   STORM    │ │  Self-RAG  │
          │  (adapters)  │   │ (adapters) │ │ (adapters) │
          └──────────────┘   └────────────┘ └────────────┘
```

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


| Data type  | Fields                                      |
| ---------- | ------------------------------------------- |
| `Document` | `doc_id`, `content`, `metadata`             |
| `Chunk`    | `chunk_id`, `doc_id`, `content`, `metadata` |


**Real implementations:**

- LongRAG groups documents into 4K-token units using graph-based merging
(`longRAG_example/preprocess/group_documents.py`)
- STORM receives pre-chunked web snippets from search APIs
- `SelfRAGChunking` -- MD5-based single-passage chunking with optional DocStore
persistence (`self-rag_langgraph/self-rag-wtb/selfrag/adapters.py`)

### Stage 2 -- Embedding

Convert text chunks into vector representations.

```python
class Embedding(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

**Real implementations:**

- LongRAG uses Tevatron/Contriever dense embeddings for retrieval
- STORM uses TF-IDF cosine similarity inside `StormInformationTable`
- `SelfRAGEmbedding` -- Contriever mean-pool encoding with optional VectorStore
persistence (`self-rag_langgraph/self-rag-wtb/selfrag/adapters.py`)
- `IdentityEmbedding` returns empty vectors when a pipeline skips this stage

### Stage 3 -- Query

Expand or decompose a user query into multiple retrieval-ready queries.

```python
class Query(Protocol):
    def process(self, query: str, context: QueryContext) -> list[str]: ...
```


| Data type      | Fields                         |
| -------------- | ------------------------------ |
| `QueryContext` | `topic`, `history`, `metadata` |


**Real implementations:**

- STORM's persona-driven query expansion (`StormQueryAdapter` in
`storm/storm_langgraph/adapters.py`): a simulated expert asks a research
question, then `LLMQueryGenerator` decomposes it into 5 diverse search queries
- `IdentityQuery` returns the original query unchanged (used by LongRAG, which
does single-query retrieval)

### Stage 4 -- Retrieval

First-stage retrieval of candidate chunks.

```python
class Retrieval(Protocol):
    def retrieve(
        self, queries: list[str], top_k: int = 10
    ) -> list[RetrievalResult]: ...
```


| Data type         | Fields                                               |
| ----------------- | ---------------------------------------------------- |
| `RetrievalResult` | `source_id`, `content`, `score`, `title`, `metadata` |


**Real implementations:**

- `DuckDuckGoRetrieval` -- web search via DuckDuckGo (`real_swap_demo.py`)
- `LLMRetrieval` -- asks an LLM to generate background context (`real_swap_demo.py`)
- `HFDatasetRetrieval` -- looks up pre-joined context from the HuggingFace
`TIGER-Lab/LongRAG` dataset (`longRAG_example/longrag_langgraph/adapters.py`)
- `StormRetrievalAdapter` -- wraps STORM's `DuckDuckGoRetriever` / `YouRetriever`
to return `RetrievalResult` instead of STORM's `Information`
(`storm/storm_langgraph/adapters.py`)
- `FallbackRetrieval` -- tries primary retrieval first, falls back to an
alternative if 0 results are returned (`real_swap_demo.py`)
- `SelfRAGRetrieval` -- Contriever-encoded query + VectorStore cosine search
+ DocStore passage lookup (`self-rag_langgraph/self-rag-wtb/selfrag/adapters.py`)

### Stage 5 -- Reranking

Second-stage reordering or filtering of retrieval candidates.

```python
class Reranking(Protocol):
    def rerank(
        self, query: str, results: list[RetrievalResult], top_k: int = 10
    ) -> list[RetrievalResult]: ...
```

**Real implementations:**

- `StormRerankingAdapter` -- wraps `StormInformationTable.retrieve_information`
(cosine similarity over TF-IDF vectors) as a canonical reranker
(`storm/storm_langgraph/adapters.py`)
- `SelfRAGReranking` -- generates per-passage answers with the vLLM model and
computes Self-RAG logprob scores (relevance + grounding + utility) to reorder
passages. Internally generates but only returns the ranking
(`self-rag_langgraph/self-rag-wtb/selfrag/adapters.py`)
- `IdentityReranking` -- returns results unchanged, truncated to `top_k` (used
when a pipeline has no dedicated reranking step)

### Stage 6 -- Generation

Produce final output from query + reranked context.

```python
class Generation(Protocol):
    def generate(
        self,
        query: str,
        context: list[RetrievalResult],
        instruction: str = "",
    ) -> GenerationResult: ...
```


| Data type          | Fields                            |
| ------------------ | --------------------------------- |
| `GenerationResult` | `output`, `citations`, `metadata` |


**Real implementations:**

- `LongRAGReaderGeneration` -- concise extractive QA reader
(`real_swap_demo.py`)
- `LongRAGGeneration` -- wraps LongRAG's `GPTInference` / `ClaudeInference`
reader with NQ and HotpotQA prompt templates
(`longRAG_example/longrag_langgraph/adapters.py`)
- `StormWriterGeneration` -- produces a Wikipedia-style article section with
inline citations (`real_swap_demo.py`)
- `StormGenerationAdapter` -- wraps STORM's `LLMSectionWriter` to return
`GenerationResult` (`storm/storm_langgraph/adapters.py`)
- `SelfRAGGeneration` -- generates per-passage candidate answers, scores them
using Self-RAG's logprob-based relevance/grounding/utility signals, and
returns the highest-scoring answer as `GenerationResult`
(`self-rag_langgraph/self-rag-wtb/selfrag/adapters.py`)

---

## How Component Swap Works in Real Code

### Step 1: Define a component that satisfies a Protocol

No base class needed -- just implement the right method signature (duck typing).

```python
from rag_contracts import RetrievalResult

class MyCustomRetrieval:
    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        # your implementation here
        return [RetrievalResult(source_id="...", content="...")]
```

This automatically satisfies `rag_contracts.Retrieval` because it has a
`.retrieve(queries, top_k)` method with the right signature. You can verify at
runtime:

```python
from rag_contracts import Retrieval
assert isinstance(MyCustomRetrieval(), Retrieval)  # True
```

### Step 2: Inject into a LangGraph pipeline via dependency injection

The pipeline's `build_graph()` accepts component instances as arguments:

```python
from longRAG_example.longrag_langgraph.main_pipeline import build_graph

graph = build_graph(
    retrieval=MyCustomRetrieval(),       # swap this
    generation=MyCustomGeneration(),     # swap this
    reranking=MyCustomReranking(),       # swap this (optional)
)

result = await graph.ainvoke({"query": "What is X?", ...})
```

Internally, each LangGraph node receives its component via closure:

```python
# longRAG_example/longrag_langgraph/nodes/retrieval_node.py
def build_node(retrieval: Retrieval, reranking: Reranking | None = None):
    _reranking = reranking or IdentityReranking()

    async def node(state):
        query = state["query"]
        results = retrieval.retrieve([query], top_k=10)     # <-- uses injected component
        reranked = _reranking.rerank(query, results, top_k=10)
        return {"retrieval_results": reranked}

    return node
```

### Step 3: Cross-project swap -- use STORM's component inside LongRAG

```python
from longRAG_example.longrag_langgraph.main_pipeline import build_graph
from rag_contracts import IdentityReranking

# STORM's DuckDuckGo web retrieval + LongRAG's reader
graph = build_graph(
    retrieval=DuckDuckGoRetrieval(k=5),       # from STORM
    generation=LongRAGReaderGeneration(llm),   # from LongRAG
    reranking=IdentityReranking(),
)
```

### Step 4: Adapt existing project-internal components

When a project has its own types (e.g., STORM uses `Information` instead of
`RetrievalResult`), write a thin adapter:

```python
# storm/storm_langgraph/adapters.py
@dataclass
class StormRetrievalAdapter:
    storm_retriever: Any  # STORM's DuckDuckGoRetriever

    def retrieve(self, queries: list[str], top_k: int = 10) -> list[RetrievalResult]:
        infos = self.storm_retriever.retrieve(queries)        # returns Information[]
        return [information_to_retrieval_result(i) for i in infos[:top_k]]  # -> RetrievalResult[]
```

Reverse adapters allow canonical components to be used inside STORM's pipeline:

```python
@dataclass
class CanonicalToStormRetriever:
    canonical_retrieval: Any  # any rag_contracts.Retrieval

    def retrieve(self, queries: list[str], exclude_urls=None) -> list[Information]:
        results = self.canonical_retrieval.retrieve(queries, top_k=10)  # RetrievalResult[]
        return [retrieval_result_to_information(r) for r in results]    # -> Information[]
```

---

## Running the Demos

### Prerequisites

```bash
# Set up environment
echo "LLM_API_KEY=sk-..." > .env
echo "LLM_BASE_URL=https://..." >> .env   # optional, for custom endpoints
echo "DEFAULT_LLM=gpt-4o-mini" >> .env

pip install openai python-dotenv ddgs
```

### Demo 1: LongRAG + STORM swap (4 configurations)

```bash
python real_swap_demo.py
```

Builds the LongRAG pipeline 4 times with different component combinations:

| Config | Retrieval          | Generation     | Output style                    |
| ------ | ------------------ | -------------- | ------------------------------- |
| A      | LLM context        | LongRAG reader | Concise extractive answer       |
| B      | DuckDuckGo web     | LongRAG reader | Web-grounded answer             |
| C      | LLM context        | STORM writer   | Structured Wikipedia article    |
| D      | DDG + LLM fallback | STORM writer   | Web-cited article with fallback |

### Demo 2: Self-RAG + LongRAG + STORM 3-way swap (6 configurations)

```bash
python real_selfrag_swap_demo.py
```

Exercises both the Self-RAG modular pipeline and the LongRAG pipeline, mixing
components from all three systems:

| Config | Pipeline | Retrieval          | Reranking        | Generation           | Output style                    |
| ------ | -------- | ------------------ | ---------------- | -------------------- | ------------------------------- |
| A      | Self-RAG | LLM context        | Identity         | SelfRAGGeneration    | Scored extractive answer        |
| B      | Self-RAG | DuckDuckGo web     | Identity         | LongRAG reader       | Web-grounded concise answer     |
| C      | Self-RAG | LLM context        | Identity         | STORM writer         | Wikipedia-style article section |
| D      | LongRAG  | LLM context        | Identity         | SelfRAGGeneration    | Self-RAG scored answer          |
| E      | LongRAG  | DDG + LLM fallback | SelfRAGReranking | LongRAG reader       | Reranked web-grounded answer    |
| F      | Self-RAG | DDG + LLM fallback | SelfRAGReranking | STORM writer         | Full 3-way mix article          |

### Demo 3: Self-RAG native LLM test (4 configurations)

```bash
python self-rag_langgraph/self-rag-wtb/real_llm_test.py
```

Tests the original Self-RAG pipeline (query, no-retrieval, long-form, WTB
execution) with a real LLM simulating the vLLM interface via OpenAI chat
completions.

### Demo Q&A format and data

All demos use a **single open-ended factual question** as input (configurable
via the `DEMO_QUERY` environment variable). The default question is:

> *What were the main causes and consequences of the 2023 Silicon Valley Bank collapse?*

No pre-loaded document corpus is needed. Context is gathered at runtime through
one of two retrieval strategies:

- **LLM-generated context**: the LLM itself produces background passages when
  asked for factual information about the query topic.
- **DuckDuckGo web search**: live web search via the `ddgs` library, returning
  snippets with URLs as citations.

Each retrieval strategy returns a `list[RetrievalResult]` -- the canonical data
type shared across all systems:

```python
@dataclass
class RetrievalResult:
    source_id: str          # URL or synthetic ID like "llm-context://0"
    content: str            # passage text
    score: float = 0.0      # relevance score
    title: str = ""         # passage title
    metadata: dict = field(default_factory=dict)
```

Generation output is always a `GenerationResult`:

```python
@dataclass
class GenerationResult:
    output: str             # the answer or article section
    citations: list[str]    # source_ids of passages used
    metadata: dict = field(default_factory=dict)  # e.g. {"style": "longrag-reader"}
```

The three generation styles produce visibly different outputs from the same
query and context:

| Style           | Component              | Typical output                                |
| --------------- | ---------------------- | --------------------------------------------- |
| LongRAG reader  | `LongRAGReaderGeneration` | 1--3 sentence concise answer                  |
| STORM writer    | `StormWriterGeneration`   | Multi-paragraph Wikipedia section with `[1]` citations |
| Self-RAG scorer | `SelfRAGGeneration`       | Short answer with `selfrag_score` metadata    |

### Sample demo output

```
========================================================================
  COMPARISON SUMMARY
========================================================================
  A) Self-RAG native (LLM + SelfRAG gen)              366 chars, 1 cites  score=2.46
  B) Self-RAG pipe (DDG + LongRAG reader)              333 chars, 3 cites
  C) Self-RAG pipe (LLM + STORM writer)               2466 chars, 1 cites
  D) LongRAG pipe (LLM + SelfRAG gen)                  428 chars, 1 cites  score=2.46
  E) LongRAG pipe (DDG + SelfRAG rerank + reader)      333 chars, 3 cites
  F) 3-way mix (DDG + SelfRAG rerank + STORM)          1282 chars, 3 cites
```

---

## Running Tests

```bash
# Core tests (rag_contracts, adapters, cross-project swaps)
python -m pytest tests/ -v

# Self-RAG modular swap tests (20 tests: protocol conformance + 8 cross-project configs)
python self-rag_langgraph/self-rag-wtb/test_modular_swap.py

# Self-RAG WTB integration tests (7 tests)
python self-rag_langgraph/self-rag-wtb/test_wtb_integration.py

# AG-UCT engine tests
python -m pytest AG-UCT/uct_engine/tests/ -v
```

**Total: 41 offline tests + 14 real-LLM demo configurations.**

Tests verify:

- `rag_contracts` types, identity implementations, and `@runtime_checkable` protocol conformance
- Protocol conformance of actual Self-RAG adapter classes (`SelfRAGRetrieval`, `SelfRAGReranking`, `SelfRAGGeneration`, `SelfRAGEmbedding`, `SelfRAGChunking`) -- not just mocks
- STORM adapter round-trips (`Information` <-> `RetrievalResult`)
- LongRAG adapter behavior (NQ, HotpotQA, error handling)
- Self-RAG adapter passage conversion round-trips
- Cross-project component swaps through real LangGraph execution (all 3 systems)
- Self-RAG modular pipeline with 8 swap configurations (LongRAG/STORM/Self-RAG mix)
- Reverse adapters (`CanonicalToSelfRAGRetrieval`, `CanonicalToSelfRAGGeneration`)

---

## Project Structure

```
ominirag/
├── rag_contracts/              # Shared protocol layer
│   ├── types.py                #   Document, Chunk, RetrievalResult, GenerationResult, QueryContext
│   ├── protocols.py            #   Chunking, Embedding, Query, Retrieval, Reranking, Generation
│   ├── identity.py             #   IdentityChunking, IdentityQuery, IdentityEmbedding, IdentityReranking, IdentityGeneration
│   └── __init__.py
│
├── longRAG_example/
│   └── longrag_langgraph/      # Refactored LongRAG as modular LangGraph
│       ├── state.py            #   LongRAGGraphState (TypedDict)
│       ├── main_pipeline.py    #   build_graph(retrieval, generation, reranking, query)
│       ├── nodes/
│       │   ├── query_node.py       # DI-based query processing node
│       │   ├── retrieval_node.py   # DI-based retrieval node
│       │   ├── reranking_node.py   # DI-based reranking node
│       │   └── generation_node.py  # DI-based generation node
│       ├── adapters.py         #   HFDatasetRetrieval, LongRAGGeneration
│       └── wtb_integration.py  #   WTB WorkflowProject registration
│
├── self-rag_langgraph/self-rag-wtb/
│   ├── selfrag/
│   │   ├── adapters.py             # Forward + reverse adapters (Self-RAG <-> canonical)
│   │   ├── modular_pipeline.py     # build_selfrag_modular_graph() -- canonical DI graph
│   │   ├── state.py                # QueryState, SelfRAGModularState
│   │   ├── graph_query.py          # Original Self-RAG query pipeline (vLLM-native)
│   │   ├── graph_query_longform.py # Beam search long-form pipeline
│   │   ├── graph_index.py          # Indexing pipeline (chunk -> embed)
│   │   └── nodes/
│   │       ├── modular_query_node.py      # Canonical Query node
│   │       ├── modular_retrieval_node.py  # Canonical Retrieval node
│   │       ├── modular_reranking_node.py  # Canonical Reranking node
│   │       ├── modular_generation_node.py # Canonical Generation node
│   │       ├── prompt_node.py             # Self-RAG prompt builder
│   │       ├── decision_node.py           # Adaptive retrieval decision
│   │       ├── retrieval_node.py          # Contriever + VectorStore retrieval
│   │       ├── generation_node.py         # Evidence generation + scoring
│   │       └── aggregate_node.py          # Best-answer selection
│   ├── wtb_integration.py     # WTB: original + modular project factories
│   ├── test_wtb_integration.py  # 7 WTB integration tests
│   ├── test_modular_swap.py   # 20 cross-project swap tests (protocol + 8 configs)
│   └── real_llm_test.py       # 4 real-LLM tests (query, no-retrieval, longform, WTB)
│
├── storm/storm_langgraph/
│   ├── interfaces.py           # STORM's internal protocols
│   ├── adapters.py             # Forward: Storm -> canonical, Reverse: canonical -> Storm
│   ├── wtb_integration.py      # WTB WorkflowProject registration
│   └── demo/real_components.py # Real LLM-backed STORM components
│
├── tests/
│   ├── test_rag_contracts.py
│   ├── test_storm_adapters.py
│   ├── test_longrag_adapters.py
│   └── test_cross_project_swap.py
│
├── real_swap_demo.py           # Real LLM demo: 4 LongRAG+STORM swap configs
├── real_selfrag_swap_demo.py   # Real LLM demo: 6 Self-RAG+LongRAG+STORM 3-way swap configs
├── wtb_demo.py                 # WTB integration demo with mock components
└── .env                        # LLM_API_KEY, LLM_BASE_URL, DEFAULT_LLM
```

---

## Self-RAG: The Reranking-Generation Entanglement

Self-RAG's scoring mechanism is fundamentally different from standard rerankers.
For each retrieved passage, the LLM generates a candidate answer and scores it
using logprob-based signals:

1. **ISREL** (relevance) -- probability of `[Relevant]` token
2. **ISSUP** (grounding) -- probability of `[Fully supported]` / `[Partially supported]`
3. **ISUSE** (utility) -- expected utility across 5 levels

This means reranking and generation are **fused**: you must generate to score.
The adapters handle this in two ways:

- `SelfRAGGeneration` -- wraps the full pipeline (generate per-passage, score,
pick best). Use as a drop-in `Generation` component in any pipeline.
- `SelfRAGReranking` -- generates internally to compute scores, but only returns
the passage ordering. Pair with any external generator.

### Cross-Project Swap with Self-RAG

```python
from selfrag.modular_pipeline import build_selfrag_modular_graph

# Use LongRAG retrieval + STORM generation inside Self-RAG's pipeline
compiled = build_selfrag_modular_graph(
    retrieval=HFDatasetRetrieval(),          # from LongRAG
    generation=StormGenerationAdapter(sw),    # from STORM
    reranking=SelfRAGReranking(model=llm, rel_tokens=rel_tokens, ...),
)
result = await compiled.ainvoke({"query": "Who wrote Hamlet?"})
```

Or use Self-RAG scoring inside LongRAG's pipeline:

```python
from longRAG_example.longrag_langgraph.main_pipeline import build_graph

graph = build_graph(
    retrieval=SelfRAGRetrieval(doc_store=ds, vector_store=vs, ...),
    generation=SelfRAGGeneration(model=llm, rel_tokens=rel_tokens, ...),
    reranking=IdentityReranking(),  # reranking is inside SelfRAGGeneration
)
```

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

Then plug it in:

```python
graph = build_graph(
    retrieval=PineconeRetrieval("my-index"),
    generation=StormWriterGeneration(llm),
)
```

No other code changes required. The protocol contract guarantees compatibility.