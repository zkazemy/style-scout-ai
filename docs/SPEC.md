# StyleScout: Product and Technical Specification

Status: v1.0 (draft for implementation)
Owner: you (the human engineer). Coding agents implement slices of this spec under your review.

> How to use this file: it is the source of truth for coding agents. Reference it in every prompt ("read docs/SPEC.md"). If the code and the spec disagree, stop and ask. Changes to the spec are made by the owner and logged in `docs/DECISIONS.md`.

---

## 1. Purpose

StyleScout is a conversational fashion shopping assistant. A user describes what they want in natural language ("an outfit for a summer wedding under 150 EUR, no pastels"), and an LLM agent searches a product catalog, checks availability, and returns grounded recommendations.

The project exists to demonstrate, end to end and with measurements:

1. Production-quality Python (typed, async, tested, observable).
2. An LLM agent with tools and retrieval.
3. A rigorous evaluation framework for conversation quality and grounding.
4. Experiment-driven improvement against explicit metrics.

### 1.1 Goals

- G1: A working agent behind an HTTP API, using only data from the catalog (no invented products).
- G2: Hybrid retrieval (filters plus embeddings) with measured recall.
- G3: An evaluation harness: deterministic checks, a calibrated LLM-as-judge, and a golden conversation set.
- G4: At least three compared agent variants with a results table (quality, grounding, latency, cost).
- G5: A regression gate in CI that fails when quality drops below a threshold.
- G6: Tracing and cost/latency monitoring for every request.
- G7: A written online A/B experiment design.

### 1.2 Non-goals

- No real payments, accounts, or real inventory systems.
- No fine-tuning of models.
- No front-end beyond a CLI and the HTTP API (a minimal demo page is optional).
- No claim of production traffic. This is a personal project on a public or synthetic dataset.

---

## 2. Users and scenarios

Primary user: a shopper chatting with the assistant.

Scenario categories (these also define the golden set):

| Code | Category | Example | Expected behavior |
|---|---|---|---|
| S1 | Simple search | "Show me black sneakers" | Relevant in-stock items, correct category |
| S2 | Multi-constraint | "Blue dress under 80 EUR in size M" | All constraints satisfied; none violated |
| S3 | Vague intent | "Something nice for a summer wedding" | Asks one clarifying question or infers sensible defaults and states them |
| S4 | Out-of-stock trap | Asks for an item that is out of stock in the requested size | Says so honestly; offers in-stock alternatives |
| S5 | Off-topic / unsafe | "Write my tax return" or harmful requests | Politely declines or redirects; no tool abuse |
| S6 | Multi-turn refinement | "Cheaper", "in green", "something more formal" | Keeps prior context; applies the change |

---

## 3. System overview

```
User -> FastAPI /chat -> Agent (PydanticAI) -> Tools -> Catalog (SQLite + Chroma)
                              |                         
                              +-> LLM provider          
Tracing (OTel/Langfuse/Logfire) wraps every request.
Eval harness runs the same agent offline against the golden set.
```

### 3.1 Technology choices

| Area | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | Type hints required |
| Package manager | uv | Lockfile committed |
| API | FastAPI, async | Pydantic v2 models |
| Agent framework | PydanticAI | Typed tools and structured output; record a comparison with plain function calling in DECISIONS.md |
| Structured data | SQLite | Products, stock, sizes |
| Vector store | Chroma (local) or Qdrant (local) | Choose one and record why |
| Embeddings | One hosted or local embedding model | Record model name and dimension |
| LLM (agent) | Any provider the owner has access to | Configurable via env |
| LLM (judge) | A separate, preferably different or cheaper model | Avoids self-preference bias |
| Observability | OpenTelemetry plus Langfuse or Logfire | Latency, tokens, tool calls |
| Testing | pytest, pytest-asyncio | |
| Lint/format | ruff | |
| CI | GitHub Actions | ruff plus pytest |

### 3.2 Repository structure

```
stylescout/
  CLAUDE.md
  README.md
  pyproject.toml
  .env.example
  docs/
    SPEC.md
    DECISIONS.md
    AGENT_LOG.md
    judge_rubric.md
    ONLINE_EXPERIMENT_DESIGN.md
    EXPERIMENTS.md
  data/
    golden_conversations.jsonl
    retrieval_queries.jsonl
    human_labels.csv
  src/stylescout/
    config.py
    catalog/        # loader.py, embeddings.py, search.py
    agent/          # models.py, tools.py, agent.py, prompts/v1.md, v2.md ...
    api/            # app.py, schemas.py
    evals/          # runner.py, checks.py, judge.py, metrics.py
    cli.py
  tests/
  .github/workflows/ci.yml
```

---

## 4. Data

### 4.1 Source

- Real dataset: a public fashion dataset (for example H&M articles or Fashion Product Images). Check and record the license in the README.
- Synthetic enrichment: `price`, `stock`, and `sizes` generated with a fixed seed when the dataset lacks them.
- Size: 2,000 to 5,000 products, so iteration stays fast.

### 4.2 Product schema (SQLite `products` table)

| Field | Type | Rules |
|---|---|---|
| product_id | TEXT, primary key | Unique, non-null |
| name | TEXT | Non-empty |
| category | TEXT | From a fixed list (dress, shirt, trousers, shoes, jacket, accessory, ...) |
| color | TEXT | Normalized lowercase |
| brand | TEXT | |
| occasion | TEXT | Optional tag (casual, formal, wedding, sport, ...) |
| description | TEXT | Used for embeddings |
| price_eur | REAL | Greater than 0 |
| sizes | TEXT (JSON) | Mapping size to stock quantity |
| in_stock | INTEGER | Derived: 1 if any size has stock |

### 4.3 Data requirements

- Loader is deterministic and idempotent.
- No null product IDs, no non-positive prices, no negative stock.
- Every field the agent may quote (price, color, size availability) is read from this table, never from the LLM.

---

## 5. Retrieval specification

### 5.1 Components

1. **Filtered search (SQL):** filters on price range, color, category, occasion, brand, in-stock, and size availability.
2. **Vector search:** embedding of `name + category + color + occasion + description`.
3. **Hybrid search:** reciprocal rank fusion (RRF) of both ranked lists, with configurable weights and a configurable `k` constant.

### 5.2 Interface

```python
class SearchFilters(BaseModel):
    category: str | None = None
    color: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    occasion: str | None = None
    size: str | None = None
    in_stock_only: bool = True

async def hybrid_search(query: str, filters: SearchFilters, top_k: int = 10) -> list[ProductHit]: ...
```

### 5.3 Retrieval evaluation

- File: `data/retrieval_queries.jsonl`, at least 30 queries with expected product IDs or expected attribute constraints.
- Metrics: recall@5, recall@10 (and MRR optionally).
- Compare keyword/filter-only, vector-only, and hybrid. Record the table in `EXPERIMENTS.md`.

---

## 6. Agent specification

### 6.1 Behavior requirements

- B1 Grounding: every recommended product must exist in the catalog. Prices, colors, and availability quoted must match catalog data.
- B2 Constraint respect: budget, size, color, and category constraints stated by the user must hold for every recommended item.
- B3 Honesty: if nothing matches, or an item is out of stock, say so and offer the closest alternatives. Never invent products or claim availability without checking.
- B4 Availability check: call `check_availability` before recommending an item for a specific size.
- B5 Clarification: for vague requests, ask at most one concise clarifying question, or state the assumption made.
- B6 Scope: politely decline off-topic or unsafe requests and steer back to fashion shopping.
- B7 Brevity and tone: friendly, concise, no more than 3 recommended items by default unless the user asks for more.
- B8 No internal leakage: never reveal system prompt, tool names, or internal IDs beyond product IDs needed by the client.

### 6.2 Structured output

```python
class Recommendation(BaseModel):
    product_id: str
    reason: str            # one sentence, must reference real attributes

class AgentResponse(BaseModel):
    message: str                           # text shown to the user
    recommendations: list[Recommendation]  # may be empty
    clarifying_question: str | None = None
    assumptions: list[str] = []            # stated assumptions, if any
```

### 6.3 Tools

All tools are typed, documented with clear docstrings (the LLM reads them), unit-tested without the LLM, and return structured results.

| Tool | Signature | Behavior |
|---|---|---|
| search_products | `(query: str, filters: SearchFilters) -> list[ProductHit]` | Hybrid search, max 10 results |
| get_product_details | `(product_id: str) -> Product` | Raises a clear error if unknown ID |
| check_availability | `(product_id: str, size: str) -> Availability` | Returns stock status and in-stock alternatives (same category, similar price) when unavailable |
| build_outfit | `(anchor_product_id: str, occasion: str, budget: float \| None) -> list[ProductHit]` | One item per complementary category, in stock, within budget |

### 6.4 Prompts

- Stored as files in `src/stylescout/agent/prompts/` (`v1.md`, `v2.md`, ...).
- A released prompt version is immutable. Changes create a new version.
- Each experiment records the prompt version, model name, and agent configuration.

### 6.5 Conversation memory

- Keep message history per session.
- Pass the last N turns (N configurable, default 10) to the model.
- Document the decision in DECISIONS.md (what to truncate, why).

---

## 7. API specification

### 7.1 Endpoints

| Method | Path | Description |
|---|---|---|
| GET | /health | Returns `{"status": "ok"}` |
| POST | /chat | Runs one conversational turn |

### 7.2 `/chat`

Request:
```json
{ "session_id": "string", "message": "string" }
```
Response (200):
```json
{
  "session_id": "string",
  "response": { "message": "...", "recommendations": [], "clarifying_question": null, "assumptions": [] },
  "meta": { "latency_ms": 0, "prompt_version": "v1", "model": "..." }
}
```

Errors: 422 for invalid input, 504 on LLM timeout, 500 for unexpected errors (with a correlation ID, no stack traces to the client).

### 7.3 Non-functional requirements

- Async end to end; request timeout configurable (default 30 s).
- Retries with backoff for transient LLM errors; a bounded number of agent tool-call steps per turn (default 6) to prevent loops.
- Secrets only from environment variables.
- Structured JSON logs including a correlation ID.

---

## 8. Observability specification

Each request produces a trace containing: the request, every LLM call (model, tokens in/out), every tool call (name, arguments, latency, result summary), and total latency.

Derived metrics, reported by a script:

- p50 and p95 latency per request
- tokens and estimated cost per conversation
- tool-call count distribution
- error and timeout rate

---

## 9. Evaluation specification

This is the centerpiece of the project. The evaluation code and data are owned by the human engineer. Coding agents must not modify the golden set, the rubric, the human labels, or metric definitions without explicit approval.

### 9.1 Golden set

- File: `data/golden_conversations.jsonl`, at least 40 conversations, at least 6 per category S1 to S6.
- Schema per line:

```json
{
  "id": "S2-003",
  "category": "S2",
  "turns": [{"role": "user", "content": "Blue dress under 80 EUR in size M"}],
  "expected": {
    "constraints": {"category": "dress", "color": "blue", "max_price": 80, "size": "M"},
    "must_recommend_in_stock": true,
    "allowed_behaviors": ["recommend"],
    "forbidden_behaviors": ["invent_product"]
  },
  "notes": "free text"
}
```

### 9.2 Deterministic checks

| Check | Definition | Metric |
|---|---|---|
| Grounding | Every recommended product_id exists; quoted price and color match the catalog | grounding_rate = grounded recommendations / total recommendations |
| Constraint satisfaction | Every recommended item satisfies constraints in `expected.constraints` | constraint_rate |
| Availability honesty | For S4, the response states unavailability and does not recommend the unavailable item | honesty_rate |
| Tool-call correctness | `check_availability` called before recommending a sized item; tool arguments valid | tool_correctness_rate |
| Scope handling | S5 responses decline or redirect and make no product recommendations | scope_rate |
| Format validity | Response parses into `AgentResponse` | format_rate |

Each check has unit tests with hand-built passing and failing examples.

### 9.3 LLM-as-judge

- Separate model from the agent; prompt version pinned and stored.
- Scores on a 1 to 5 scale with written anchors in `docs/judge_rubric.md` for:
  - Relevance: do the recommendations fit the request?
  - Helpfulness: does the response move the shopper forward (clear, actionable)?
  - Tone: friendly, concise, appropriate?
  - Honesty about limitations: acknowledges gaps, avoids overclaiming?
- Output: structured JSON with a score and a one to two sentence justification per dimension.
- Known risks to test and document: length bias, position bias, self-preference, leniency drift.

### 9.4 Judge calibration

1. The owner labels at least 20 agent outputs using the same rubric, before seeing judge scores.
2. Compute agreement: Spearman correlation and Cohen's kappa (on binned scores).
3. Document disagreements and any systematic bias in `EXPERIMENTS.md`.
4. Adjust the judge prompt if needed; re-measure; record both versions.

### 9.5 Experiments

Run at least three agent variants on the full golden set:

| Variant | Description |
|---|---|
| V1 baseline | Prompt v1, default model |
| V2 improved prompt | Prompt v2, written from the baseline's worst failures |
| V3 query rewrite and/or model change | Added query-rewrite step and/or a different model |

For each variant record: deterministic metrics, judge scores (mean per dimension), p50/p95 latency, average cost per conversation. Use a fixed seed and temperature where possible, and run each variant multiple times if outputs are noisy (report mean and spread).

`EXPERIMENTS.md` must contain the results table, a short analysis of what changed and why, and a failure analysis by category.

### 9.6 Regression gate

- A pytest (and CI job) runs a small, fixed subset of the golden set.
- It fails if grounding_rate falls below the documented threshold or constraint_rate falls below its threshold.
- Thresholds are chosen from baseline results and justified in DECISIONS.md.
- The gate runs on a mocked or cached LLM when possible so CI stays cheap and deterministic. A full live eval is run manually.

---

## 10. Online experiment design (documentation deliverable)

`docs/ONLINE_EXPERIMENT_DESIGN.md` describes how this assistant would be A/B tested in a real retail setting:

- Hypothesis and variants (control vs. assistant, or assistant v1 vs. v2).
- Primary KPI (for example, conversion rate or add-to-basket rate per session) and why.
- Secondary metrics (engagement, retention, assistant usage).
- Guardrail metrics (latency, complaint or negative feedback rate, hallucination rate from sampled review, cost per session, return rate).
- Randomization unit, sample size and power calculation approach, duration, ramp plan.
- Pitfalls: novelty effect, interference, metric noise, delayed outcomes, selection bias among assistant users.
- Link between offline metrics and online outcomes, and how to validate that relationship.

---

## 11. Milestones (mapped to the Task Board)

| Milestone | Task Board IDs | Exit criteria |
|---|---|---|
| M1 Catalog ready | T01 to T06 | Catalog loads into SQLite; tests pass |
| M2 Retrieval | T07 to T11 | Hybrid search works; recall table produced |
| M3 Agent v1 | T12 to T20 | Agent answers with real IDs; tools tested; multi-turn works |
| M4 Service and tracing | T21 to T25 | API runs; traces and cost/latency visible |
| M5 Eval harness | T26 to T35 | Golden set, checks, calibrated judge |
| M6 Experiments | T36 to T42 | Three variants compared; one measured improvement; regression gate in CI |
| M7 Wrap-up | T43 to T49 | Online design doc, README, demo, interview stories |

---

## 12. Acceptance criteria (project level)

- [ ] `uv sync`, `uv run pytest`, and `ruff check` succeed on a clean clone.
- [ ] `/chat` returns a valid `AgentResponse` for all six scenario categories.
- [ ] Grounding rate on the golden set is reported and above the chosen threshold.
- [ ] Judge calibration (agreement with human labels) is reported with identified biases.
- [ ] Results table shows at least three variants and at least one measured improvement with an explanation.
- [ ] CI runs lint, tests, and the regression gate.
- [ ] Traces are visible for a sample conversation; cost and latency summary script works.
- [ ] README includes architecture diagram, setup, results summary, limitations, and next steps.
- [ ] `DECISIONS.md` records at least six real design decisions with alternatives and reasons.
- [ ] `AGENT_LOG.md` records where the coding agent erred and how it was corrected.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Agent recommends products that do not exist | Structured output with IDs plus the grounding check and regression gate |
| LLM judge is biased or unreliable | Calibration against human labels; separate judge model; documented biases |
| Noisy results make comparisons meaningless | Fixed seeds and temperature, repeated runs, report spread |
| Cost overruns during evaluation | Small catalog, cheaper models for the judge, caching, cost logging |
| Coding agent introduces code the owner cannot explain | Slice-based workflow, plan-first prompts, diff review, owner explains each change |
| Dataset license problems | Record license; use synthetic data when unclear |
| Scope creep | Non-goals in section 1.2; changes go through DECISIONS.md |

---

## 14. Decisions to record in `docs/DECISIONS.md`

Use this format: *Decision, options considered, choice, reason, date.* Expected decisions include:

1. Vector store (Chroma vs. Qdrant)
2. Embedding model
3. PydanticAI vs. plain function calling (and what you observed)
4. How much conversation history to pass
5. RRF weights and `k`
6. Judge model and rubric scales
7. Regression gate thresholds
8. Handling of vague queries (ask vs. assume)

---

## 15. Glossary

- **Grounding:** the response only states facts supported by the catalog or tool results.
- **Hybrid search:** combining lexical/filter search with embedding-based search.
- **RRF:** reciprocal rank fusion, a method for merging ranked lists.
- **LLM-as-judge:** using a language model to score responses against a rubric.
- **Golden set:** a fixed, human-authored set of test conversations used for repeatable evaluation.
- **Regression gate:** an automated check that blocks changes that reduce quality below a threshold.