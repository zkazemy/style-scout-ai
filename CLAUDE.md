# StyleScout

Conversational fashion assistant. Python 3.12, FastAPI (async), PydanticAI,
SQLite + Chroma, hybrid search (filters + embeddings, RRF fusion).

Read `docs/SPEC.md` before any task. It is the source of truth. If code and spec
disagree, stop and ask me.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint and format: `uv run ruff check . && uv run ruff format .`
- Chat in terminal: `uv run stylescout chat`
- Run API: `uv run uvicorn stylescout.api.app:app --reload`
- Eval: `uv run stylescout eval`

## Working rules
- Always propose a short plan first and wait for my approval before writing code.
- Type hints everywhere. Pydantic models for all tool inputs and outputs.
- Every function you add gets a pytest test. Tests and ruff must pass before you say you are done.
- Keep functions small. Add a short docstring explaining non-obvious design choices.
- Prompts live in `src/stylescout/agent/prompts/` as versioned files (v1.md, v2.md).
  Never edit a released version; create a new one.
- Read secrets from `.env` only. Never commit keys. Never print them.
- No new dependencies without asking me.
- Prices, colors, and availability must come from the catalog tables, never from the LLM.

## Do not change without asking me
- `data/golden_conversations.jsonl`
- `data/human_labels.csv`
- `docs/judge_rubric.md`
- Metric definitions in `src/stylescout/evals/metrics.py`

The evaluation must stay independent of the agent being evaluated.

## When you finish a task
1. Run tests and lint, and show me the output.
2. Summarize what changed and which files you touched.
3. List anything you were unsure about or assumed.
4. If you hit a wrong API, a failing approach, or a spec conflict, tell me so I can log it in `docs/AGENT_LOG.md`.

## Architecture
`catalog/` (loading, embeddings, search) -> `agent/` (models, tools, prompts) -> `api/` -> `evals/`

Design decisions are recorded in `docs/DECISIONS.md`.