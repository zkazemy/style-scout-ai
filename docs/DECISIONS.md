# Design Decisions

Each entry records what I decided, what else I considered, and why. Newest entries go at the bottom.

Format:

- **Decision:** what I chose
- **Options considered:** the alternatives
- **Reason:** why this one, in my own words
- **Trade-offs / revisit if:** what this costs and when I would reconsider
- **Date:**

---

## D1: Real dataset: Fashion Product Images (Small)

- **Decision:** Use the Kaggle "Fashion Product Images (Small)" dataset, sampled to 3,000 products.
- **Options considered:** H&M Personalized Fashion Recommendations (articles file); a fully synthetic catalog.
- **Reason:** It has category, color, and a usage field that maps well to our `occasion` field, and it is small enough to iterate quickly. H&M is larger and richer in descriptions but heavier to handle. (Add your own reason.)
- **Trade-offs / revisit if:** The dataset has no price, sizes, stock, or real descriptions, so those are synthesized. Results measure the agent on this catalog, not on real retail data. Check the dataset license before publishing anything derived from it. License noted: (fill in)
- **Date:** (update)

## D2: Synthesized fields (price, sizes, stock, description)

- **Decision:** Generate price, sizes, stock, and descriptions with a fixed random seed during catalog build. `in_stock` is derived from sizes.
- **Options considered:** Leaving the fields empty; using only the synthetic catalog generator.
- **Reason:** The agent needs availability and price to be testable. A fixed seed keeps the catalog reproducible, so evaluation results are comparable across runs. Stock lives in one place (sizes), and `in_stock` is derived, so the two cannot disagree.
- **Trade-offs / revisit if:** Synthetic values are not realistic. The evaluation measures grounding and constraint handling, not pricing accuracy. (Add anything you noticed.)
- **Date:** (update)

## D3: Brand derivation rule

- **Decision:** (fill in the rule you chose)
- **Options considered:** (list the options the agent presented, plus any you thought of)
- **Reason:** (why this rule; give an example where it works and one where it fails)
- **Trade-offs / revisit if:** (fill in)
- **Date:** (update)

## D4: Catalog build is deterministic and idempotent

- **Decision:** `build_catalog` uses a fixed seed and can be run repeatedly without creating duplicate rows or changing the result.
- **Options considered:** Appending rows on each run; random generation without a seed.
- **Reason:** Evaluation only means something if the catalog is identical between runs, and rebuilding it should always be safe. (Add what you tested.)
- **Trade-offs / revisit if:** Changing the seed or the enrichment rules changes every downstream result, so catalog changes must be recorded here and the evaluation re-run.
- **Date:** (update)

---

## Decisions still to make (from the spec)

Add an entry when you decide each of these:

1. Vector store (Chroma vs. Qdrant)
2. Embedding model
3. PydanticAI vs. plain function calling
4. How much conversation history to pass to the model
5. RRF weights and `k`
6. Judge model and rubric scales
7. Regression gate thresholds
8. Handling vague queries (ask vs. assume)