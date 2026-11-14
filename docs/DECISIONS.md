# DECISIONS additions (D5 to D10)

Append these to `docs/DECISIONS.md`. Edit every entry into your own words, and fill in each "(fill in)" from what you actually saw. Do not keep a claim you cannot explain.

---

## D5: Vector store: Chroma

- **Decision:** Use Chroma as the local vector store for the product catalog.
- **Options considered:** Qdrant, pgvector, FAISS, and hosted vector databases such as Pinecone.
- **Reason:** Chroma is easy to set up, integrates well with Python, and supports local persistence without a separate database service. With only 3,000 products it is sufficient for our retrieval experiments and keeps the architecture simple.
- **Trade-offs / revisit if:** Suitable for this project's scale. A production system with millions of products, high concurrency, or distributed deployment may need Qdrant or another scalable vector database. The index is rebuilt from scratch on every build (the old collection is deleted and recreated), which is simple and safe at this size but re-embeds everything each time.
- **Date:** 2026-11-14

## D6: Embedding model: all-MiniLM-L6-v2

- **Decision:** Use the local `all-MiniLM-L6-v2` model for product documents and user queries.
- **Options considered:** Other Sentence Transformers models, larger embedding models, hosted embedding APIs.
- **Reason:** Lightweight, runs locally, 384-dimensional embeddings, good speed/quality balance for a small catalog. No API cost, and local embeddings make experiments more reproducible. The same model is used for documents and queries.
- **Trade-offs / revisit if:** The model is trained mostly on English, so German queries (likely for a Berlin retailer) will probably do worse. It may also struggle with fashion-specific language and subtle differences. Revisit if recall is too low or if multilingual or domain-specific retrieval is needed. Changing the model requires re-embedding the whole catalog.
- **Date:** 2026-11-14

## D7: Keyword search: SQLite FTS5 with OR matching and bm25 ranking

- **Decision:** Keyword search uses an SQLite FTS5 index over the products table, ranked with bm25. Query words are combined with OR, so "blue dress" finds products with either word and ranks products with both higher. A short stopword list (a, an, the, and, or, not, in, for, of, with, is, to, at) is removed before searching.
- **Options considered:** (fill in: FTS5 vs. LIKE queries, and what the agent said about trade-offs. Also note whether AND matching was considered. FTS5 uses AND by default.)
- **Reason:** (fill in your own reasons, for example ranking quality, no new dependency, and fit with the filters.)
- **What I observed:** For "black sneakers", OR matching returned black sunglasses and earrings in the top 5, because they match "black". Vector search returned only black shoes for the same query. Neither method is enough alone, which is the reason for the hybrid search in T09.
- **Trade-offs / revisit if:** OR matching is noisy for multi-word queries. The stopword list is short and English-only (it lacks words like "something"). Revisit after the retrieval evaluation (T10).
- **Date:** (update)

## D8: User input is quoted as literals, never parsed as FTS syntax

- **Decision:** Each query token is wrapped in double quotes before it reaches FTS5, and tokens with no letter or digit are dropped. Characters like `*`, `(`, `:` and words like `NEAR` are treated as plain text.
- **Options considered:** Stripping special characters from tokens (the agent's first plan).
- **Reason:** Stripping is fragile, because FTS5 has operators (such as NEAR) that stripping does not clearly cover. Quoting treats the input as literal text and removes the whole class of syntax errors. Parameterized SQL is used throughout for the SQL injection side.
- **Evidence:** Tests cover quotes, wildcards, parentheses, column prefixes, bare AND/OR/NOT, a trailing operator, NEAR, `NEAR(`, punctuation-only queries, and an injection-style string. Each pairs the dangerous text with a real word and checks that the real word still matches.
- **Date:** (update)

## D9: Keyword index covers all six spec fields; None or empty query means filter-only

- **Decision:** The FTS index covers name, brand, color, category, occasion, and description. A None, empty, whitespace-only, or stopwords-only query skips keyword search and returns products that match the filters only, ordered by product_id, with a score of 0.0 meaning "not ranked". A missing FTS table raises `CatalogIndexMissingError`, whose message says to run `uv run stylescout build-catalog`. Other database errors propagate unchanged.
- **Options considered:** For the index fields, leaving category and occasion out and relying on the description text, which already contained them (the first version of the index did this). For errors, logging a warning and returning no results.
- **Reason:** Relying on the description text is a hidden dependency: if the description template changed, keyword search would get worse and nothing would warn us. Failing loudly on a missing index prevents the assistant from reporting "no products found" when the real problem is a missing index.
- **What I observed:** Adding the fields did not change today's results (the words were already in the descriptions). It protects against future change.
- **Trade-offs / revisit if:** A stopwords-only query with no filters returns the first products by id, which is arbitrary. Acceptable for now.
- **Date:** (update)

## D10: Known data limitations

- **Decision:** Accept these limitations for now and keep them visible.
- **Limitations observed:**
  - The category "dress" mixes in kids' skirts, kurtis, kaftans, and nightdresses. The dataset has gender and age columns that our schema does not carry, so kids' items cannot be filtered out yet.
  - Prices, sizes, stock, and descriptions are synthetic. Two products with the same name can have prices of 150.45 and 39.41 EUR, so prices are not realistic and must not be read as real.
- **Why it matters:** The evaluation checks constraints (category, price, size) against this data. Results measure the agent on this catalog, not on real retail data.
- **Revisit if:** Constraint checks in the evaluation produce false failures because of category granularity. Then add age and gender, or a finer category mapping.
- **Date:** (update)