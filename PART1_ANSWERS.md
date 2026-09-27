# Design Note — Offline Satellite Tile Classifier

*Part 1 · Backend Engineer, ML Systems · GalaxEye*

## 1. How I'd build it

### Components

- **API layer** — HTTP contract, input validation, error mapping
- **Inference layer** — preprocessing, forward pass, softmax, confidence
- **Storage layer** — appending predictions, running filtered queries
- **Observability layer** — one JSON log line per request
- **Training layer** — offline; emits the model artifact and its metrics

### How a tile flows through

- Tile arrives at `POST /classify` as a multipart image
- **Cheap guards first** — 10 MiB cap, then content type. Reject before spending any CPU
- Read the bytes; reject if empty
- Hash the content with SHA-256 — computed once, reused everywhere downstream
- Decode → convert to RGB → resize 224×224 → ImageNet normalise
- Forward pass under `no_grad`, in a **threadpool** so requests don't serialise on the GIL
- Softmax gives `(label, confidence)`; flag anything below the threshold for review
- Append to the results store and to the log
- Return the label, the confidence, and the **full seven-class probability vector**

The model loads once at startup, not per request.

### Storage — the decision that took the most thought

- **JSONL, one record per prediction** — append-only, never drifts schema. But every query is a full file scan, so filtering degrades linearly as history grows. *A log format, not a store.*
- **SQLite** — indexed filters, a real schema, safe under concurrent readers, one file that copies cleanly off an air-gapped box, no server process. Costs a dependency and a migration story.
- **Postgres** — correct at real scale, wrong here. Needs a running server, so the submission stops being one command to run, and it pushes the offline story onto ops.

**I'd ship SQLite.** The submitted slice uses CSV, because CSV is the smallest thing that proves the contract and still lets an analyst filter today. Storage sits behind one module, and the filters I expose — label, confidence band, review flag, time window, limit — are exactly what SQLite indexes, so moving across is additive rather than a redesign.

## 2. The decisions and trade-offs

### Predictions the model isn't confident about

- **Always return the top class** — simplest, and quietly launders a guess into a fact
- **Abstain below a threshold** — honest, but hands the decision to the client and throws away the ranking a human reviewer needs
- **Return the guess and flag it** — ✅ chosen

Why the third one:

- Every response carries the label, confidence, the full probability vector, and a `needs_review` flag
- The review queue is then a **query** (`needs_review=true`), not a parallel code path
- Nothing can be flagged without being stored, and nothing gets stored unflagged

**The threshold is a per-request field, not a constant.**

- The correct cutoff is class-dependent — my worst confusions are `SeaLake→Forest` and `AnnualCrop→River`
- Any single global number will be a compromise, so I expose it per request rather than baking it in
- Each analyst can tune it without a redeploy
- Per-class thresholds are the better answer; because I store the full probability vector, they're additive later

### What to store

One row per prediction: `prediction_id`, `tile_name`, `tile_sha256`, `label`, `confidence`, `needs_review`, `model_version`, `latency_ms`, `created_at`.

- **`tile_sha256` is the column that matters most** — filenames aren't stable but content is, so hashing makes ingestion idempotent-by-detection and makes *"why was this tile classified twice?"* a question with an answer
- **`model_version` is a hash of the weights**, not a build tag — when a retrained model disagrees with stored data, this is what makes the cause visible
- **`latency_ms` and `created_at`** turn *"it's gotten slow"* from a feeling into a query
- I don't store tiles or embeddings in the results store — the hash is enough to join back to whatever ingest retained

### What "querying" should mean

**Filter-and-list** — not aggregation, joins, or geospatial selection.

The questions an analyst actually asks, and what they map to:

- *"Show me this class"* → `label`
- *"Show me what the model wasn't sure about"* → `needs_review`
- *"Show me what arrived this week"* → time window

Each is one query and each is indexed. I'd hold off on aggregates until someone asks for one, because every aggregate is another thing that has to stay correct.

## 3. Assumptions, and questions I'd ask

### Assumptions

- Tiles arrive one per request, not as mosaics
- The seven classes are fixed for this exercise
- Input is already georeferenced, with a 64×64 crop as the atomic unit
- Single writer, single process
- **No analyst identity or permissions** — every caller is anonymous and every stored row is mutable. A real gap, not an oversight.

### Questions I'd ask

1. **Must historical tiles be re-classified when the model changes?** If so, `model_version` becomes a primary key and backfill matters — it changes the storage answer.
2. **What's the real ingest volume?** Tens of thousands of tiles settles the storage decision outright; a few thousand doesn't. This changes the design more than anything else, and I can't infer it from the data I was given.
3. **What does a wrong answer actually cost?** If a missed industrial site carries regulatory consequence, 0.60 is far too permissive and the interface should lean toward abstention. If it's a triage hint, 0.60 is fine. A business decision wearing a constant's clothing.
4. **Do tiles come from distinct scenes?** Tiles within a scene are near-duplicates, so a random split would leak scene identity — making my accuracy figure an upper bound, not a field estimate.
5. **What geospatial context is available at all?** `SeaLake` vs `River` is my worst confusion and depends heavily on location and season. A 64×64 crop with no coordinates, no date, and no neighbouring context is a genuinely hard version of this task — more input data would help more than a bigger model.
6. **What latency and batch profile do you need?** Interactive single tiles and nightly bulk runs justify quite different architectures.
