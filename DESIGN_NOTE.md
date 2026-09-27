# Design Note — Offline Satellite Tile Classifier

*Part 1 · Backend Engineer, ML Systems · GalaxEye*

---

## 1. What I'm building

A single-purpose service that accepts a satellite tile, classifies it into one of seven
land-use classes, stores the result, and lets an analyst query stored results. It runs
offline on CPU with no network calls and no hosted model APIs.

The submitted slice is deliberately thin. It implements the **core path only** —
ingest → classify → store → query — plus the operational endpoints needed to run it.
Batch ingest, a real analyst UI, model retraining pipelines, and multi-tenant auth are
explicitly **not** built. I've said where the seams are instead of stubbing fake
versions of them.

## 2. Components

| Component | Responsibility | Lives in |
|---|---|---|
| **API layer** | HTTP contract, input validation, error mapping, response shaping | `server/main.py` |
| **Inference** | preprocessing, forward pass, softmax, confidence, review flag | `server/core/model.py` |
| **Storage** | append predictions, query by filter, count | `server/core/storage.py` |
| **Observability** | structured JSONL for every request outcome | `server/core/logging_utils.py` |
| **Config** | one place for paths, classes, thresholds, limits | `server/core/config.py` |
| **Training** | offline embedding extraction + linear head fit | `server/train.py`, `server/core/dataset.py` |

The model loads once at process start (lifespan hook), not per request. Cold start is
~0.15 s; warm inference is ~20–30 ms per tile on CPU.

## 3. How a tile flows through

```
POST /classify (multipart)
  │
  ├─ 413  size > 10 MiB            ── cheap guard, before any decode
  ├─ 415  content-type not image
  ├─ read bytes ── 400 if empty
  ├─ sha256(bytes)                ── content address, computed once
  ├─ decode + RGB convert          ── 400 if undecodable
  ├─ resize 224×224, ImageNet norm
  ├─ forward pass (torch.no_grad, threadpool)
  ├─ softmax → (label, confidence)
  ├─ needs_review = confidence < review_threshold
  ├─ append row to predictions.csv   ── fsync'd, header created on demand
  ├─ append one JSON line to logs/service.log
  └─ 200 with label, confidence, full probability vector, id, hash, latency
```

Two choices worth calling out:

- **Inference runs in a threadpool.** Torch releases the GIL during the forward pass,
  so an `async def` endpoint that called it directly would serialise requests anyway.
  The endpoint stays async; the blocking work goes to a worker thread.
- **Everything is logged as JSON.** Offline, nobody is watching a terminal. A JSONL
  file is greppable, append-only, needs no daemon, and can be shipped later if a log
  shipper ever exists. Human-readable strings would have to be re-parsed later.

## 4. Storage: the real decision

An analyst needs to query stored results, so results have to persist. I considered
three options.

**(a) JSONL per prediction** — simplest possible, append-only, no schema drift.
Rejected as the primary store: every query is a full file scan, so "show me all
low-confidence SeaLake tiles from last week" gets linearly slower forever. Fine for
logs, wrong for a queryable store.

**(b) SQLite** — real indexed queries, a proper schema, survives concurrent readers,
single file that copies cleanly off an air-gapped box, zero-ops. Costs a dependency
and a migration story.

**(c) Postgres** — the right answer at real scale, but wrong for this slice. It wants
a server process, so the thing I submit stops being "one command to run". It also
pushes the offline story onto ops.

**I chose (b) as the direction and shipped (a)-shaped CSV for the slice.** That is the
honest version of the answer: CSV is not my recommendation, it is the smallest thing
that demonstrates the contract and lets an analyst filter results today, with the
storage interface already isolated behind `storage.py` so swapping in SQLite touches
one module and no route code. The query surface (label, confidence band, review flag,
time window, limit) is deliberately the set of filters SQLite would index directly, so
the migration is additive rather than a redesign.

**Concurrency caveat, stated plainly:** CSV append is not safe against multiple
writer processes. This is fine for a single uvicorn worker, which is the deployment
here, and it is the first thing that breaks if someone runs `--workers 4`. I would not
ship it that way.

## 5. Low-confidence predictions

The question is what to do when the model is unsure. Options:

1. **Always return the top class.** Simple, and quietly launders a guess into a fact.
2. **Abstain** (return no label when below threshold). Honest, but pushes the decision
   onto the client and loses the ranking that a human reviewer needs.
3. **Return the guess *and* flag it.** Chosen.

Option 3 means every response carries the label, the confidence, the full seven-class
probability vector, and `needs_review`. The analyst's queue is a query
(`needs_review=true`), not a separate code path, so nothing can be flagged without
being stored and nothing is stored without a flag.

The threshold defaults to `0.60` and is a **per-request form field**, not a constant.
That is deliberate: different analysts trust the model at different levels, and the
right threshold depends on the class being asked about — the confusion matrix says
`SeaLake → Forest` and `AnnualCrop → River` are where this model is weakest. A single
global number is a compromise, and I would rather it be visible and overridable than
baked in. I am aware per-class thresholds are better and have not built them; the
per-tile probability vector is stored, so they can be added without a schema change.

Flagging rate at the default threshold is 9.5% — small enough to be reviewable.

## 6. What gets stored, and why

One row per prediction: `prediction_id`, `tile_name`, `tile_sha256`, `label`,
`confidence`, `needs_review`, `model_version`, `latency_ms`, `created_at`.

- `tile_sha256` is the important one. Filenames are not stable; content is. It makes
  ingestion idempotent-by-detection and makes "why is this tile classified twice"
  answerable.
- `model_version` is a hash of the weights, not a build tag. When a re-trained model
  starts disagreeing with stored data, this is the column that makes the cause
  visible instead of mysterious.
- `latency_ms` and `created_at` turn "the service got slow" from a feeling into a query.
- I do **not** store raw tiles or embeddings in the results store. The hash is enough
  to join back to whatever the ingest pipeline kept. If the tiles are not retained
  anywhere, that is a gap — see assumptions.

## 7. Assumptions

- Tiles arrive as individual image files, one request each, not as large mosaics.
- The seven EuroSAT-style classes are fixed for this exercise.
- The input is already georeferenced imagery; a 64×64 crop is the atomic unit.
- Analyst identity, permissions, and audit are **not** modelled — every caller is
  anonymous and every stored row is mutable. That is a real gap, not an oversight.
- Single writer, single process (see the CSV caveat above).
- "Query" means filter-and-list, not aggregation, joins, or geospatial selection.

## 8. Questions I would ask

1. **Do we need to re-classify historical tiles when the model changes?** If yes,
   `model_version` needs to become a first-class key and a migration/backfill path
   matters. It changes the storage answer.
2. **What is the real ingest volume?** Tens of thousands of tiles changes the storage
   decision outright; a few thousand does not. This is the question that most changes
   the design and I cannot answer it from the data given.
3. **What is the cost of a wrong answer?** If a missed industrial site has a
   regulatory consequence, a 0.60 threshold is far too permissive and the interface
   should lean toward abstention (option 2). If it is a triage hint, 0.60 is fine.
   The threshold is a business decision masquerading as a constant.
4. **Is the imagery per-scene, and do tiles from one scene share conditions?** Tiles
   from a single scene are correlated, so a random train/test split leaks scene
   identity and inflates my reported 87.6%. That number is optimistic and I would not
   defend it as a field estimate.
5. **What geospatial context is available?** Land-use classification of `SeaLake` vs
   `River` — my worst confusion — depends heavily on scene context and season. A
   single 64×64 tile with no location, date, or neighbouring context is a genuinely
   hard version of this task, and more input data would help more than a bigger model.
6. **What is the expected latency and batch profile?** Interactive single tiles versus
   nightly bulk would justify different architectures (batching, a queue, async workers)
   and possibly a different serving stack.

## 9. What would break first

The CSV store, under concurrent writers — then the single-tile input assumption, if
real ingest arrives as multi-tile scenes. Both are called out above rather than hidden.
