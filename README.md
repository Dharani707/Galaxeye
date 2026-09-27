# GalaxEye — Offline Satellite Tile Classifier

Take-home submission · Backend Engineer, ML Systems

An offline service that classifies satellite image tiles into one of seven land-use
classes using a locally-run CPU model, stores every prediction, and lets an analyst
query those stored results. No internet connection, no hosted model APIs, no frontend.

| Deliverable | File |
|---|---|
| Part 1 — Design note | [`DESIGN_NOTE.md`](DESIGN_NOTE.md) |
| Part 2 — Code + this README | `server/`, `README.md` |
| Part 3 — Reasoning answers | [`PART3_ANSWERS.md`](PART3_ANSWERS.md) |

**Held-out accuracy: 87.6%** (184/210) · macro F1 0.877 · ~30 ms per tile on CPU.

---

## Contents

1. [What this does](#1-what-this-does)
2. [Requirements](#2-requirements)
3. [Setup](#3-setup)
4. [Starting the server](#4-starting-the-server)
5. [API reference](#5-api-reference)
6. [Testing in Postman](#6-testing-in-postman)
7. [Querying stored results](#7-querying-stored-results)
8. [Where results are stored](#8-where-results-are-stored)
9. [Retraining the model](#9-retraining-the-model-optional)
10. [Project layout](#10-project-layout)
11. [What is and isn't built](#11-what-is-and-isnt-built)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. What this does

You hand the service a satellite tile. It returns the predicted land-use class, how
confident it is, and a full probability breakdown — then records the result so you can
find it later.

Seven classes, fixed:

```
AnnualCrop  Forest  Highway  Industrial  Residential  River  SeaLake
```

The model is a **ResNet18 pre-trained on ImageNet, frozen, with a trained linear
head** on top. Only 3,591 of 11.2M parameters are trained, which is why training takes
under a minute on a laptop CPU. See [`DESIGN_NOTE.md`](DESIGN_NOTE.md) for why that was
chosen over fine-tuning.

The trained weights ship with this repo as `model.pt`, so **you can run the service
immediately without training anything.** Retraining is optional (section 9).

**There is no frontend.** This is a JSON API, as the assignment specified. The
interactive docs at `/docs` are provided by FastAPI automatically and are the only UI.

---

## 2. Requirements

| Requirement | Notes |
|---|---|
| **Python 3.12+** | Check with `python --version` |
| **[uv](https://docs.astral.sh/uv/)** | The only other dependency — see below |
| ~1 GB free disk | CPU PyTorch is a large download |
| No GPU | CPU-only PyTorch is installed deliberately |
| No internet at runtime | Training/prep needs internet once; serving does not |

If you don't have `uv`:

```powershell
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## 3. Setup

From the project root:

```powershell
cd "path\to\GalaxEye project"
```

Install everything:

```powershell
uv sync
```

That's it. `uv` creates a virtual environment, downloads the Python packages, and
installs them. It takes a minute or two on first run because CPU PyTorch is ~200 MB.

You do **not** need to activate anything. Every command below starts with `uv run`,
which finds and uses the environment automatically. If you see
`warning: VIRTUAL_ENV=... does not match`, you're in a manually activated environment —
close that terminal and open a fresh one (see [troubleshooting](#12-troubleshooting)).

**To use a different environment folder name**, set this before running anything:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "my-env-name"
```

> The command in this README assumes uv's default `.venv`. Any environment name works
> identically as long as `UV_PROJECT_ENVIRONMENT` matches it.

Verify the install:

```powershell
uv run python -c "import torch, fastapi; print('torch', torch.__version__); print('ok')"
```

---

## 4. Starting the server

```powershell
uv run uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Wait for these three lines:

```
[INFO ] service_start pid=... device=cpu
[INFO ] model_ready model_version=a564100c14b4 load_seconds=0.16
INFO:  Uvicorn running on http://127.0.0.1:8000
```

`model_ready` is the important one — it means the model loaded successfully.

**Leave this terminal open.** It's the server. Stop it with `Ctrl+C`.

Now open a browser:

```
http://127.0.0.1:8000/docs
```

You should see four endpoints listed with a **Try it out** button each. If that page
loads, the server is working.

### Why `--host 127.0.0.1`

`127.0.0.1` means "this machine only." Use `--host 0.0.0.0` if you need to reach it from
another machine, but there's no auth, so don't expose it to a network you don't control.

---

## 5. API reference

Base URL: `http://127.0.0.1:8000`

| # | Method | Endpoint | Input | Purpose |
|---|---|---|---|---|
| 1 | `GET` | `/health` | none | Is the process alive? |
| 2 | `GET` | `/model/info` | none | Model architecture, params, accuracy |
| 3 | `POST` | `/classify` | **image file** | Classify a tile and store the result |
| 4 | `GET` | `/predictions` | optional filters | Query stored results |

Only `/classify` needs a request body. Everything else is a plain GET.

---

## 6. Testing in Postman

### 6.1 Setup

1. Open Postman → **New** → **HTTP Request**
2. Set the method dropdown to **GET** or **POST**
3. Enter the URL in the address bar
4. Hit **Send**

### 6.2 `GET /health` — no input

- Method: **GET**
- URL: `http://127.0.0.1:8000/health`
- Body: none

```json
{
  "status": "ok",
  "service": "galaxeye-tile-classifier",
  "version": "0.1.0",
  "uptime_seconds": 42.7
}
```

### 6.3 `GET /model/info` — no input

- Method: **GET**
- URL: `http://127.0.0.1:8000/model/info`

```json
{
  "model": {
    "backbone": "resnet18",
    "unfreeze": "none",
    "classes": ["AnnualCrop", "Forest", "Highway", "Industrial", "Residential", "River", "SeaLake"],
    "num_classes": 7,
    "image_size": 224,
    "model_version": "a564100c14b4",
    "total_parameters": 11180103,
    "trainable_parameters": 3591
  },
  "evaluation": {
    "accuracy": 0.87619,
    "macro_f1": 0.877151,
    "mean_confidence": 0.88472,
    "flagged_fraction": 0.095238
  },
  "stored_predictions": 2
}
```

### 6.4 `POST /classify` — the one that needs a file

This is the endpoint that stores a result, so run it before testing section 7.

- Method: **POST**
- URL: `http://127.0.0.1:8000/classify`
- **Body tab → dropdown → `form-data`**

Now build a single row:

| Column | Value |
|---|---|
| **KEY** | `file` |
| **VALUE** | click the cell → choose **File** → select a `.png` |

> **The value must be an actual attached file.** If you type a filename as text,
> Postman sends a text field and the API replies
> `{"type": "missing", "loc": ["body", "file"]}`.
> In the value cell there is a small dropdown reading **Text** — switch it to **File**,
> then browse for the image.

**The key is exactly `file`** (lowercase). The value is the image itself.

Then **Send**. A successful response:

```json
{
  "prediction_id": "0cf9421776d44c36",
  "tile_name": "tile_008.png",
  "tile_sha256": "4898bb1c5460f558909bb607b4adccc7064aa964dbb25872e2b9626c695792cd",
  "label": "Forest",
  "confidence": 0.926586,
  "needs_review": false,
  "review_threshold": 0.6,
  "probabilities": {
    "AnnualCrop": 0.000289,
    "Forest": 0.926586,
    "Highway": 0.051877,
    "Industrial": 0.001701,
    "Residential": 0.005088,
    "River": 0.013612,
    "SeaLake": 0.000847
  },
  "model_version": "a564100c14b4",
  "latency_ms": 59.2,
  "created_at": "2026-09-27T15:26:30.935+00:00"
}
```

Reading it:

| Field | Meaning |
|---|---|
| `label` | Predicted class |
| `confidence` | Softmax probability of that class, 0–1 |
| `probabilities` | Score for **all seven** classes — this is the honest picture |
| `needs_review` | `true` if confidence fell below the threshold |
| `review_threshold` | The cutoff used (default `0.6`) |
| `tile_sha256` | Content hash — the same image always gets the same hash |
| `model_version` | Which model produced this |
| `latency_ms` | Inference time |

**Optional: the review threshold.** Add a second row to force the review flag on:

| KEY | VALUE | type |
|---|---|---|
| `review_threshold` | `0.99` | Text |

The same tile now returns `needs_review: true`, because `0.926 < 0.99`. Useful for
demonstrating the human-review path. Leave it out to use the default `0.6`.

### 6.5 If Postman fights you

Both alternatives hit the identical endpoint:

**cURL** (run from the project folder in PowerShell):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/classify" -F "file=@data/eval_set/tile_008.png"
```

**Swagger UI** — no client setup at all:

```
http://127.0.0.1:8000/docs
```
→ `POST /classify` → **Try it out** → **Select Files** → pick the image → **Execute**

### 6.6 Error responses

| Try this | Response |
|---|---|
| Upload a `.txt` file | `415` unsupported content type |
| Upload a 0-byte file | `400` uploaded file is empty |
| Upload a `.txt` renamed to `.png` | `400` could not decode the upload as an image |
| Upload over 10 MiB | `413` payload too large |
| `review_threshold=1.5` | `422` value out of range |
| Send no file at all | `422` field required |

---

## 7. Querying stored results

Every successful `/classify` appends a row to `predictions.csv`. `GET /predictions`
reads that file back with optional filters — this is the analyst-facing half of the
service.

### 7.1 All results

- Method: **GET**
- URL: `http://127.0.0.1:8000/predictions`
- No body. Filters go **in the URL** after `?`, or in the **Params** tab.

```json
{
  "count": 2,
  "returned": 2,
  "predictions": [
    { "prediction_id": "0cf9421776d44c36", "tile_name": "tile_008.png", "label": "Forest",
      "confidence": 0.926586, "needs_review": false, "...": "..." },
    { "prediction_id": "9cd98da132eb4044", "tile_name": "tile_060.png", "label": "River",
      "confidence": 0.971813, "needs_review": false, "...": "..." }
  ]
}
```

`count` is the total stored; `returned` is how many matched your filters.

### 7.2 Filters

| Parameter | Type | Example | Meaning |
|---|---|---|---|
| `label` | string | `Forest` | Exact class match |
| `min_confidence` | float | `0.9` | Only results at or above this confidence |
| `max_confidence` | float | `0.6` | Only results at or below this confidence |
| `needs_review` | bool | `true` | Only flagged-for-review results |
| `since` | ISO date | `2026-09-01T00:00:00+00:00` | Results created after this time |
| `until` | ISO date | `2026-12-31T00:00:00+00:00` | Results created before this time |
| `limit` | int | `10` | Max rows (1–10000, default 100) |

Any combination, joined with `&`:

```
http://127.0.0.1:8000/predictions?label=Forest&min_confidence=0.9
http://127.0.0.1:8000/predictions?needs_review=true
http://127.0.0.1:8000/predictions?since=2026-09-01T00:00:00+00:00&limit=5
```

**In Postman:** click the **Params** tab and add rows, or paste the whole URL
including the query string. Both work — Postman parses `?a=1&b=2` into the table
automatically.

### 7.3 Real queries worth trying

| Goal | Request |
|---|---|
| Everything | `/predictions` |
| Only forest | `/predictions?label=Forest` |
| The analyst's review queue | `/predictions?needs_review=true` |
| Confident results only | `/predictions?min_confidence=0.95` |
| Borderline results | `/predictions?max_confidence=0.6` |
| Last 10 | `/predictions?limit=10` |
| Wrong class | `/predictions?label=NotAClass` → `422` |

Results are always sorted **newest first**.

---

## 8. Where results are stored

Two files are written at runtime, both gitignored:

| Path | Contents |
|---|---|
| `predictions.csv` | One row per prediction — this is the queryable store |
| `logs/service.log` | JSON log line per request, success or rejection |
| `logs/train_run.jsonl` | Per-epoch training history |

CSV columns:

```
prediction_id,tile_name,tile_sha256,label,confidence,needs_review,model_version,latency_ms,created_at
```

Sample row:

```csv
0cf9421776d44c36,tile_008.png,4898bb1c...,Forest,0.926586,False,a564100c14b4,59.2,2026-09-27T15:26:30.935+00:00
```

Sample log line:

```json
{"ts":"2026-09-27T15:26:30.935+00:00","level":"info","logger":"service","run_id":"a1b2c3d4e5f6","event":"classified","prediction_id":"0cf9421776d44c36","tile_name":"tile_008.png","label":"Forest","confidence":0.926586,"needs_review":false,"model_version":"a564100c14b4","latency_ms":59.2}
```

Both paths can be overridden with environment variables (`GALAXEY_PREDICTIONS_PATH`,
`GALAXEY_LOG_DIR`, `GALAXEY_MODEL_PATH`, `GALAXEY_DATA_DIR`) — see `server/core/config.py`.

To start from a clean slate, delete `predictions.csv`. The header is recreated on the
next request.

---

## 9. Retraining the model (optional)

`model.pt` is committed, so **skip this unless you want to reproduce or change the
model.** Retraining needs the tile dataset in `data/` and an internet connection for
the ImageNet weights (once).

```powershell
uv run python -m server.train
```

Takes about a minute on CPU. Writes `model.pt`, `metrics/eval_report.json`, and
`metrics/confusion_matrix.csv`.

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--unfreeze` | `none` | `none` = linear probe (default); `layer4` or `all` to fine-tune |
| `--backbone` | `resnet18` | Any of resnet18/34/50, efficientnet_b0, mobilenet_v3_small |
| `--head-epochs` | `300` | Max epochs for the linear head |
| `--finetune-epochs` | `30` | Max epochs when unfreezing |
| `--val-fraction` | `0.20` | Hold-out split |
| `--seed` | `42` | Reproducibility |
| `--no-eval` | off | Skip the held-out evaluation |
| `--device` | `cpu` | `cuda` if you have a GPU |

Fine-tuning the last block, if you want to try it:

```powershell
uv run python -m server.train --unfreeze layer4
```

**Restart the server after retraining** — the model is loaded once at startup.

---

## 10. Project layout

```
.
├── DESIGN_NOTE.md          Part 1 — approach, trade-offs, assumptions
├── PART3_ANSWERS.md        Part 3 — the four reasoning questions
├── README.md               This file
├── pyproject.toml          Dependencies
├── model.pt                Trained weights (committed, so no training needed)
│
├── server/
│   ├── main.py             FastAPI app: the four endpoints
│   ├── train.py            Training + evaluation entrypoint
│   └── core/
│       ├── config.py       Classes, paths, thresholds, CSV schema
│       ├── model.py        Architecture, training, metrics, inference
│       ├── dataset.py      Tile loading, splitting, embeddings
│       ├── storage.py      CSV append and query
│       └── logging_utils.py  JSONL logging
│
├── metrics/
│   ├── eval_report.json    Accuracy, per-class metrics, confusions
│   └── confusion_matrix.csv
│
├── data/                   Tile dataset (gitignored)
├── logs/                   Runtime logs (gitignored)
└── predictions.csv         Stored results (gitignored, created on first request)
```

---

## 11. What is and isn't built

**Built and working:**

- `POST /classify` — the core path: upload → validate → classify → store → return
- `GET /predictions` — filtered queries over stored results
- `GET /health`, `GET /model/info` — operational endpoints
- Input validation: size, content type, decodability, threshold range
- Content hashing, model versioning, structured JSON logging
- Offline CPU training pipeline with held-out evaluation

**Deliberately not built** — the assignment said *"Stub or skip the rest, and say so"*:

| Not built | Why |
|---|---|
| **Frontend / UI** | Not requested. JSON API only, plus FastAPI's `/docs` |
| Batch ingest | One tile per request is the core path; batching would need a queue |
| Authentication | No analyst identity modelled — every caller is anonymous |
| SQLite / Postgres | CSV is enough at this scale; `storage.py` isolates the swap. **This is the weakest part of the design** — see [`DESIGN_NOTE.md`](DESIGN_NOTE.md) |
| Geospatial queries | No coordinates or scene metadata in the provided data |
| Automated monitoring | Discussed in Part 3 answer 2, not implemented |
| Containerisation | `uv sync` + one command is enough for a single box |

---

## 12. Troubleshooting

**`warning: VIRTUAL_ENV=... does not match the project environment path .venv`**
You manually activated an environment. Don't — `uv run` handles it. Close the terminal,
open a fresh one, `cd` to the project, and use `uv run` directly.

**`Creating virtual environment at: .venv` when you expected a different folder**
`UV_PROJECT_ENVIRONMENT` isn't set for that shell. Either accept `.venv` (it works
fine) or set it:
```powershell
$env:UV_PROJECT_ENVIRONMENT = "my-env-name"
```

**uv isn't recognised in PowerShell**
Close and reopen the terminal after installing, or add uv's directory to `PATH`.

**`Address already in use` / port 8000 occupied**
```powershell
taskkill /F /PID (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
```
Or start on a different port: `--port 8001`.

**`model artifact not found at .../model.pt`**
Train it (section 9), or check `model.pt` was included when you copied the repo.

**`413` on a normal-looking image**
The limit is 10 MiB. Check the file size.

**Postman says `{"type": "missing", "loc": ["body", "file"]}`**
The value cell holds text instead of an attached file. Change its type to **File**,
or use `/docs` instead — see section 6.5.

**Stuck: is the server actually running?**
```powershell
curl.exe http://127.0.0.1:8000/health
```
If that returns JSON, the server is fine and the problem is the request.

---

## License / notes

Built for the GalaxEye take-home assignment. The tile data is not redistributed here
(`data/` is gitignored). Pre-trained ImageNet weights come from `torchvision`.
