# Part 3 — Problem-Solving Answers

*Short answers. The assignment asks for reasoning rather than volume.*

## 1. The classifier is wrong 30% of the time — what do I do, and how do I decide whether it's "good enough"?

**Thirty percent isn't a verdict, it's a description.** On its own it doesn't tell me whether the model is bad or my labels are. So the order of work is:

**First, check the label noise.**

- Sample ~50 tiles and have a human label them independently
- If annotators disagree with *each other* at roughly the rate they disagree with the model, then a large slice of that 30% is label noise
- No amount of retraining moves that number — the ceiling is annotator agreement, not the network
- This is the unglamorous check most projects skip, and it's the one that tells you whether you have a data problem or a modelling problem

**Second, look at the shape of the error, not the amount.**

- On my held-out set accuracy is 87.6%, but mistakes are **concentrated**
- `SeaLake→Forest` and `AnnualCrop→River` account for most of the loss
- `Industrial` and `Highway` are nearly perfect
- That says the model isn't broadly weak — it has two specific confusions between classes that are genuinely ambiguous in a 64×64 crop with no season or location context
- Which is a data problem with a known shape, not a model that needs to be generally better

**Third, establish what I'd be compared against.** The bar isn't perfection, it's beating whatever the analyst does today.

- If the status quo is a human eyeballing every tile, a model that reliably builds a shortlist genuinely helps — the human still decides, the model just orders the queue
- That's a much easier thing to be 70% right about
- If the status quo is "auto-populate a report that gets signed," no level of accuracy saves you, and the *interface* has to change instead of the model

**So the metric I'd optimise is precision on the review queue, not accuracy:**

- Flag the lowest-confidence 10% of tiles
- Ask: what fraction of those flags are truly the class the analyst cares about?
- Ship once that clears a threshold we agree on together
- Tune to move that number, not to move accuracy

**In order, then:** measure label noise → break error down per class and per confusion → agree the operating point with the people who'd use the output → ship predict-and-flag rather than predict-and-assert → log the hash and full probability vector so the 30% figure is always arguable with a query rather than an opinion → spend accuracy effort last, and only on the two confusions actually costing us.

**The honest framing:** "good enough" is never a property of the model alone. The same 70%-accurate model is useless behind a "Commit" button and valuable behind a "here are fifty tiles worth looking at" button. I build the second one deliberately.

## 2. This runs offline with nobody watching — a month later, how would I know it's still working?

**Starting position: I wouldn't, unless I built the check before I needed it.** A service nobody is watching degrades silently and finds out at the worst possible moment. So monitoring has to be a property of the system, not a dashboard someone remembers to open.

**The key distinction: liveness isn't correctness.**

- A 200 from `/health` proves a process is alive
- It says nothing about whether it's still loading the right weights, normalising pixels correctly, or persisting results
- Most failure modes that actually matter are invisible to a health check
- I'd rather have no health endpoint than believe it covers the risk

**What I'd add, in priority order:**

**A scheduled self-test against held-out ground truth — the single most valuable thing.**

- Keep a labelled gold set that's never used for training
- A small cron job on the box runs it through the full serving path on a schedule
- Writes a JSON status file; exits non-zero on regression against a recorded baseline
- No cloud to alert from, so a file is the right output
- Turns *"is it still working"* into a number anyone can check, and reuses the harness I'd want anyway
- It's also the only check that catches the failure I most expect: someone updates a dependency, the ImageNet normalisation constants change, predictions shift quietly, and every green check stays green

**Confidence distribution monitoring** — the cheapest high-signal metric available.

- Mean confidence and flag rate over a rolling window are single queries over data I already log
- A collapse from 0.88 to 0.5 would be visible long before anyone complained about accuracy

**Input drift tracking.**

- Hash the inputs so I can diff the set of tiles seen against the training distribution
- Turns *"have we started getting different imagery?"* into a set difference rather than a hunch

**Artifact integrity.**

- `model_version` is recorded on every prediction
- Comparing the version in the logs against a fresh hash of the file on disk catches a corrupted, truncated, or accidentally swapped model
- Because every stored row carries its version, it turns *"this data came from a different model"* into a filter rather than a mystery

**Boring operational checks.** Disk space on the results volume, error rate, request rate, p95 latency, and confirmation that rows are still being written. A full disk is the classic silent killer — requests keep succeeding while nothing is persisted.

**What I would deliberately not build:** a metrics stack. A time-series database, a dashboard, and an alerting pipeline are the wrong shape for a single offline box. A nightly job writing one status file, plus a log anyone can grep, covers most of the risk at a fraction of the complexity — and if nobody's checking that file, a fancier alert wouldn't have helped either.

## 3. Tiles are arriving fine but stored results look wrong — how do I find the cause?

**My instinct would be to suspect the model first, and I'd resist it.** The model is the slowest thing to check and by far the least likely culprit. I'd work the pipeline from input to output, cheapest and most likely first, making each step falsifiable before moving on.

**Step 0 — the check that splits the problem in half.** Take a tile whose true label I know and compare three things:

- The true label
- The **live API response**
- The **stored row**

Then:

- Response wrong, stored row agrees → the bug is in inference or the model. Stop looking elsewhere
- Response right, stored row wrong → the bug is in the write path, and the model is innocent

Half of all "results look wrong" reports die right there, and it costs two minutes.

**Step 1 — establish the shape of the damage.** One query, and it's the biggest clue available:

- Filter by time: is this new, or has it always been there?
- A sharp break at a specific timestamp points hard at a deploy, config change, or upstream data change — and gives me an exact moment to diff against
- Gradual drift points at drift
- Wrong on every tile of one class and not others points at the class mapping

**Step 2 — verify the inputs are what I think they are.**

- Decode the tiles, check dimensions and channel count
- Compare their statistics to the training distribution
- An upstream pipeline may have started sending grayscale tiles, a different resolution, or a different sensor entirely
- Cheap, and it sits upstream of everything else

**Step 3 — check that serving-time preprocessing still matches training.** This is the most common cause of a model that "suddenly" got worse, and it's silent — no exception, no log line, just quietly wrong numbers.

- Candidates: different resize interpolation, different normalisation constants, RGB/BGR confusion, different input size
- Verify by pushing training tiles through the **exact serving code path** and checking predictions match what training produced
- If they don't, I've found it without touching the weights at all

**Step 4 — check the artifact itself.**

- Compare `model_version` in the logs against a fresh hash of the file on disk
- A truncated download or a stale `model.pt` all present identically as correct code with wrong weights

**Step 5 — look for permuted labels.** If predictions are *right* but the labels are scrambled in a consistent way, the cause is almost always the class list drifting out of order relative to the head that was fitted. That looks like catastrophic model failure and is a one-line fix.

**Step 6 — read the raw CSV directly** rather than through my own query code, because I don't trust the code I'm debugging to tell me the truth about itself.

**Step 7 — only now, the model.** Re-run the full evaluation and consider retraining.

**The principle underneath: cheap-first, high-prior-first.** Every step either confirms or eliminates an entire category of cause. And I never change the model until everything upstream is eliminated — because retraining as a first move destroys the evidence, and after a retrain the original bug is unreproducible and you learn nothing.

## 4. What's the weakest part of my design, and what would break it first?

**The storage layer, without question.** Everything else is either correct or fails loudly; the CSV store is neither. It's an append-and-scan file with no transactions, no locking, no schema enforcement, no indexes, and no concurrency safety — a fine placeholder and a genuinely bad database.

**What breaks first — the most likely failure: concurrent writers.**

- CSV append isn't atomic across processes
- It works perfectly for the single uvicorn worker I run
- The first person who deploys with `--workers 4` — the obvious next thing anyone would do — gets interleaved, corrupted rows
- Nothing warns you. The service keeps returning 200, and the damage turns up weeks later when a row has a label sitting in the confidence column

**The nastier failure: a truncated final line.**

- If the process dies mid-append, the last row is half-written and isn't valid CSV
- My own reader tolerates a malformed line, but pandas, a spreadsheet, or the next person to write a query will fail to parse the file and lose access to *every* row rather than just the broken one
- All the good data, made unreadable by a single bad line — data loss wearing the costume of a read-only query problem

**Why this is the weakest *design* choice, not just a weak implementation:** the rest of the system quietly assumes the store is trustworthy.

- The review queue is a query against this file
- The idempotency story is "dedupe by content hash" — which only means something if every row is intact
- If a write is lost, a tile silently vanishes from the analyst's queue
- That's precisely the failure mode where a review flag gets ignored, because nobody notices the tile that never showed up
- So the storage weaknesses propagate into a *silent correctness problem in the workflow* rather than an obvious crash

**What I'd fix, in order:**

1. **Durable writes first** — atomic write-and-rename, or move to SQLite, which gets real transactions and multi-process safety together
2. **Integrity checking** — so a truncated tail is detectable and skippable rather than fatal
3. **Proper indexing** — so query cost stops growing with history

**Close second: my train/test split.** It's random rather than grouped by scene. Tiles from one satellite scene are near-duplicates, so a random split leaks scene identity and my reported accuracy is optimistic. A grouped split would fix it, but I can't — the provided data has no scene identifiers, so it's a limit of the input rather than a choice I made. I'd still treat that number as an upper bound.
