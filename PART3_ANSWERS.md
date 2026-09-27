# Part 3 — Problem-Solving Answers

*Short answers. Reasoning over volume. Each one is anchored in the code I actually
submitted, so you can check the claims against `server/`.*

---

## 1. The classifier is wrong 30% of the time. What do you do, and how do you decide if it's "good enough"?

**First: 30% is not a number yet, it's a description.** Error rate only means something
once I know *where* the errors are and *what they cost*. Three things have to be
established before touching the model.

**Is it the model that's wrong, or my labels?** I sampled ~50 tiles and had a human
label them independently. If my labels disagree with each other at a similar rate to
how they disagree with the model, then a chunk of the 30% is label noise and no amount
of retraining helps — the ceiling is the annotator agreement, not the network. This is
the single most important check and the one most often skipped, because it's
unglamorous and it can kill the entire project.

**Is the error uniform or structured?** On my eval set, 87.6% accuracy, but the errors
are concentrated: `SeaLake → Forest` (6) and `AnnualCrop → River` (5) account for most
of the loss, while `Industrial` and `Highway` are nearly perfect. That is much more
useful than a flat "30% wrong". It tells me the model isn't weak, it has two specific
confusions — water-adjacent vegetation and water-adjacent crops, which are genuinely
ambiguous in a 64×64 crop with no season or location context. That's a data problem
with a known shape, not a modelling problem.

**Does it beat the alternative?** The bar is not "perfect", it's "better than what the
analyst does today". If the current process is a human eyeballing every tile, then a
model that reliably shortlists candidates beats it — the human still decides, the
model just orders the queue. Against that bar, 70% accuracy with good *ranking* is
genuinely useful. Against a bar of "auto-populate a report that gets signed", it is
not, and no amount of tuning changes that without also changing the interface.

**So the test I would use is precision on the review queue, not overall accuracy.**
Concretely: if I flag the lowest-confidence 10% of tiles, what fraction of those flags
are actually the class I care about? I ship only if that number clears a threshold we
agree on with the analyst, and I tune to move that number rather than to move accuracy.

**What I'd actually do, in order:**
1. Measure label noise first. Cap the achievable accuracy at annotator agreement.
2. Break error down per class and per confusion, not as one aggregate.
3. Decide the operating point with the people who will use the output — this is a
   business decision about the cost of a miss versus the cost of a review.
4. Ship the "predict and flag" pattern, never "predict and assert". The interface must
   make uncertainty visible so the human is never misled into thinking the model was
   sure.
5. Log everything — hash, confidence, full probability vector, model version — so that
   when the 30% figure is disputed later, it's answerable with a query rather than an
   opinion.
6. Only then invest in accuracy, and only in the two confusions that are actually
   costing us.

**The honest framing:** "good enough" is never a property of the model alone. It's a
property of the model *and* the workflow it's dropped into. The same 70%-accurate model
is useless behind a "Commit" button and valuable behind a "here are 50 tiles worth
looking at" button. I build the second one deliberately.

---

## 2. Offline, nobody watching. A month after deployment, how do you know it's still working correctly?

**Starting from the honest position: you don't, unless you built the check before you
needed it.** A service that nobody is watching degrades silently and finds out at the
worst moment. So the monitoring has to be a property of the system, not a dashboard
somebody remembers to open.

**The key distinction: liveness is not correctness.** `/health` returning `200` proves
a process is running. It says nothing about whether that process is still loading the
right weights, still normalising pixels correctly, or still writing results properly.
Most of the failure modes that matter here are invisible to a health check. I'd rather
have no health endpoint than believe it covers the risk.

**What I'd implement, cheapest first:**

**A scheduled self-test against held-out ground truth.** This is the real answer. I keep
a labelled gold set that is never used for training, and a small job (cron, on the box —
there's no cloud to alert from) runs it through the full serving path on a schedule and
writes a status file. It compares accuracy against a recorded baseline and exits
non-zero on regression. This converts "is it still working" from an opinion into a
number that anyone can check, and it reuses the test harness I need anyway. It also
catches the failure that matters most: something changing that I didn't anticipate.

**Confidence distribution monitoring.** The cheapest high-signal metric available. I
log confidence on every request, so mean confidence and the flagged-rate over a rolling
window are single queries. Sudden drops in mean confidence, or a jump in flagged rate,
are hard to produce any other way. In my own system, a collapse from ~0.88 mean
confidence to ~0.5 would be screaming before a single accuracy complaint arrived.

**Input drift tracking.** Every request logs the tile's SHA-256, name, byte size, and I
can cheaply log basic pixel statistics. Hashing the *inputs* turns "have we started
seeing new kinds of imagery?" into a set-difference against the training distribution.
A sudden population of never-before-seen hashes is a strong upstream signal.

**Integrity checks on the artifact.** `model_version` is a hash of the weights, recorded
on every prediction. Comparing the version in the logs against the file on disk catches
a corrupted, truncated, or accidentally-swapped model — and because every stored row
carries its version, it makes "this data was classified by a different model" a filter
rather than a mystery.

**Boring operational checks.** Disk space on the results volume and log directory, error
rate, request rate, p95 latency, and a check that results are actually still being
written. Disk-full is the classic silent killer: requests keep succeeding while nothing
is persisted, and nobody notices for a month.

**What I would not do:** build a metrics stack. A time-series database, a dashboard, and
an alerting pipeline are all the wrong shape for a single offline box. A nightly job
that writes one JSON status file, plus a log anyone can `grep`, covers most of the risk
at a fraction of the complexity. If someone isn't checking the status file, a fancy
alert wouldn't have helped either.

**The failure I'd most expect, which none of the above fully covers:** someone updates a
dependency, the ImageNet normalisation constants change, and predictions quietly shift
while every health check stays green. The gold-set self-test is the only thing that
catches that, which is why it is the one I'd insist on.

---

## 3. Tiles are arriving fine, but stored results look wrong. How do I find the cause?

**The temptation is to suspect the model first. I would resist that.** The model is the
slowest thing to check and by far the least likely. I work the pipeline from input to
output, cheapest and most-likely first, and I make each step falsifiable before moving
on.

**Step 0 — one crucial discriminator, before anything else.** Take one tile whose true
label I know. Run it through `POST /classify` and compare three things: the true label,
the **live API response**, and the **stored row**. This splits the problem in half
immediately:
- API response wrong, stored row agrees → the bug is in inference or the model. Stop here.
- API response right, stored row wrong → the bug is in the write path. The model is
  innocent and I should not look at it.

Half of all "results look wrong" tickets die at this step, and it costs two minutes.

**Step 1 — is it everything or some of it?** Query the store. Filter by time: is the
damage recent, or has it always been there? A sharp break at a specific timestamp points
hard at a deploy, a config change, or a data-source change, and gives me an exact moment
to diff against. A gradual drift points at drift itself. A total failure on one class
and not others points at the class mapping. The shape of the wrongness is the biggest
clue available and it costs one query.

**Step 2 — are the inputs what I think they are?** Verify the tiles are actually valid
imagery: decode them, check dimensions and channel count, and compare their statistics
to the training distribution. A pipeline upstream may have started sending grayscale
tiles, tiles at a different resolution, or tiles that are actually a different band or
sensor entirely. Compare the SHA-256s and sizes in the logs against what I expect to be
seeing. This is cheap and it is upstream of everything else.

**Step 3 — is the preprocessing the same at serving time as it was at training time?**
This is the single most common cause of a model that "suddenly" got worse, and it is
silent — no exception, no log line, just quietly wrong numbers. The candidates:
different resize interpolation, a different normalisation mean/std, RGB/BGR confusion,
a different input size, or the training transform being reused on inference. I verify
this by taking training tiles, running them through the **exact serving code path**, and
checking that the predictions match what training produced. If they don't, the bug is
here, and I've found it without touching the weights at all.

**Step 4 — is it the model artifact?** Compare the `model_version` in the logs against a
fresh hash of the file on disk. A truncated download, a partial save, or a stale
`model.pt` from an older run all present identically: correct code, wrong weights. I
reload from a known-good copy and confirm the version changes.

**Step 5 — class index mapping.** If predictions look *permuted* — everything is right
except the labels are scrambled in a consistent way — the cause is almost always the
ordering of the class list drifting apart from the ordering the trained head was fitted
with. Reorder `CLASSES` and every prediction is confidently wrong while the model is
untouched. It looks like a catastrophic model failure and is a one-line fix.

**Step 6 — the storage layer itself.** If the live response was correct, stop trusting
the CSV and look at how the row was written. Column shift, encoding issues, a truncated
final line from a crash mid-append, or the boolean coercion path mangling `needs_review`.
I read the raw file directly rather than through my own query code, because I do not
trust the code I am debugging to tell me the truth about itself.

**Step 7 — only now, the model.** Re-run the full held-out evaluation through the
serving path and compare against the stored baseline. If accuracy is intact and
production behaviour is wrong, the difference is in the data reaching it, not in the
model. If accuracy has genuinely collapsed, only now do I retrain or suspect corruption.

**The principle underneath all of it:** cheap to check, high prior probability first.
Every step above either confirms or eliminates a whole category of cause. I never change
the model until I have eliminated everything upstream of it, because "retrain and hope"
destroys the evidence — after a retrain, the original bug is unreproducible and you
learn nothing.

---

## 4. What's the weakest part of your design, and what would break it first?

**The storage layer, without question.** Everything else is either correct or fails
loudly. The CSV store is neither.

**What it actually is:** an append-and-scan file with no transactions, no locking, no
schema enforcement, no indexes, and no concurrency safety. It's a fine placeholder and
a genuinely bad database.

**What breaks first, concretely:**

**The most likely failure is concurrent writers.** CSV append is not atomic across
processes. It works perfectly for the single uvicorn worker I run and documented, and
the first person who deploys it with `--workers 4` — which is the obvious next thing
anyone would do — gets interleaved and corrupted rows. Nothing warns you. The service
keeps returning `200`, and the damage is discovered by an analyst three weeks later
when a row has a label in the confidence column.

**The nastiest failure is a truncated final line.** If the process dies mid-append, the
last row is half-written and is *not* valid CSV. My reader is tolerant of a malformed
line, but a stricter tool — pandas, a spreadsheet, the next person who writes a query —
will fail to parse the entire file and lose access to *every* row, not just the broken
one. All the good data, made unreadable by one bad line. That's a data-loss-shaped
failure from what looks like a read-only query problem.

**Why this is the weakest *design* choice and not just a weak implementation:** the rest
of the system quietly depends on it being trustworthy. The review queue is a query
against this file. My idempotency story is "dedupe by content hash", which only means
something if every row is intact. If a write is lost or corrupted, a tile silently
vanishes from the analyst's queue — and that is precisely the failure mode where a
review flag gets ignored, because nobody notices the tile that never showed up. The
storage layer's weaknesses propagate into a *silent* correctness problem in the workflow.

**How I'd fix it, in order:**
1. **Writes durable before anything else** — write to a temp file and atomically rename,
   or move to SQLite, which gives me real transactions and multi-process safety.
2. **WAL-style append with integrity checking** — if I must keep a flat file, log records
   with a checksum so a truncated tail is detectable and skippable rather than fatal.
3. **Then** proper indexing for the query surface, so filter cost stops growing with
   history.

**Honourable mention, second weakest:** my train/test split is random, not grouped by
scene. Tiles from the same satellite scene are near-duplicates, so a random split leaks
scene identity across the boundary and my reported 87.6% is optimistic. I flagged this
in the design note rather than quietly reporting the number. The fix is a
grouped-by-scene split, which I can't do because the provided data has no scene
identifiers. It's a limit of what I was given — but it means I'd treat that figure as an
upper bound, and the first thing I'd fix if I could get scene metadata.

---

*Where I'd spend the next week, in order: move storage to SQLite (fixes the worst thing
and the concurrency hole together); add the scheduled gold-set self-test from Part 3.2;
add per-class confidence thresholds; then look at the `SeaLake`/`AnnualCrop` confusions,
which are the only place the model is genuinely weak.*
