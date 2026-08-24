# BjjVision

A labelling engine for Brazilian Jiu-Jitsu match video.

**The deliverable is a labelled dataset, not a video.** The end goal is a position
classifier trained on per-athlete pose sequences; the annotated video exists so a
human can verify the extracted data is correct. See `PROMPT.md` for the full
framing.

The chain is: **colour-anchored mask → per-athlete pose → dataset.** The mask
pipeline is the enabler, not the product — off-the-shelf pose estimators fail in
grappling because they cannot tell whose limb is whose once the bodies fuse, and
the gi-colour anchor is what solves that.

What follows describes the mask stage: SAM2 masks, a gi-colour identity anchor, a
self-auditing recalibration loop, and a VLM supervisor that arbitrates what the
pixels cannot.

## The core idea

Classical person Re-ID assumes you can crop a bounding box around one person and
describe them. In grappling that assumption collapses: for most of a match the two
athletes' boxes overlap almost completely, so a box-cropped embedding describes
*the pair*, not either fighter. Every downstream identity decision inherits that
ambiguity.

Two things fix it.

**A mask instead of a box.** SAM2 gives a per-pixel assignment, so the two
athletes occupy genuinely disjoint pixel sets even when interlocked.

**Colour as the anchor, not the tracker.** The gi colour is a physical invariant
of the entire match. So it is not one appearance cue among several — it is the
ground truth that every other cue is corrected *against*. Tracking supplies
continuity; colour supplies correction. When SAM2 swaps the athletes during a
scramble, colour catches it on the very next frame rather than propagating the
error for minutes.

The same CIELAB histogram does double duty: as a *signature* it measures the
distance from a mask to a fighter prototype; as a *likelihood table* it gives
per-pixel `P(fighter | colour)`, which is what detects a contaminated mask and
splits it back into two without re-running the segmenter.

## Measured on synthetic adversarial data

`python tests/test_color_anchor.py` — CPU, no GPU needed.

| Property | Result |
|---|---|
| Gi separability (white vs navy) | 0.946 |
| Mask purity at 92% body overlap | A 1.00 · B 0.81 |
| **Purity after an injected identity swap** | **A 0.17 · B 0.00** — caught in one frame |
| Merged-blob mask re-split by colour | IoU A 0.90 · B 0.80 |
| Referee distance to both gi prototypes | 0.98 / 0.75 (rejected above 0.35) |
| Prototype bank vs. a poisoned exemplar | rejected, bank did not grow |

## Measured on the actual footage

Both matches are IBJJF broadcast, 720p30. Colours sampled from verified torso
crops; gi assignment read off the broadcast scoreboard (the colour block beside
each score is the competitor's gi).

| Comparison | Galvão–Xande | Buchecha–Lo |
|---|---|---|
| white gi vs blue gi (identity) | 0.84–0.87 | 0.96 |
| blue gi vs **blue mat** | 0.996 | 0.986 |
| white gi vs blue mat | 1.000 | 0.905 |
| black-suited official vs blue gi | 0.930 | — |
| black-suited official vs white gi | 0.882 | — |

The blue-gi-on-blue-mat case looked like the obvious way for this to fail, and it
does not — but for a reason worth stating, because it was luck-adjacent. The navy
gi measures `L=33, b=93`; the mat measures `L=138, b=94`. The *chroma is
effectively identical* — both are blue — and the entire separation lives in
lightness. A hue- or saturation-based colour model, which is the more common
choice, would have collapsed these two into one class and bled every mask into
the floor. Keeping L in the histogram is what makes this work.

## Broadcast structure: this is cut footage

The single biggest correction to the original design. These are multi-camera
broadcasts, not a locked-off mat camera:

| | Galvão–Xande | Buchecha–Lo |
|---|---|---|
| runtime | 12:57 | 14:00 |
| shot cuts | 24 | 63 |
| median shot | 15.1 s | 11.7 s |
| trackable (two athletes on mat) | 94% | 89% |

SAM2's memory attention assumes temporal continuity. Propagated across a hard cut
it maps the previous mask onto an unrelated camera angle and reports high
confidence while doing it. So propagation windows are bounded by *shots*, and a
cut forces a full re-detect rather than a colour re-seed.

This is where colour-as-anchor earns its keep: a cut destroys the tracker's state
but leaves the prototypes untouched, so the fresh detections are re-bound to A and
B immediately. Non-trackable shots — close-ups, podium, chroma transition plates —
are detected, passed through unmodified, and labelled as untracked rather than
being fed to a two-athlete tracker that would invent a second competitor.

**Threshold calibration.** Cut sensitivity is `z > 150` on MAD-normalised frame
dissimilarity. The first attempt used `z > 10` on the reasoning that a false
positive costs one cheap re-anchor while a false negative costs a corrupted
window. That reasoning has a limit it did not state: `z=10` found 381 cuts in
840 s — a reset every 2 s, which discards the benefit of propagation entirely.
The real separation is wide: genuine cuts measure z = 270–2900, camera and athlete
motion tops out near z = 40. Validated by sampling six kept cuts (all real) and
six rejected ones (all false positives).

## Athlete vs referee vs crowd

Three filters with deliberately non-overlapping failure modes:

1. **Mat membership** — learned as a *colour model*, not a fixed polygon, because
   broadcast cameras pan and zoom and a hardcoded region drifts off the mat.
2. **Sustained contact** — the load-bearing signal. The two competitors are in
   near-continuous contact; the referee circles and only brushes past. Integrated
   over a rolling window so a referee leaning in to check a choke does not
   register as grappling.
3. **Colour outlier** — anyone far from *both* gi prototypes is not a competitor.

Posture is deliberately not trusted alone. "The referee stands, the fighters are
on the ground" holds for most of a match and fails for exactly the opening
exchange — which is when identity is first being locked in.

## The recalibration loop

Every frame is a hypothesis to be audited, not a result to be trusted. Health is
measured on five axes that fail at *different* times:

| Signal | What it catches | When it fires |
|---|---|---|
| `purity` | mask pixels disagreeing with their own gi | earliest — before the shape looks wrong |
| `proto_dist` | slow colour drift | gradual degradation |
| `cross_iou` | one mask bleeding into the other | outright bleed |
| `area_jump` | mask collapse or explosion | discontinuity |
| `sam2_score` | segmenter self-confidence | honest but late; identity-blind |

Escalation ladder, cheapest first — `HEALTHY → SOFT` (re-split from the colour
posterior, ~2 ms, no segmenter call) `→ HARD` (re-prompt SAM2 from colour-confident
points, with the opponent's pixels supplied as *negative* prompts) `→ ESCALATED`
(hand it to the VLM).

Two details carry disproportionate weight:

- **Negative prompts on the opponent.** Under entanglement "this is A" is a weak
  constraint — the boundary between two interlocked bodies is genuinely ambiguous
  from shape alone. Adding "and *that* is explicitly not A" resolves it in one step.
- **The prototype update gate.** Exemplars enter the bank only from high-confidence
  frames. Update on every frame and one bad frame poisons the prototype, after
  which the next bad frame looks *consistent* with the poison and the error becomes
  self-reinforcing. The prototype must be harder to move than the thing it corrects.

## The supervisor

`claude-opus-5` in three separate roles: `adjudicate()` on escalation only (rules
on identity from annotated frames and isolated cut-outs), `narrate()` on a slow
cadence (position and event commentary for the overlay), and `tune()` once per pass
(reads the run's own telemetry and proposes threshold changes for the next pass).

It is instructed to abstain rather than guess — a confident wrong verdict is
written into the prototype bank and becomes the new reference, which costs far more
than an honest "unclear".

## Layout

```
src/bjjvision/
  ingest.py         yt-dlp + ffmpeg normalisation (runs locally)
  detect.py         YOLO11-pose + ByteTrack
  segment.py        SAM2 windowed propagation, colour-guided re-anchoring
  appearance.py     gi colour model — signature and per-pixel likelihood
  roles.py          mat model, contact tracker, athlete/referee/crowd
  identity.py       health audit, escalation ladder, prototype bank
  llm_supervisor.py adjudicate / narrate / tune
  render.py         composited on-screen report
  pipeline.py       orchestrator
remote/             VAST.AI provisioning and run notes
tests/              synthetic adversarial validation, HUD demo
```

## Usage

```bash
bjjvision doctor
bjjvision fetch "<youtube-url>" --name galvao-1
bjjvision sync-up galvao-1 root@<vast-host>
# on the GPU host:
bash remote/bootstrap_vast.sh
python -m bjjvision.cli frames galvao-1
python -m bjjvision.cli run galvao-1 --max-frames 900   # always smoke-test first
python -m bjjvision.cli run galvao-1
# back local:
bjjvision sync-down galvao-1 root@<vast-host>
```

Why the download runs locally: YouTube blocks datacenter IPs, and VAST.AI is a
datacenter. The laptop does ingest and delivery; the GPU host only does compute.
