# BjjVision

Fighter identification and re-identification for Brazilian Jiu-Jitsu match video.
SAM2 masks, a gi-colour identity anchor, a self-auditing recalibration loop, and
a VLM supervisor that arbitrates what the pixels cannot.

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
