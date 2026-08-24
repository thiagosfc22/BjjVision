# Running on VAST.AI

## Picking an instance

The pipeline is VRAM-bound on SAM2-Hiera-L, not compute-bound.

| GPU | VRAM | Verdict |
|---|---|---|
| RTX 4090 | 24 GB | Sweet spot. Best $/frame for this workload. |
| RTX 3090 | 24 GB | Fine, ~35% slower. |
| A5000 | 24 GB | Fine. |
| L40S / A100 | 48-80 GB | Works, but you are paying for VRAM this job never uses. |
| Anything ≤ 12 GB | — | Will OOM on Hiera-L. Drop to `sam2.1_hiera_base_plus` first. |

Filters when searching: **CUDA ≥ 12.1**, **≥ 50 GB disk** (frames are the bulk),
**≥ 200 Mbps** up (you are pulling a video back), template
`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel`.

Disk is the one people under-provision: a 10-minute match at 30fps/720p is
~18,000 JPEGs, roughly 2-4 GB, plus the composited output.

## Flow

Local:
```bash
bjjvision fetch "<youtube-url>" --name galvao-1
bjjvision sync-up galvao-1 root@<host> --remote-dir ~/BjjVision
```

On the instance (SSH in):
```bash
cd ~/BjjVision
bash remote/bootstrap_vast.sh
export ANTHROPIC_API_KEY=...        # supervisor; omit and pass --no-llm to skip
python -m bjjvision.cli frames galvao-1
python -m bjjvision.cli run galvao-1 --max-frames 900   # smoke test first
python -m bjjvision.cli run galvao-1                    # full match
```

Back on the laptop:
```bash
bjjvision sync-down galvao-1 root@<host>
```

## Always smoke-test with `--max-frames 900`

30 seconds of output tells you whether calibration locked onto the right two
people and whether the gi separability number is healthy. Finding that out after
a full-match run is how you burn an hour of GPU rental on a run that was
mis-calibrated in its first ten seconds.
