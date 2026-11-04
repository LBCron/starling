# embeddinggemma-2 `video-fp32-cpu` from the FFmpeg mirror (#277)

Run on 2026-11-03 and 2026-11-04 at `493765b` on the maintainer's shared machine, CPU only, 4 threads. BtbN deleted the pinned FFmpeg autobuilds, so this job had produced no report since at least 2026-10-26. The two LGPL archives now come unchanged from `Starlinghq/ffmpeg-lgpl-shared-builds`, commit `ebbcfa02b5e6785acba4d8ad4361f320f94531af`.

**All three gates passed, but not in one run of `tools/release-gates.sh`.** That run exited 1 because its G3 producer was killed. G3 then passed from a separate run of the same two commands. No run of the script has yet exited 0 on the mirror.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids 11/11; frame indices, decoded frames, resized pixels, pixel values and positions exact 12/12; still image exact; decoder frames vs torchcodec exact 12/12 on FFmpeg 8 and on FFmpeg 9 | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 | **passed** | embedding max abs 1.29e-6 (1e-5); min cosine 0.999999999959 (0.9999995); soft tokens 2.37e-4 row-relative (1e-3); token embeddings 3.81e-4 (4e-3) | same |
| G3 | **passed**, separate run | `video-retrieval-v1`, 50 queries, 50 videos: nDCG@10 0.8126 (minimum 0.797), MRR@10 0.7714 (minimum 0.751) | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

These equal the figures of the [2026-10-14 evidence](../release-set-gates-2026-10-14/embeddinggemma2-video-fp32-cpu/README.md). Nothing under `protected/` changed, and the evaluator verified the protected files against Git.

## The fetch

In the script run, both fetch steps passed from the mirror with their SHA-256 checks (`bf3db18a…`, `5838264a…`). `fetch_eg2_ffmpeg.sh` makes no use of the GPL archive by default. With `--with-gpl` and no local GPL archive it exits 1 and says why; with one it verifies `4f58c09f…` as before.

## The killed producer

In the script run, the G3 producer exited 137 (SIGKILL) at 23:17:32 local time, after 11 of 50 videos, so the evaluation had no results file and reported exit 2. The cause was not found. The user journal shows no systemd-oomd action, the kernel log is not readable from this account, the parent shell survived, and no core dump exists. Three CI runner jobs started at about that moment, and the CPU load was 39.8 on 12 cores.

The rerun used the same arguments in a scope with `MemoryMax=20G` and `MemorySwapMax=0`. It passed the point of failure and finished all 50 videos in 3,616 s, with a sampled peak RSS of 7.2 GB and RSS flat near 6 GB. That argues against growth in the producer. It does not rule out a kill by something else on the machine, and it cannot say what happened in the first run.

## Not covered

- The `tools/ci/runpod-cuda-pod.sh` change was linted, not run. No GPU pod was used.
- This does not produce release-set evidence for the 0.4.0 or 0.5.0 tags themselves. That needs the qualification workflow.
- The model-pipeline environment was a fresh install of this tree rather than the job's hash-locked one. The reference Python only fetched the media, as in the job.

## Commands

```sh
export STARLING_MODELS=<model cache>
tools/release-gates.sh embeddinggemma-2/video-fp32-cpu --out <out> --work <work> --reference-python <eg2-video venv>/bin/python --threads 4
```

The G3 rerun, with the FFmpeg and media that run fetched:

```sh
SET=model-pipeline/protected/embeddinggemma-2/release/video-retrieval-v1
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll \
  retrieval --model embeddinggemma2-video --dataset $SET --media <work>/media/video-retrieval-v1 --out <work>/g3/embeddings.jsonl \
  --ffmpeg <work>/ffmpeg/ffmpeg8/ffmpeg-n8.1.3-17-gc389840894-linux64-lgpl-shared-8.1/lib --snapshot <snapshot> --threads 4
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/video-fp32-cpu/manifest.yaml --g3 retrieval --dataset $SET \
  --results <work>/g3/embeddings.jsonl --out <out>/g3-retrieval
```

Wall clock (shared machine, not benchmarks): media 417 s; .NET stages 1,058 s; G3 producer 3,616 s.
