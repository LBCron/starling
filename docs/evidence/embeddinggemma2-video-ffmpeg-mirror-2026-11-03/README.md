# embeddinggemma-2 `video-fp32-cpu` from the FFmpeg mirror (#277)

Run on 2026-11-03 and 2026-11-04 at `493765b` on the maintainer's shared machine, CPU only, 4 threads. BtbN deleted the pinned FFmpeg autobuilds, so this job had produced no report since at least 2026-10-26. The two LGPL archives now come unchanged from `Starlinghq/ffmpeg-lgpl-shared-builds`, commit `ebbcfa02b5e6785acba4d8ad4361f320f94531af`.

**All three gates passed, but not in one run of `tools/release-gates.sh`.** That run exited 1 because its G3 producer was killed. G3 then passed from a separate run of the same two commands. No run of the script has yet exited 0 on the mirror.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids 11/11; frame indices, decoded frames, resized pixels, pixel values and positions exact 12/12; still image exact; decoder frames vs torchcodec exact 12/12 on FFmpeg 8 and on FFmpeg 9 | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 | **passed** | embedding max abs 1.29e-6 (1e-5); min cosine 0.999999999959 (0.9999995); soft tokens 2.37e-4 row-relative (1e-3); token embeddings 3.81e-4 (4e-3) | same |
| G3 | **passed**, separate run | `video-retrieval-v1`, 50 queries, 50 videos: nDCG@10 0.8126 (minimum 0.797), MRR@10 0.7714 (minimum 0.751) | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

These equal the figures of the [2026-10-14 evidence](../release-set-gates-2026-10-14/embeddinggemma2-video-fp32-cpu/README.md). The G3 embeddings are byte-identical to that evidence's run of 2026-10-15: the results file has SHA-256 `abda524cf373d68f19b05348df75378e51e76267ad0392c1166107f2bf3f38c1` in both reports, and `metrics` and `per_query` are equal. Nothing under `protected/` changed, and the evaluator verified the protected files against Git.

Each `report.json` differs from the evaluator's file only in its local paths, which read `<work>`. That text is not part of any digest.

## The fetch

In the script run, both fetch steps passed from the mirror with their SHA-256 checks (`bf3db18a…`, `5838264a…`). `fetch_eg2_ffmpeg.sh` makes no use of the GPL archive by default. With `--with-gpl` and no local GPL archive it exits 1 and says why; with one it verifies `4f58c09f…` as before.

## The killed producer

In the script run, the G3 producer exited 137 (SIGKILL) at about 07:18:32Z (23:18:32 local time), after 11 of 50 videos, so the evaluation had no results file and reported exit 2. That time is the start of the producer, 22:59:25 local, plus the 1,147 s the script recorded, and it matches the copy of the producer's log made just after exit. The log's own last write, at 23:17:32, is the line for the eleventh video, not the kill.

The cause was not found. The user journal shows no systemd-oomd action, the kernel log is not readable from this account, the parent shell survived, and no core dump exists. CI jobs started at 07:16:34Z (`ai-dotnet`), 07:18:12Z (`ai release-candidate drill`) and 07:18:14Z (`docs site (docfx)`), so two of them started 18 to 20 s before the kill, and the CPU load was 39.8 on 12 cores. That is correlation only.

The rerun used the same arguments in a scope with `MemoryMax=20G` and `MemorySwapMax=0`. It passed the point of failure and finished all 50 videos in 3,616 s. RSS has no data for about the first 9.5 minutes: the run started near 23:23:27, the first sampler read the wrong process and logged 4 MB, and the corrected one started at 23:33:01. The samples do cover the first run's kill point, 1,147 s into the run, and everything after it. Over that interval the peak was 7.2 GB, and the means of ten consecutive samples stayed between 5.4 and 6.5 GB. That argues against growth in the producer. It does not rule out a kill by something else on the machine, and it cannot say what happened in the first run.

## Not covered

- The `tools/ci/runpod-cuda-pod.sh` change was linted, not run. No GPU pod was used.
- This does not produce release-set evidence for the 0.4.0 or 0.5.0 tags themselves. That needs the qualification workflow.
- The evaluator ran from a fresh install of this tree. The job installs it the same way, unlocked (`pip install -e model-pipeline`). Only the job's reference environment is hash-locked, and the reference Python only fetched the media, here and in the job.
- A green script run. The kill is unexplained and could recur on the shared runners, so #277 stays open until the qualification workflow passes this profile.
- The 22 fixture videos of `video-fp32-cpu`. CI does not regenerate them. Regenerating them needs the GPL build, which upstream deleted and the mirror does not carry, so it takes a private copy (`fetch_eg2_ffmpeg.sh --with-gpl`).

## Open: third-party source

The mirror carries FFmpeg's source, not the source of the libraries linked into these builds. In the FFmpeg 8 build, the `NEEDED` entries of `libavcodec`, `libavformat` and `libavfilter` name only FFmpeg and system libraries, and `lib/` bundles no other library, yet `libavcodec.so.62` holds `libtwolame` strings and `libavformat.so.62` holds `libbluray version 1.5.1`. Their exact revisions cannot be identified from what is published. The licences of those projects were not checked here. Whether the redistribution is acceptable is a maintainer decision, recorded on the pull request.

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
