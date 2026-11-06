# Local regression of the 0.5.0 Stable profiles on an RTX 3090, 2026-11-04 to 2026-11-06

The qualification run on the tag `ai-v0.5.0` (41560779069) left two gaps. `cuda (runpod)` never started, because RunPod had no RTX 3090 for its 30-minute wait, and `release gates embeddinggemma-2/video-fp32-cpu` could not download its pinned FFmpeg builds (#277). This entry fills both gaps on a local machine at the tag's commit, `ce5225854fa7c007fc1cfb9adb50e665e6c1f611`.

**Result.** No model regression. Every gate, test and soak passed on the seven Stable CUDA profiles and on `embeddinggemma-2/video-fp32-cpu`. One recorded check is `failed`: `builds`, a stale lock file at the tag (below). Nothing here changes a manifest, a gate status, a tolerance or a `protected/` file.

**This is regression evidence, not a qualification round.** It isn't a G4 or G5 run, and no manifest cites it as gate evidence. The 10-minute soaks are screening runs, and no benchmark was measured.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, driver 615.78.08 (the 2026-10-16 binding round ran on 580.x) |
| CUDA libraries | CUDA 13 and cuDNN 9 from the tag's hash-locked `requirements-cuda.txt`, on `LD_LIBRARY_PATH` |
| CPU | Intel Core i7-5820K, 12 logical CPUs, no CPU quota, so the Python reference stages ran with 12 threads |
| RAM | 94 GB, shared with other work |
| Python | 3.11 from uv, in the reference environments the script builds at the tag |
| .NET SDK | 10.0.100, as `global.json` pins it |
| FFmpeg | The archives the tag pins, each checked against its SHA-256 before use (FFmpeg 8 `bf3db18a…`, FFmpeg 9 `5838264a…`) |

The machine is shared, so other jobs ran during these runs. The CUDA run's cgroup was frozen twice between 20:30Z and 20:38Z on 2026-11-04, during the EmbeddingGemma 2 text gates, while other work held the memory. Those gates compare outputs, not time, and both freezes ended before any soak started. The video CPU gates and the one-hour soak were not frozen.

## What ran

| Run | Commit | Result | Records |
| --- | --- | --- | --- |
| `tools/ci/runpod-cuda-pod.sh` on the local machine: CUDA tests, G1 to G3 on the release sets, a 10-minute soak per profile | tag | 31 of 32 checks passed; `builds` failed | [`cuda/summary.tsv`](cuda/summary.tsv), [`cuda/status.txt`](cuda/status.txt), [`cuda/reports/`](cuda/reports), [`cuda/soak/`](cuda/soak) |
| `tools/release-gates.sh embeddinggemma-2/video-fp32-cpu`, `--threads 4` | tag | G1, G2 and G3 passed | [`video-cpu/`](video-cpu) |
| One-hour CUDA soak, `qwen3.5-0.8b/ortgenai-fp32x-cuda` | tag | PASS under the G4 growth rule | [`soak-1h/`](soak-1h) |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda` with the video fixtures and FFmpeg set | tag | 29 passed, 0 failed, 0 skipped | [`video-tests.txt`](video-tests.txt) |

### CUDA test projects

| Project | Passed | Failed | Skipped |
| --- | --- | --- | --- |
| `Starlings.Models.E5.Tests.Cuda` | 10 | 0 | 1 |
| `Starlings.Models.Gliner2.Tests.Cuda` | 8 | 0 | 0 |
| `Starlings.Models.EmbeddingGemma2.Tests.Cuda` | 21 | 0 | 8 |
| `Starlings.Models.Qwen35.Tests.Cuda` | 116 | 0 | 0 |

The e5 skip needs a machine where the CUDA provider can't load. The 8 EmbeddingGemma 2 skips need `STARLING_FFMPEG_DIR`, which the pod script doesn't set, so the job has been skipping them on RunPod too (7 of 26 on the `ai-v0.4.0` run). They all passed in the separate run in the table above.

### 10-minute soaks

| Profile | Cycles | Requests checked | Wrong outputs | Errors |
| --- | --- | --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cuda` | 214 | 38,520 | 0 | 0 |
| `gliner2-base-v1/onnx-fp32-cuda` | 129 | 11,610 | 0 | 0 |
| `embeddinggemma-2/onnx-fp32-cuda` | 87 | 4,698 | 0 | 0 |
| `embeddinggemma-2/image-fp32-cuda` | 73 | 292 | 0 | 0 |
| `embeddinggemma-2/video-fp32-cuda` | 53 | 159 | 0 | 0 |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | 69 | 759 | 0 | 0 |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | 122 | 1,342 | 0 | 0 |

The growth verdict of a 10-minute soak isn't binding, because a short run sits in the allocator's warm-up ramp. Six profiles reported a plateau. `qwen3.5-0.8b/ortgenai-fp32x-cuda` reported `FAIL: loaded working_set grows 36 MB (277 MB/h); loaded rss_anon grows 35 MB (267 MB/h)`, so that profile also ran for an hour.

### One-hour soak of `qwen3.5-0.8b/ortgenai-fp32x-cuda`

PASS with the tool's unchanged growth rule: 854 load and unload cycles, 9,394 requests checked, 0 wrong outputs and 0 errors. Loaded `rss_anon` grew 4.3 MB (limit 32 MB) and loaded `working_set` 4.8 MB. Every gating series is a plateau. So the 10-minute reading was the warm-up ramp. The soak ran on an otherwise idle GPU. This is one profile's run on this machine, and it doesn't replace the binding G4 record.

### `embeddinggemma-2/video-fp32-cpu`

G1 and G2 passed against the release set, with all 11 hosted reference files verified. G3 passed with nDCG@10 0.8126 (minimum 0.797) and MRR@10 0.7714 (minimum 0.751), the same values as the local runs recorded in #277. The G3 producer took 6,027 s and ran to completion. Peak memory for the whole service was 12.2 GB.

## The `builds` failure

At the tag, the CUDA builds of `Starlings.Qwen35.Fidelity` and `Starlings.Soak.Tool` fail locked restore with `NU1004`. Their `packages.cuda.lock.json` files still carried the 0.4.0 versions of Starling's own packages and lacked `Starlings.Models.Abstractions`. So the tagged commit couldn't have passed this job, even with a GPU: without these two tools, the Qwen gates and every soak would have failed.

To measure the profiles anyway, both tools were built at the tag from a separate clone with `-p:RestoreLockedMode=false`, into the paths the script expects. [`cuda/unlocked-lock-changes.diff`](cuda/unlocked-lock-changes.diff) is what restore changed: only `Starlings.*` versions moving from 0.4.0 to 0.5.0, and the new `Starlings.Models.Abstractions` entries. No third-party package moved. `builds` stays `failed` in `summary.tsv`, because that's what happened at the tag.

`main` no longer has this problem: `2d9c6e74b` regenerated these lock files. Nothing in the release path caught it before the tag.

## Differences from the tagged scripts

- `cuda/harness-changes.diff` is the whole difference from the tag's pod script: the two root paths, `-m:2` on `dotnet build` (the machine's memory rules), and one line for a bug that only shows off a pod. With no CPU quota, `cpu_quota` prints ` none`, `read` strips the space, and `none` reaches arithmetic under `set -u`. The line routes that case to the script's own no-quota branch.
- `video-cpu/harness-changes.diff` is the whole difference from the tag's `release-gates.sh`: `fetch_ffmpeg` is a no-op because the archives are already in place, and `-m:2`.
- The release media and hosted references were fetched and verified by the scripts themselves. The model cache was this machine's, verified by `tools/acquire-models.sh`.
- The pinned FFmpeg archives came from a local copy of the mirror proposed in #286. Their digests equal the tag's pins, so the bytes are the ones the gates specify. This doesn't settle the mirror's open licence question, and it doesn't make the job fetch them: #277 stays open until `release-gates` passes in the qualification workflow.
