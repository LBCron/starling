# Local regression of the 0.5.0 Stable profiles on an RTX 3090, 2026-11-04 to 2026-11-07

The qualification run on the tag `ai-v0.5.0` (41560779069) left two gaps. `cuda (runpod)` never started, because RunPod had no RTX 3090 for its 30-minute wait, and `release gates embeddinggemma-2/video-fp32-cpu` could not download its pinned FFmpeg builds (#277). This entry fills both gaps on a local machine at the tag's commit, `ce5225854fa7c007fc1cfb9adb50e665e6c1f611`.

**Result.** No output regressed, on the tag or on `main`. One open finding remains: e5 CUDA's memory growth can fail the G4 rule on this machine (see [e5 CUDA growth](#e5-cuda-growth)). Every gate and test passed on the seven Stable CUDA profiles and on `embeddinggemma-2/video-fp32-cpu`, and every soak passed its smoke check, with one non-binding growth FAIL that the one-hour soak below resolves. One recorded check is `failed`: `builds`, a stale lock file at the tag (below). Nothing here changes a manifest, a gate status, a tolerance or a `protected/` file.

**This is regression evidence, not a qualification round.** It isn't a G4 or G5 run, and no manifest cites it as gate evidence. The 10-minute soaks are screening runs, and no benchmark was measured.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090, driver 615.78.08 (the 2026-10-16 binding round ran on 580.x). The machine also has an RTX 4060 Ti; `CUDA_VISIBLE_DEVICES=0` with PCI ordering selects the 3090 |
| CUDA libraries | CUDA 13 and cuDNN 9 from the tag's hash-locked `requirements-cuda.txt`, on `LD_LIBRARY_PATH` |
| CPU | Intel Core i7-5820K, 12 logical CPUs, no CPU quota, so the reference thread cap was 12 (the GLiNER2 stage sets 4) |
| RAM | 94 GB, shared with other work |
| Python | 3.11 from uv, in the reference environments the script builds at the tag |
| .NET SDK | 10.0.100, as `global.json` pins it |
| FFmpeg | The BtbN archives the tag pins, each checked against its SHA-256 before use (FFmpeg 8 `bf3db18a…`, FFmpeg 9 `5838264a…`) |

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

The e5 skip needs a machine where the CUDA provider can't load. The 8 EmbeddingGemma 2 skips need the video set (`STARLING_EMBEDDINGGEMMA2_VIDEOS`) and `STARLING_FFMPEG_DIR`. The pod script sets neither, so the job skips them on RunPod too (7 of 26 on the `ai-v0.4.0` run). They all passed in the separate run in the table above.

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

PASS with the tool's unchanged growth rule: 854 load and unload cycles, 9,394 requests checked, 0 wrong outputs and 0 errors. Loaded `rss_anon` grew 4.3 MB (limit 32 MB) and loaded `working_set` 4.8 MB. Every gating series is a plateau. So the 10-minute reading was the warm-up ramp: in this run `rss_anon` rose about 38 MB in the first 600 s, then about 5 MB/h. The GPU showed 0% utilisation at the start, with only resident idle processes. It wasn't sampled during the hour, and CPU load rose from about 2 to about 10 by the end. This is one profile's run on this machine, and it doesn't replace the binding G4 record.

### `embeddinggemma-2/video-fp32-cpu`

G1 and G2 passed against the release set, with all 11 hosted reference files verified. G3 passed with nDCG@10 0.8126 (minimum 0.797) and MRR@10 0.7714 (minimum 0.751), the same values as the local runs recorded in #277. The G3 producer took 6,027 s and ran to completion. Peak memory for the whole service was 12.2 GB.

## The `builds` failure

At the tag, the CUDA builds of `Starlings.Qwen35.Fidelity` and `Starlings.Soak.Tool` fail locked restore with `NU1004`. Their `packages.cuda.lock.json` files still carried the 0.4.0 versions of Starling's own packages and lacked the dependency edges 0.5.0 added, among them `Starlings.Models.Abstractions` on `Starlings.Models.Qwen35.OnnxGenAI` and the `starlings.models.abstractions` project entry in the Qwen lock. So the tagged commit couldn't have passed this job, even with a GPU: without these two tools, the Qwen gates and every soak would have failed.

To measure the profiles anyway, both tools were built at the tag from a separate clone with `-p:RestoreLockedMode=false`, into the paths the script expects, while the run was in progress and before it reached the Qwen gates. [`cuda/unlocked-lock-changes.diff`](cuda/unlocked-lock-changes.diff) is what restore changed: `Starlings.*` versions moving from 0.4.0 to 0.5.0, the new `Starlings.Models.Abstractions` entries, and one new `Starlings.Acquisition` dependency on `starlings.models.kev`. Every changed line is a `Starlings.*` package. No third-party package moved, and `Starlings.Tokenizers` stays at 0.3.4. `builds` stays `failed` in `summary.tsv`, because that's what happened at the tag.

`main` no longer has this problem: `2d9c6e74b` regenerated these lock files. Nothing in the release path caught it before the tag.

## Differences from the tagged scripts

- `cuda/harness-changes.diff` is the whole difference from the tag's pod script: the two root paths, `-m:2` on `dotnet build` (the machine's memory rules), and one line for a bug that only shows off a pod. With no CPU quota, `cpu_quota` prints ` none`, `read` strips the space, and `none` reaches arithmetic under `set -u`. The line routes that case to the script's own no-quota branch.
- `video-cpu/harness-changes.diff` is the whole difference from the tag's `release-gates.sh`: `fetch_ffmpeg` is a no-op because the archives are already in place, and `-m:2`. That patch is why the two video reports record the commit as `ce522585…-dirty`.
- The soak reports' notes say "scheduled CUDA qualification on RunPod" because the tag's script hard-codes that text. These runs were local. Their reports show `git_commit n/a`, so the commit comes from that note.
- [`harness/`](harness) holds the two scripts this entry adds to the tagged ones. `run-soak-1h.sh` is the one-hour soak. `run-video-tests.sh` is the test run, which also sets `MALLOC_MMAP_THRESHOLD_=65536`, unlike the pod's test step, and uses FFmpeg 8 for `STARLING_FFMPEG_DIR`.
- The release media and hosted references were fetched and verified by the scripts themselves. The model cache and NuGet cache were this machine's, and the models were verified by `tools/acquire-models.sh`. `images-r1` and `video-spike` were local fixture directories.
- The FFmpeg archives were a local copy of the BtbN builds the tag pins, with digests equal to those pins, so the bytes are the ones the tag's gates specify. BtbN has deleted them, so this run can't be repeated from the tag's URLs.

## Since the tag

#286 merged after these runs started and closed #277. It replaces those FFmpeg builds on `main` with builds from pinned sources, under new pins, and it recorded that the pod script and `embeddinggemma-2/video-fp32-cuda` had not run on them. So the pod script also ran at `main`.

## The pod script at `main`

`tools/ci/runpod-cuda-pod.sh` at `3a021dc0b7f8eb18f9d1f257cee53c9399043d34`, on the same machine, 10-minute soaks: **31 of 31 checks passed, exit 0**, with no freezes. [`main/harness-changes.diff`](main/harness-changes.diff) is the same four-line change. Records: [`main/summary.tsv`](main/summary.tsv), [`main/status.txt`](main/status.txt), [`main/soak/`](main/soak), [`main/video-cuda-reports/`](main/video-cuda-reports).

- **Locked builds.** All the CUDA tools built in locked mode, so the `builds` failure above is gone in practice, not only in the lock files.
- **FFmpeg.** The script fetched and verified the rebuilt archives from the pinned URLs.
- **`embeddinggemma-2/video-fp32-cuda` on the rebuilt FFmpeg.** G1 and G2 passed, with decoded frames exact on FFmpeg 8 and 9, and G3 and the 10-minute soak passed.
- **Tests.** The same results as the tag's table. With the video set and the rebuilt FFmpeg 8, `Starlings.Models.EmbeddingGemma2.Tests.Cuda` passed 29 of 29 with none skipped ([`main/video-tests.txt`](main/video-tests.txt)).

This doesn't replace the qualification workflow: it ran on this machine, not on the `qualification` runner or a pod. The workflow hasn't run on the rebuilt FFmpeg.

The 10-minute soaks had 0 wrong outputs and 0 errors in 83,195 checked requests. Their growth verdict isn't binding. Six reported a plateau. `multilingual-e5-small/onnx-fp32-cuda` reported growth, which led to the next section.

## e5 CUDA growth

**Finding.** On this machine, with default settings, `multilingual-e5-small/onnx-fp32-cuda` doesn't reliably pass the G4 growth rule. Native memory climbs by about 200 MB and levels off near 780 MB, and whether a run passes depends on how early it levels off. With `MALLOC_MMAP_THRESHOLD_=65536` it stays flat near 470 MB. Outputs were correct throughout. The evidence points to glibc heap fragmentation, not a leak.

| Run ([`e5-soaks/`](e5-soaks)) | Commit | Length | Verdict | Unloaded `rss_anon` fitted growth (1st, 2nd half), MiB | Cycles | Requests checked | Wrong / errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `tag-1h` | tag | 1 h | PASS | 115.4 (124.3, -2.0) | 1,844 | 331,920 | 0 / 0 |
| `tag-2h` | tag | 2 h | PASS | 121.2 (63.7, -5.7) | 3,498 | 629,640 | 0 / 0 |
| `tag-2h-gpu-overlap` | tag | 2 h | FAIL, loaded series | 29.8 (41.2, 12.9) | 3,532 | 635,760 | 0 / 0 |
| `main-1h` | main | 1 h | FAIL | 115.3 (87.9, 20.5) | 1,854 | 333,720 | 0 / 0 |
| `main-2h` | main | 2 h | FAIL | 174.6 (136.8, 53.9) | 3,704 | 666,720 | 0 / 0 |
| `main-2h-malloc-trim-stopped` | main | 43 min of 2 h | FAIL | 79.7 (56.3, 35.9) | 1,286 | 231,480 | 0 / 1 |
| `main-2h-mmap-threshold` | main | 2 h | **PASS** | 4.0 (8.0, 21.5) | 3,524 | 634,320 | 0 / 0 |

Unloaded `rss_anon`, median of each 10-minute bin, MiB:

| Run | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 110 | 120 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `tag-2h` | 606 | 657 | 684 | 684 | 690 | 714 | 727 | 769 | 783 | 780 | 774 | 769 |
| `main-2h` | 556 | 591 | 620 | 645 | 687 | 702 | 732 | 745 | 748 | 778 | 782 | 783 |
| `main-2h-mmap-threshold` | 469 | 470 | 470 | 466 | 476 | 481 | 474 | 462 | 465 | 470 | 475 | 478 |

What these runs show:

- **No regression between the tag and `main` is shown.** Both reach the same level. `tag-2h` got there by about 90 minutes and then held. `main-2h` got there by about 100 minutes, which left growth in its second half, so the rule failed it. The commits differ in no e5 or runtime logic, and the diff of those projects is metadata, documentation and lock files. This is one clean two-hour run each.
- **Fragmentation, not a leak.** `MALLOC_MMAP_THRESHOLD_` changes only how glibc places allocations, and with it the series are flat for two hours at about 300 MiB less, with request and load times about the same (p50 2.77 ms against 2.68 ms, load 1,209 ms against 1,115 ms). `rss_file` is flat in every run.
- **`malloc_trim` doesn't recover it.** In the stopped run, `trimmed` equals `unloaded`. That fits fragmentation, which leaves pages partly in use, and it also shows this isn't the free-memory retention that the runtime's trim on release handles.
- **Levelling off isn't proven beyond two hours.** No run went longer, and the bound rests on these curves.

What differs from the binding round: the 2026-10-16 G4 soak passed at 1 hour on a RunPod RTX 3090, with driver 580.126.20, under a CPU quota, with growth 67.8 MiB (42.0, -1.7). `docs/evidence/cuda-soak-growth/` reported a similar saturating curve on 2026-10-15. This entry doesn't re-run G4 on that host, so it doesn't say whether the same variation happens there.

**Decision for the maintainer.** The `Starlings.Models.E5` README says no `MALLOC_*` setting is required for this profile. The runs above don't contradict its outputs or its contract, and the growth looks bounded, so this isn't shown to be the "unbounded growth" that `docs/support-policy.md` lists as a regression. But the profile's G4 verdict can fail on a host like this one. Nothing here changes the README, a manifest or a rule.

Three runs need explaining:

- `main-2h-malloc-trim-stopped` stopped at 2,572 s because other work on the machine took about 16 GB of the 3090, and a session load then failed with an out-of-memory error from ONNX Runtime's GPU arena (the one error in the table). It wasn't a Starling fault. Up to that point the samples are valid, and the run's verdict covers that window only.
- `tag-2h-gpu-overlap` started right after that, while the other work still held the GPU. It started about 90 MiB higher than `main-2h` and is why `tag-2h` was run again on an idle GPU. It's kept for completeness and isn't used for the comparison.
- `tag-2h` and `main-2h-mmap-threshold` ran with GPU samples every 30 s, none above 3 GB. `main-2h` wasn't sampled, but it started and ended on an idle GPU.

The scripts are in [`harness/`](harness).
