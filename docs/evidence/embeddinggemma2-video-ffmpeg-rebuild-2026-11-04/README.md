# embeddinggemma-2 `video-fp32-cpu` on FFmpeg rebuilt from pinned sources (#277)

Run on 2026-11-04 on the maintainer's shared machine, CPU only, 4 threads. BtbN deleted the pinned FFmpeg autobuilds, so this job had produced no report since at least 2026-10-26. The gates now use LGPL shared builds of the same two FFmpeg commits, built by `model-pipeline/reference/embeddinggemma2/build_eg2_ffmpeg.sh` and hosted, with all their source, in [`Starlinghq/ffmpeg-lgpl-rebuilds`](https://huggingface.co/datasets/Starlinghq/ffmpeg-lgpl-rebuilds) at commit `caf396f176296fc12620c18a344a31f628a4613a`.

**`tools/release-gates.sh` exited 0 at `9f6b202`, fetching both builds from the pinned URLs, with G1, G2 and G3 passed in that one run.** Decoded frames are exact, and the G3 embeddings are byte-identical to those from the BtbN builds. The qualification workflow still has to confirm this on its own runner.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 | **passed** | ids 11/11; frame indices, decoded frames, resized pixels, pixel values and positions exact 12/12; still image exact; decoder frames vs torchcodec exact 12/12 on FFmpeg 8 and on FFmpeg 9 | [`g1-g2-video-release/report.md`](g1-g2-video-release/report.md) |
| G2 | **passed** | embedding max abs 1.29e-6 (1e-5); min cosine 0.999999999959 (0.9999995); soft tokens 2.37e-4 row-relative (1e-3); token embeddings 3.81e-4 (4e-3) | same |
| G3 | **passed** | `video-retrieval-v1`, 50 queries, 50 videos: nDCG@10 0.8126 (minimum 0.797), MRR@10 0.7714 (minimum 0.751) | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

The G3 results file has SHA-256 `abda524cf373d68f19b05348df75378e51e76267ad0392c1166107f2bf3f38c1`, the same as the runs of 2026-10-15 and 2026-11-04 on the BtbN builds, and `per_query` is equal. `mrr@10` in `metrics` differs from the 2026-11-04 report in the sixteenth digit (0.771388888888889, now 0.7713888888888889) although its inputs are equal; this was not looked into. These gate figures equal the [2026-10-14 evidence](../release-set-gates-2026-10-14/embeddinggemma2-video-fp32-cpu/README.md).

The evaluator verified the protected files against Git, and nothing under `protected/` changed. It marks its own commit `9f6b202…-dirty`, because the only uncommitted change at the time was this evidence directory's rename. Each `report.json` differs from the evaluator's file only in its local paths, which read `<work>`.

## The builds

| Build | FFmpeg commit | Archive SHA-256 |
| --- | --- | --- |
| `n8.1.3-17-gc389840894` (8.1) | `c3898408946f6cbfe8e244372c250366d5639d00` | `240d0c45e4a684776f2ed6f841345c9cd1b7937a4e3e121104157fcd04941d68` |
| `n9.0.2-28-g669f5ab083` (9.0) | `669f5ab083c7732c10f1ce9e4f1915960dd88457` | `db79bc561add07fe48cbe013f714f4ceaf12e49553e56aff9775eff088f8b5ed` |

The commits are the ones the adapters' bindings and the protected references name. `tools/generate-ffmpeg-bindings.py --check` passes against both builds' headers for its major. The bindings keep their source line, which names the BtbN archive they were generated from.

**What the fixtures need.** ffprobe over the media of this run and the 22 fixture videos found H.264 (8 and 10 bit), HEVC, VP8, VP9 (8 and 10 bit, 4:2:0 and 4:2:2), Theora and AV1 (8 and 10 bit), with Vorbis and Opus audio, in MP4, WebM and Ogg. FFmpeg decodes all of these itself except AV1. So dav1d 1.5.4 (commit `54706fc6bc0cdecab7e9593974a4039cc038fca7`, BSD-2-Clause) is the one external library, linked statically into `libavcodec`. The still image of the fixtures is a video frame, so FFmpeg decodes no PNG here and zlib stays out. The BtbN build's configure line had 47 `--enable-lib…` options; none but `--enable-libdav1d` remains.

**Configure.** `--arch=x86_64 --target-os=linux --disable-autodetect --disable-iconv --enable-shared --disable-static --disable-debug --disable-doc --enable-libdav1d --pkg-config-flags=--static`, and an rpath for `bin/`. x86 assembly is on, built with nasm 2.15.05; the script fails without it. `--disable-autodetect` alone left glibc's iconv on, so it is off by name.

**Checks of the result, FFmpeg 8 and 9 alike:**

- configure's external library list is `libdav1d` alone, and its hardware acceleration list is empty. The script fails otherwise.
- `NEEDED` of every library and of `ffmpeg` and `ffprobe` names only FFmpeg's own libraries, `libc.so.6` and `libm.so.6`.
- `strings` finds no twolame, libbluray, OpenSSL, libxml2, libvpx, libaom, rav1e, SVT-AV1, libopus, libvorbis, libwebp, libass, freetype, fribidi, harfbuzz, fontconfig, libssh, SRT, RIST, ZeroMQ, GnuTLS, LAME or zlib library strings in any library. Three hits remain, all FFmpeg's own "not available" messages: two in `libavfilter`, about freetype and fontconfig, and one in `libavformat`, about OpenSSL, GnuTLS and SecureTransport for https or dtls.
- `libdav1d` decodes the 10-bit AV1 retrieval file, reporting `libdav1d 1.5.4`.
- The configure summary reads `License: LGPL version 2.1 or later`, and `TheLoadedFFmpegIsTheGatedMajorAndLgpl` passes on both, so `avcodec_license()` and the other libraries' strings pass the adapter's GPL and nonfree check.

**Portability.** The build runs in a pinned `ubuntu:22.04` image with packages from an Ubuntu snapshot of 2026-11-03, with no network. The newest glibc symbol version needed is `GLIBC_2.35`. Both builds ran on this machine (glibc 2.44) and in `ubuntu:24.04` (glibc 2.39, the RunPod image's Ubuntu). In the container they decoded 5 frames each of AV1 10-bit, Theora, VP9 10-bit 4:2:2 and VP8. Those frame digests equal the BtbN FFmpeg 8 build's.

**Reproducibility.** The three source archives, the script and the Dockerfile were fetched anonymously from the pinned dataset commit, and a second full build produced byte-identical archives (`SHA256SUMS` equal). Both builds ran on this machine; another machine was not tried. The source archives are `git archive` output. `build_eg2_ffmpeg.sh sources` checked `git describe` of each commit against upstream, and the version comes from a `VERSION` file holding that output, as in FFmpeg's release tarballs. The `ubuntu:22.04` image was rebuilt from cache for the second build, so this does not show that the snapshot's packages can be fetched again.

## Version strings in protected evidence

Protected files record the old builds' version, for example `n8.1.3-17-gc389840894-20261011` in `video-environment.json`. The suffix was BtbN's `--extra-version`. No gate compares that string or an archive digest: `evaluate_video.py` records `ffmpeg_version` as a metric and gates only on `supported_ffmpeg_major`, and `DecoderTests` requires only the `n8.` or `n9.` prefix. The report's new value is the plain `n8.1.3-17-gc389840894`. The protected decoded-frame digests themselves were unchanged and matched exactly.

## Other tests

Run against the same two builds, with all video assets present:

| Test project | Result |
| --- | --- |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg.Tests` | 28 of 28 passed, including every frame of the 22 fixture videos and the nine edge-case videos |
| `Starlings.Models.EmbeddingGemma2.Video.FFmpeg9.Tests` | 27 of 28 passed, 1 skipped by design (`ReleaseVideosSampleTorchcodecsSeekedFrames`: its digests are torchcodec's on FFmpeg 8) |

## The earlier run

Before this rebuild, the same gates ran on the BtbN archives (2026-11-03, at `493765b`). G1, G2 and G3 passed there too, but the script exited 1 because its G3 producer got SIGKILL after 11 of 50 videos, and G3 passed only in a separate run of the same two commands. The cause was not found; the kill came 18 to 20 s after two CI jobs started on this machine, which is correlation only. This run did not repeat the kill: the whole script ran in a unit with `MemoryMax=20G` and `MemorySwapMax=0`, with CI and other work on the machine, and peaked at 13.7 GB. That does not explain the earlier kill.

Those archives were redistributed from `Starlinghq/ffmpeg-lgpl-shared-builds`, which is private now. They linked third-party libraries whose source and revisions could not be identified, which is why this rebuild exists. Nothing in this repository uses that dataset.

## Not covered

- The `tools/ci/runpod-cuda-pod.sh` change was linted, not run. No GPU pod was used. It moves its cache to `ffmpeg8-lgpl` and `ffmpeg9-lgpl`, so a volume holding the old builds refetches, and it deletes the old cache directories.
- CUDA gates. Decoded frames do not depend on the GPU, but `video-fp32-cuda` has not run on these builds.
- torchcodec on these builds. The gates compare with its protected digests and do not run it. `libavdevice` and `libswresample` are in the archive for it, but the reference runners were not run.
- This does not produce release-set evidence for the 0.4.0 or 0.5.0 tags themselves. That needs the qualification workflow, and #277 is only confirmed once a qualification run passes this profile.
- The evaluator ran from a fresh install of this tree, as the job does (`pip install -e model-pipeline`, unlocked). `--reference-python` was the local eg2-video environment, which only fetched the media.
- The 22 fixture videos of `video-fp32-cpu` were not regenerated. CI does not regenerate them. It needs the BtbN GPL build for its encoders, which upstream deleted and nothing here hosts, so it takes a private copy (`fetch_eg2_ffmpeg.sh --with-gpl`). `make_eg2_videos.py` still names that build, because it made the videos.

## Commands

```sh
S=model-pipeline/reference/embeddinggemma2
$S/build_eg2_ffmpeg.sh sources <source dir>            # git archive of the pinned commits
$S/build_eg2_ffmpeg.sh build <source dir> <out dir> 2  # both builds, in a container without network
export STARLING_MODELS=<model cache>
tools/release-gates.sh embeddinggemma-2/video-fp32-cpu --out <out> --work <work> --reference-python <eg2-video venv>/bin/python --threads 4
```

Wall clock (shared machine, under load from CI and other work; not benchmarks): each FFmpeg build about 10 minutes with `-j2`; media 407 s; .NET stages 1,596 s; G3 producer 7,840 s; the whole script 2 h 45 min.
