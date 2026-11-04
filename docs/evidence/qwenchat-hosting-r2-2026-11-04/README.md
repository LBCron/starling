# QwenChat hosting revision r2: model.onnx with the builder's own stamp (issue #278)

Run on 2026-11-04. This is the evidence for hosting revision r2 of the two QwenChat bundles:

| Profile | Repository | r2 commit | r1 commit (unchanged) |
| --- | --- | --- | --- |
| `qwen3-0.6b/ortgenai-fp32-cpu` | `Starlinghq/qwen3-0.6b-ortgenai` | `002782754045f2f8f15acf238d888455642181ab` | `c24cd59cad1444863ddfdb372736da610a09c9c4` |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` | `265459af18b9dac105d3c7a528d22cd926cfe8f7` | `6b6957c8305ef950450da2c0c846bce6fac3e8a0` |

The qualification jobs `experimental qwen3` and `experimental qwen25` failed at the onboarding `dotnet` stage with exit 70, because the pinned `model.onnx` carried `producer_version` `0.17.1+1bada25`, the Git HEAD of the checkout that built r1. PR #283 made the build stamp the wheel's commit (`83de55a`) on any host. r2 moves the pins to that stamp.

Each r2 commit changes `model.onnx`, and rewrites `README.md`, `NOTICE` and `SHA256SUMS` to describe r2. The five other bundle files and `LICENSE` carry over untouched. Licence terms are unchanged (Apache-2.0, the 2026-10-19 hosting decision in `model-pipeline/admissions/qwen-chat.md`).

Nothing here sets a tolerance, a gate or a golden. No `model-pipeline/protected/` or `tolerances.*` file changed, and no old digest was found in those paths. Both profiles stay Experimental, and their references stay proposed.

## 1. Environment

- Host: the shared machine of `docs/evidence/qwen35-hosting-r2-2026-11-02/README.md`, Linux x86-64. Nothing here is a benchmark.
- .NET SDK 10.0.100 (`global.json`), Python 3.12, `model-pipeline/reference/generation/requirements.txt` installed with `--require-hashes` (onnxruntime-genai 0.17.1, onnx 1.23.1, onnxruntime 1.30.0, torch 2.14.1+cu130, transformers 5.18.0).
- Inference on CPU. No run used the RTX 3090.

## 2. Two builds per profile

| Build | Environment | Command |
| --- | --- | --- |
| First | inside a Git work tree (`.models/venv-generation` of the Starling checkout) | `build_bundle.py` directly |
| Second | outside any work tree, new venv, as CI uses | the `bundle` stage of `model-pipeline run` |

Both builds of each profile gave the same six digests (`qwen3/bundle-lock.json`, `qwen25/bundle-lock.json` are the second build), and `build_bundle.py` accepted the stamp `0.17.1+83de55a`. So the digest no longer depends on where the environment lives, on this machine. Another builder wheel or platform was not tested.

| File | Old SHA-256 (r1) | New SHA-256 (r2) | Bytes |
| --- | --- | --- | ---: |
| Qwen3 `model.onnx` | `b9fdea5216f1642d7926c64f77875c43a43b320531db841d8b3984b79a754ae8` | `d8c66e98638c00c1cb6a78af24dec7283a6c402b0d639db5ae98c77c6d917e3f` | 248,657 |
| Qwen2.5 `model.onnx` | `716a2db391784191cc1323417e1ff5bec682673b7105b8ddc4011e7a9f2c4a88` | `938a56f1c87fd63ca29dfcba0c45642dc93f7913f12ab70715e9db6e3ea410e7` | 147,574 |

These are the digests the PR #283 review predicted by replacing only the stamp in the hosted r1 graphs. The other five files of each bundle equal the earlier pins, and the sizes are unchanged.

## 3. Onboarding runs, before upload

```sh
model-pipeline run model-pipeline/recipes/qwen3-cpu.yaml --snapshot <snapshot> --reference-python <venv>/bin/python --out <run> --threads 2 --timeout 3600
```

The same command ran for `qwen25-cpu.yaml`, with the flags the qualification job uses. The first runs used the new digests with the hosting revisions still at r1, and both exited 0 through all eight stages, so nothing was uploaded before the `dotnet` stage passed.

After the upload, both were resumed with `--resume` at commit `eea52b1`, which has the final pins (`qwen3/onboarding-resume.log`, `qwen25/onboarding-resume.log`). The earlier stages were reused after input and output verification, `dotnet`, `evaluate` and `review` ran again, and both runs exited 0. `review.json` of each profile names that commit.

| Check | Qwen3 | Qwen2.5 |
| --- | --- | --- |
| `dotnet` stage | passed | passed |
| `evaluate` status, rows | passed, 14 | passed, 14 |
| Largest absolute logit difference, either backend | 1.278e-4 | 7.015e-5 |

`report.json` of both profiles is equal, field by field, to `docs/evidence/qwenchat-onboarding-2026-10-18/`, so the stamp change moved no output. `dotnet-command.log` and `dotnet-environment.json` are from the resumed stage. The review output says what it says at every run: references stay proposed, and release qualification is unapproved.

The runs were local. The qualification workflow itself has not run on this change.

## 4. Fresh download

`fresh-download.log`: anonymous downloads of all six files of both revisions of both repositories into empty directories, with no token, 24 lines, every SHA-256 as expected. r2 has the new `model.onnx` digests. `<profile>/hosting-r2-SHA256SUMS` is the hosted file at the r2 commit.

- `tools/check-provenance.py` on both manifests at r2: 13 of 13 lock files served anonymously with the pinned SHA-256 and size, `PROVENANCE PASS`.
- `HostingDownloadTests.Qwen25DownloadsFromTheHostingRepositoryAndVerifies`, with `STARLING_NETWORK_TESTS=1`, anonymous, into an empty cache: passed (`qwenchat-tests.log`). This test exists for Qwen2.5 only; Qwen3 is covered by `fresh-download.log` and section 6.
- The `producer_version` of the downloaded r2 graphs is `0.17.1+83de55a`; of the r1 graphs, `0.17.1+1bada25`.

## 5. r1 stays available

Packages up to 0.5.0 pin r1 by commit, so they keep working.

- Both r1 commits still resolve, and all six files of each downloaded anonymously with their r1 digests after the r2 commits existed (`fresh-download.log`, lines marked `r1`).
- Tags `r1` and `r2` resolve to the commits in the table above. Each repository history is `initial commit`, `Starling build r1`, `Starling build r2`; nothing was rewritten or force-pushed, and `main` moved to r2 by a normal commit.
- `list_repo_refs` shows the tag objects' own ids (`03f2a36c`, `8f66020d` for r1), not the commits. The packages pin the commits.

## 6. Upgrade from an r1 cache

`upgrade-drill.log`, from the temporary test `upgrade-drill-test.cs.txt` (kept here, not in the test project). It seeds a cache as an r1 download left it, for both profiles, and upgrades it.

| Step | Result, both profiles |
| --- | --- |
| `OpenLocal` at the default SHA-256 verification | `ModelAssetIntegrityException` on `model.onnx` |
| `OpenLocal` with `CacheVerification.Size` | passes, because the size is unchanged |
| `QwenChatModel.Load` on that directory | `InvalidDataException`: `Bundle asset mismatch: model.onnx.` |
| `DownloadAsync` | replaces `model.onnx`, reuses the five other files |
| `OpenLocal` afterwards | passes |

## 7. Other checks

`qwenchat-tests.log`: the QwenChat test project, 336 tests, 335 passed and 1 skipped, in a run that included the network test and the temporary drill. `BundleLockTests.PinsEqualTheRegistryManifest` passed for both profiles, and `HostingDownloadTests.EveryPinnedFileIsHostedAtThePinnedCommit` passed for both.

## 8. Not done

- The clean-consumer drill from published packages did not run, because no package with these pins exists yet.
- The Qwen3.5 int4 graph keeps its own stamp (`c98b9fc`). `build_ortgenai_bundle.py` has the same exposure, and that fix is out of scope here.
- A build on a second machine or with another builder wheel.
