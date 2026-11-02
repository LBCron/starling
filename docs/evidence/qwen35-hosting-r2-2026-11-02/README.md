# Qwen3.5-0.8B hosting revision r2: int4 variant and guidance files (issues #139 part 2, #163)

Run on 2026-11-02. This is the evidence for hosting revision r2 of `Starlinghq/qwen3.5-0.8b-ortgenai`, commit
`22477cdb8a2dc71f4c98d2e6af0a046ec0d74f45` (r1: `95a690c1774f144957cb7b76986a9126e8b1bb5f`, unchanged and still
reachable). r2 adds two things in one Hub commit:

- `ortgenai-int4/` (`genai_config.json`, `model.onnx`, `model.onnx.data`): the int4 CPU graph of the Experimental
  profile `qwen3.5-0.8b/ortgenai-int4-cpu`, rebuilt as `docs/evidence/qwen35-int4-cpu-2026-11-01/README.md` section 2
  records, uploaded unchanged.
- `tokenizer.json` and `tokenizer_config.json` in each of `ortgenai-fp32x/`, `ortgenai-fp16x/` and `ortgenai-int4/`:
  the two files ONNX Runtime GenAI's guided decoding (llguidance) opens from the model directory (#139 part 2). The
  first is a copy of `transformers-5.18.0/tokenizer.json`; the second is the checkpoint's `tokenizer_config.json` at
  `Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17`, unchanged (the digest `Qwen35Checkpoint.Lock` already
  pinned).

Every r1 file keeps its bytes; only `README.md`, `NOTICE` and `SHA256SUMS` were rewritten to describe r2 (the
`LICENSE` travels unchanged). The Hub also appended four LFS lines to `.gitattributes` (1,724 to 1,987 bytes:
`ortgenai-{fp32x,fp16x,int4}/tokenizer.json` and `ortgenai-int4/model.onnx.data`); no lock names that file.
`hosting-r2-SHA256SUMS` is the `SHA256SUMS` file of the commit.

Nothing here sets a tolerance, a gate or a golden. No `model-pipeline/protected/` file changed.

## 1. Environment

- Host: Intel Core i7-5820K (6 cores, AVX2, no AVX-512), 94 GB RAM, Linux x86-64, shared with CI runners (load
  average about 11 on 12 threads during the runs), so no timing here is a benchmark.
- GPU for the CUDA checks: **NVIDIA GeForce RTX 4060 Ti** (PCI 00000000:04:00.0, driver 615.78.08), selected with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. Not the RTX 3090 of the Stable CUDA record; no run used it.
- .NET 10.0.14, Microsoft.ML.OnnxRuntimeGenAI 0.17.1 (CPU and Cuda packages, NuGet).
- Builder: Python 3.12.14 with `model-pipeline/reference/generation/requirements.txt` (onnxruntime-genai 0.17.1,
  source commit `83de55a1…`, onnxruntime 1.30.0, torch 2.14.1+cu130, transformers 5.18.0); `int4-build-record.json`.

## 2. int4 rebuild (`int4-rebuild.log`, `int4-build-record.json`)

```sh
python model-pipeline/reference/generation/build_ortgenai_bundle.py --snapshot <snapshot> --out <bundle> --variant int4
```

The script verified the five snapshot files it reads (now including `tokenizer_config.json`) and the float32 overlay
(`295154cd…`), ran the builder with `-p int4 -e cpu --extra_options exclude_mtp=true`, and compared every output with
its pinned digest: all matched, so the uploaded int4 files are the bytes the 2026-11-01 evidence recorded.

| File | SHA-256 | Bytes |
| --- | --- | ---: |
| `ortgenai-int4/genai_config.json` | `e7e1957b2544fdad6170a9cc81cbbb1b88c8819b7016a609170ad7cbebba81ba` | 1,935 |
| `ortgenai-int4/model.onnx` | `a41f341f106a1bb5b50eb67ca824cd9c324705141d88b027de0888f140e39cf7` | 364,388 |
| `ortgenai-int4/model.onnx.data` | `709e08797497f609547f7aef1387cfb35bd0deb3f4039acf001f0680b23f9745` | 541,523,968 |
| `ortgenai-*/tokenizer.json` | `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f` | 19,989,339 |
| `ortgenai-*/tokenizer_config.json` | `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` | 16,709 |

## 3. Download from r2 (`download-drill.log`, `int4-download-test.log`)

- `download-drill.log`: a cache that already held the r1 files of fp32x and fp16x (the upgrade case). Before
  `DownloadAsync`, `OpenLocal` threw `ModelAssetMissingException` naming the two new files of each variant directory
  (and, for int4, all five). `DownloadAsync` then fetched only the missing files from r2 and verified the SHA-256 of
  every file in each lock. The Hub's tree listing gives git blob ids for the small files (`genai_config.json`,
  `tokenizer_config.json`), so this download is the check that their bytes equal the lock.
- `int4-download-test.log`: `HostingDownloadTests.Int4DownloadsFromTheHostingRepositoryAndVerifies`, anonymous
  (`UseEnvironmentToken = false`), into an empty temporary cache: the 581,893,433-byte int4 lock downloaded and
  verified; `OpenLocal` with SHA-256 verification found the variant directory with the graph and both guidance files.

## 4. Guided decoding from the downloaded files

All runs load the variant directory from the r2 cache of section 3 (`STARLING_MODELS`), with no temporary directory or
added file (the test helper `Qwen35GuidedDirectory` that built one is removed). Greedy; outputs are checked against the
constraint, not a reference.

| Log | Runs | Result |
| --- | --- | --- |
| `guidance-cpu-tests.log` | `GuidanceCpuModelTests` (fp32x, CPU: regex, JSON schema, required tool grammar with and without fast-forward tokens, fast-forward text equality) and `AssetAndCapabilityTests` | 22 passed, 0 skipped |
| `guided-chat-client-tests.log` | `GuidedRealModelTests` (the IChatClient: `ToolMode` RequireAny, JSON `ResponseFormat`) and `GuidedDecodingTests` (request mapping, including RequireSpecific, on a scripted backend) | 19 passed, 0 skipped |
| `guided-chat-client-outputs.log` | `GuidedRealModelTests` rerun after review with each constrained output written to the log, plus a RequireSpecific case on the real model | 3 passed, 0 skipped |
| `guidance-cuda-tests.log` | `Fp32xCudaGuidanceTests`, `Fp16xCudaGuidanceTests` (new: regex, JSON schema, required tool grammar with and without fast-forward), RTX 4060 Ti | 10 passed, 0 skipped |

The IChatClient outputs in `guided-chat-client-outputs.log` (fp32x, CPU):

| Request | Finish | Output |
| --- | --- | --- |
| "What's the weather in Porto?", `RequireAny` over `get_weather`, `get_time` | `tool_calls` | `get_weather({"city":"Porto"})` |
| "What time is it in Porto?", `RequireSpecific("get_weather")` over the same tools | `tool_calls` | `get_weather({"city":"Porto"})` |
| "What is 6 times 7? Reply as JSON.", JSON schema `{"answer": integer}` | `stop` | `{"answer": 42}` |

The IChatClient checks ran on CPU only; the CUDA runs use the typed API, so the adapter itself has no CUDA run.

## 5. What changed in status

- `structured_output` (package table and the fp32x-cpu, fp32x-cuda and fp16x-cuda profiles), `structured_output.cuda`,
  `extensions_ai.tool_mode.required`, `extensions_ai.structured_output`: Unsupported to **Experimental**. Not Supported:
  the checks are a handful of prompts per mode on one machine, with no latency qualification and, for CUDA, no run on
  the qualification GPU.
- `qwen3.5-0.8b/ortgenai-int4-cpu`: still Experimental. `asset_acquisition.download.int4` Unsupported to Experimental;
  `rights.redistribution` blocked to download; `structured_output` Unsupported to **untested** (the guidance files are
  hosted, but no guided run used the int4 graph).

## 6. Not done here

- No tolerances or task-quality floor for int4 (#163 step 2, maintainer's protected change).
- No guided run on int4, no guided run on the RTX 3090, no latency measurement of guided decoding.
