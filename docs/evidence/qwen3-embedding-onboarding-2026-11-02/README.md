# Qwen3 Embedding series 0.6B on ONNX Runtime, CPU fp32: onboarding evidence (2026-11-02)

Development evidence for `Starlings.Models.Qwen3Embedding` (design #144, model requests #108 and #100), from the
`qwen3-embedding-onnx-cpu-v1` onboarding route run for real on 2026-11-02 on Linux x64 (Intel Core i7-5820K, 12 logical
CPUs, 94 GB RAM shared with other work), CPU only, two threads, each run inside a 20 GB memory limit
(`systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0`). It is not release qualification: no gate ran, no
fixture is protected, and nothing here is binding.

| Candidate | Recipe | Result | Checks |
| --- | --- | --- | --- |
| `qwen3-embedding-0.6b/onnx-fp32-cpu` | `recipes/qwen3-embedding-0.6b-cpu.yaml` | **passed** (exit 0, 811 s) | 558, 0 failed |
| `qwen3-reranker-0.6b/onnx-fp32-cpu` | `recipes/qwen3-reranker-0.6b-cpu.yaml` | **failed** at `evaluate` (exit 1, 900 s): the hidden-state bound | 234, 3 failed |

## Inputs

- Upstream: `Qwen/Qwen3-Embedding-0.6B` at `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` and `Qwen/Qwen3-Reranker-0.6B` at
  `e61197ed45024b0ed8a2d74b80b4d909f1255473`; every file the candidates pin was checked by size and SHA-256 (`assets`
  stage, `*/source-lock.json`).
- Hosted graphs: `Starlinghq/qwen3-embedding-0.6b-onnx` at `0585005027acc7232133808914cd8bc2a4bd1ea8` and
  `Starlinghq/qwen3-reranker-0.6b-onnx` at `0f951df2baec8e9ac7f884f349d2820a34aa85e5` (public). The `build` stage re-ran the
  export recipe from the upstream snapshot and the released `model.onnx` and `model.onnx.data` were byte-identical to
  the hosted files (`*/build.json`, `hosted_bytes_match: true`), so the evidence is about the bytes
  `Qwen3EmbeddingModelAssets.DownloadAsync` fetches. The Hub's LFS digests of both files equal the locks.
- Reference environment: `model-pipeline/reference/qwen3-embedding/requirements.txt` (hash-locked; Python 3.11, torch
  2.14.1+cpu, transformers 5.18.0, sentence-transformers 6.1.0, onnxruntime 1.30.0). Corpus:
  `model-pipeline/reference/corpus/tokenizer-corpus-v1.jsonl` (49 texts: several scripts, emoji, special-token text,
  whitespace edge cases, the empty string) plus the model card's examples and long inputs built from the corpus.
- .NET: the library and `ai-dotnet/tools/Starlings.Qwen3Embedding.Fidelity.Tool` from this change (the pipeline records
  the working tree as `c231221a-dirty`, the base commit plus this change), ONNX Runtime 1.30.0, `Starlings.Tokenizers`
  0.3.4.
- After these runs, review fixes changed only run plumbing, not computed values: the reference stage now receives
  `--threads` (it defaulted to 2, the count used here), the `dotnet` stage fingerprint now also covers
  `Starlings.Models.Abstractions` and the shared batching and tool sources, and the fidelity tool rejects a bad
  `--threads` value or a candidate that omits a hosted graph file. A doc comment in `Qwen3EmbeddingCapabilities.cs` and
  two regenerated catalog blocks also changed after the runs, documentation only:
  `ai-dotnet/src/Starlings.Models.Qwen3Embedding/README.md` and `ai-dotnet/src/Starlings.Acquisition/README.md`. The
  recorded stage fingerprints in `*/run.json` are from before those changes, and `embedding/review.json` records the
  candidate commit as `c231221a…-dirty` (the base commit plus this change's uncommitted working tree), not a commit of
  this branch.

## Embedding: passed

| Comparison | Largest difference | Bound |
| --- | ---: | ---: |
| .NET from text vs PyTorch, L2-normalized embedding | 6.0e-7 | 1e-4 |
| .NET from reference ids vs PyTorch, embedding | 6.0e-7 | 1e-4 |
| .NET, embedding norm vs 1 | 5.7e-9 | 1e-5 |
| .NET, the card's four query-document cosines vs the published values | 3.6e-7 | 1e-3 |
| Python ONNX Runtime vs PyTorch, embedding | 6.0e-7 | 1e-4 |
| Python ONNX Runtime vs PyTorch, hidden state at every valid token | 1.1e-3 | 2e-3 |
| Python ONNX Runtime, card cosines vs the published values | 3.9e-7 | 1e-3 |
| sentence-transformers `encode` vs the graph, card texts | 2.6e-7 | 1e-4 |

Exact checks, 250 of 250: the .NET token ids equal the reference tokenizer's on every case (62 cases: the card's two
queries and two documents, the 49 corpus texts with prompt `none`, eight with prompt `query`, one long input), every
input ends with `<|endoftext|>`, and the released graph is the hosted one.

## Reranker: failed on the hidden-state bound only

| Comparison | Largest difference | Bound |
| --- | ---: | ---: |
| .NET from text vs PyTorch, score `logit[yes] - logit[no]` | 2.4e-5 | 1e-3 |
| .NET from reference ids vs PyTorch, score | 2.4e-5 | 1e-3 |
| .NET public `Score` API, card pairs | 1.6e-5 | 1e-3 |
| Python ONNX Runtime vs PyTorch, score | 2.1e-5 | 1e-3 |
| sentence-transformers `CrossEncoder.predict` vs the graph, card pairs | 1.2e-5 | 1e-3 |
| Python ONNX Runtime vs PyTorch, hidden state at every valid token | **3.7e-3** | 2e-3 |

Exact checks, 60 of 60: token ids of every case (56 cases: the card's four pairs, one pair per corpus text, two custom
instructions, one long pair), the template's prefix and suffix ids, the pinned `chat_template.jinja` rendering, and
the hosted bytes.

The three failed checks are the final hidden state of two long inputs in `verify_qwen3_onnx.py`: `long-8000` (6,649
tokens) 3.65e-3 and `corpus-long-mixed` (2,058 tokens) 2.12e-3; every other case is at most 8.4e-4. The score, the
reranker's only output, stays within 2.7e-5 on every case including those two, so this is float32 drift over 28 layers
on long inputs, not a wrong graph. The candidate's predeclared bound (2e-3, the provisional value of #144; the route cap
is 5e-3) was **not** changed after the result: whether the reranker candidate takes a different hidden-state bound
within the cap, or the hidden state is checked only for the embedding task, is the maintainer's decision. Until then
the reranker profile has no passed development parity and no registry manifest.

## Not covered

- Inputs of the full 8,192 tokens: the longest evaluated inputs have 6,572 (embedding) and 6,649 (reranker) tokens.
- CUDA, fp16/bf16, Matryoshka dimensions, the 4B and 8B checkpoints: unsupported by the package.
- Windows and macOS; package-only consumption (`--distribution`); gates G0 to G7.

## Files

| File | Contents |
| --- | --- |
| `embedding/report.json`, `report.md` | Every check of the embedding evaluator with its value and bound |
| `embedding/review.json`, `review.md` | The review stage's record (development parity passed; qualification unapproved) |
| `reranker/report.json`, `report.md` | Every check of the reranker evaluator, including the three failures |
| `*/verify.json` | `verify_qwen3_onnx.py` output: per-case differences of ONNX Runtime against PyTorch |
| `*/build.json` | The released graph's sizes and digests against the hosted ones |
| `*/source-lock.json` | The upstream files verified by the `assets` stage |
| `*/run.json` | The pipeline run record (stage inputs, digests, timings; local paths replaced by `<repo>`, `<models>`) |

Commands (from the repository root; paths shortened):

```sh
systemd-run --user --scope -p MemoryMax=20G -p MemorySwapMax=0 \
  .venv/bin/model-pipeline run model-pipeline/recipes/qwen3-embedding-0.6b-cpu.yaml \
    --snapshot <Qwen/Qwen3-Embedding-0.6B snapshot at 97b0c614> \
    --reference-python .models/venv-qwen3-embedding/bin/python \
    --out artifacts/onboarding/qwen3-embedding-0.6b --threads 2 --timeout 3600
# the same with recipes/qwen3-reranker-0.6b-cpu.yaml and the Qwen/Qwen3-Reranker-0.6B snapshot at e61197ed
```
