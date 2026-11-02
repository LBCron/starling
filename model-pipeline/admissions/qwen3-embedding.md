# Admission: Qwen3 Embedding series 0.6B (Qwen team, Alibaba Cloud)

Model requests #108 (`Qwen/Qwen3-Embedding-0.6B`) and #100 (`Qwen/Qwen3-Reranker-0.6B`), design #144. Plan section 7,
pipeline stage 1. Facts were read from the Hub at the pinned revisions; every pinned file was downloaded and hashed.

| Field | Value |
| --- | --- |
| Candidate | `Qwen/Qwen3-Embedding-0.6B` (text embeddings) and `Qwen/Qwen3-Reranker-0.6B` (reranking), through Starling's fp32 ONNX export hosted at `Starlinghq/qwen3-embedding-0.6b-onnx` and `Starlinghq/qwen3-reranker-0.6b-onnx` |
| Code revision | Export scripts `model-pipeline/reference/qwen3-embedding/` (`export_qwen3_onnx.py`, `release_qwen3_onnx.py`) at `c231221ae1582243d6d9108bd401e6f602e737cb`; the hosted NOTICE and README text comes from this change, not from `c231221a` (graph bytes are unaffected: the export script is identical and the release script differs only in its docstring). No upstream code is run: the reference is transformers' own `Qwen3Model` |
| Weight revision | Embedding `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`; reranker `e61197ed45024b0ed8a2d74b80b4d909f1255473`. Hosting: `0585005027acc7232133808914cd8bc2a4bd1ea8` and `0f951df2baec8e9ac7f884f349d2820a34aa85e5` |
| Requested by / demand evidence | #108, #100; a decoder-based multilingual embedder and reranker pair at 0.6B, the sizes the card's MTEB results lead with |
| Proposed owner | Mark |
| Proposed maturity at first release | Experimental |
| Decision | Proposed in this change, for the maintainer: admit `qwen3-embedding-0.6b/onnx-fp32-cpu` at Experimental (registry manifest added). Hold the reranker profile: its onboarding run failed the predeclared hidden-state bound (see Open risks). The package stays out of the release set until both profiles of the package are decided. |
| Decided by and date | Pending: merging the PR that adds this record is the decision. |

## Rights (hard blockers)

| Item | License or terms | Redistribution |
| --- | --- | --- |
| Code | Starling's own (MIT OR Apache-2.0); no upstream source copied. The reranker template text is attributed in the package's `THIRD-PARTY-NOTICES.md`. | Yes |
| Weights | Apache-2.0 (model card front matter of both repositories at the pinned revisions) | Yes: the derived fp32 graphs are hosted publicly with the Apache License 2.0 and a NOTICE stating the modifications (section 4) |
| Tokenizer assets | Apache-2.0 (`tokenizer.json` of each repository, unchanged) | Yes, hosted beside the graphs |
| Evaluation data | Starling's tokenizer corpus (`reference/corpus/tokenizer-corpus-v1.jsonl`) and the model card's examples | n/a |

## Score

| Criterion | Weight | Score (0 to 5) | Notes |
| --- | --- | --- | --- |
| User demand | 25% | 4 | Two model requests; strong multilingual retrieval at 0.6B |
| Backend feasibility | 20% | 4 | One ONNX graph per model on ONNX Runtime CPU; eager attention costs memory on long inputs |
| Reusable architecture coverage | 20% | 4 | First non-autoregressive Qwen3 decoder route; covers both embedding and reranking |
| Access and licensing clarity | 15% | 5 | Apache-2.0 weights and tokenizer, no gating |
| Testability | 10% | 4 | PyTorch reference, sentence-transformers cross-checks, the card's published cosines |
| Maintenance burden (5 = low) | 10% | 3 | 2.4 GB graphs per model, Starling-built and hosted |
| **Weighted total** | | **4.05** | |

## Execution route

ONNX Runtime 1.30.0, CPU provider, fp32. Starling exports `Qwen3Model` with the PyTorch dynamo exporter (opset 20) as
one graph per model that runs the whole input once (no KV cache, no `position_ids` input); the reranker graph adds a
two-row head (`no`, `yes`) from the tied embeddings. Hidden states and the two logits are available; logits over the
full vocabulary are not (the graphs cannot generate). Onboarding route `qwen3-embedding-onnx-cpu-v1`.

## Tokenizer profile

Byte-level BPE from `Starlings.Tokenizers` 0.3.4: `qwen3-bpe-embedding` (the `Sequence[ByteLevel, TemplateProcessing]`
post-processor that appends `<|endoftext|>`) for the embedding model, `qwen3-bpe-chat` for the reranker (the same file
as Qwen/Qwen3-0.6B's). Token ids equal the reference tokenizer's on every onboarding case.

## Feature contract

| Upstream feature | In scope | Evidence source |
| --- | --- | --- |
| Last-token pooling, L2, 1024 dimensions | Yes, experimental | `docs/evidence/qwen3-embedding-onboarding-2026-11-02/` (passed) |
| `query` prompt and custom instructions | Yes, experimental | same |
| Reranker score `logit[yes] - logit[no]`, template, custom instructions | Yes, experimental in the package; profile held | same (scores within 2.4e-5; hidden-state bound failed) |
| Up to 8,192 tokens | Yes, experimental | same (longest evaluated input 6,649 tokens) |
| Matryoshka dimensions | No (rejected) | n/a |
| CUDA, fp16/bf16, 4B and 8B | No | n/a |

## Hardware and compute estimate

About 2.4 GB of fp32 weights per model. Eager attention: about `rows x 16 x width^2` float32 per layer, so one
8,192-token row needs about 4.3 GB of attention scores per layer. The onboarding runs fit inside a 20 GB memory limit on
two threads (about 15 minutes each, export included).

## Open risks

- **Reranker hidden-state drift.** The reranker's final hidden state differs from PyTorch by 3.65e-3 at 6,649 tokens and
  2.12e-3 at 2,058 tokens, above the predeclared 2e-3 (route cap 5e-3), while the score stays within 2.7e-5. The bound
  was not changed after the result; the maintainer decides whether the reranker candidate takes a bound within the cap
  or the hidden state is checked only for embeddings. Either way this is a route-code change, not only a candidate
  edit: `validate_candidate_behavior` requires `evaluation.policy` to equal the shared `POLICY` constant in
  `onboarding_qwen3_embedding.py`, so a reranker-specific bound needs a per-task policy there, with a test.
- References are proposed and no gate has run.
- Inputs of the full 8,192 tokens were not evaluated: long input is untested above 6,572 (embedding) and 6,649
  (reranker) tokens.
- `Qwen3EmbeddingModelAssets.DownloadAsync` has not been run end to end against the hosting repositories; the
  evidence is the lock match (size and SHA-256 of every hosted file) and HEAD checks.
