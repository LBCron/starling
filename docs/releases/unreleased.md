# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`); of these, only `modernbert-bpe-gte-reranker` is used by a package here, by the BERT route below. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## BERT route: gte-reranker-modernbert-base, Experimental (#101, #102)

- `Starlings.Models.Bert` adds a built-in profile, `BertModelProfile.BuiltIn("gte-reranker-modernbert-base")` (`gte-reranker-modernbert-base/onnx-fp32-cpu`): `Alibaba-NLP/gte-reranker-modernbert-base` at `f7481e60`, Apache-2.0, the first ModernBERT model. Two-input graph as published upstream, BPE tokenizer profile `modernbert-bpe-gte-reranker`, raw logits through `IReranker`, 8,192 tokens. Onboarding evidence stops at 1,468 tokens. Experimental; the references are proposed and no gate has run.
- `BertSession` replaces the truncation and Fixed padding of 8000 that this `tokenizer.json` declares with no padding and the profile's limit, as it already did for every profile; `ModernBertTokenizerTests` now pins that.
- `model-pipeline inspect` suggests `bert-onnx-cpu-v1` for a `modernbert` config with a BPE tokenizer, and no longer reports the GPT-2 split or `lstrip` added tokens as unsupported, which `Starlings.Tokenizers` has implemented since 0.3.2 and 0.3.1.
