# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`), none used by a package yet. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## Qwen3.5 int4 guided decoding

`structured_output` on the Experimental profile `qwen3.5-0.8b/ortgenai-int4-cpu` moves from untested to experimental. A regex, a JSON schema and the required tool grammar were followed on int4 on CPU, through the typed API and the `IChatClient`, on the hosted r2 files ([evidence](../evidence/qwen35-int4-guided-2026-11-04/README.md)). No code path changed. `tool_calling` and `extensions_ai` stay untested on int4, and the profile still has no tolerances or task-quality floor.
