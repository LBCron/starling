# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`), none used by a package yet. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## Generation fields in the capability record

`ModelProfileCapabilities.Generation` (`ProfileGeneration`, new and Experimental API) carries tool-call mode, structured output, state reuse, caching style and max output tokens for the seven generation profiles, generated from the new manifest section `behavior.generation` (#177). The fields are informational until a release note promotes them (`docs/kickoff.md`, "Capability record fields"). No capability status changes. The validator and the contract tests hold each value to its capability row, so `untested` appears where a profile has no row: structured output on the QwenChat profiles and max output tokens on LFM3.
