# ai-dotnet: unreleased changes

Changes merged to `main` after the last release record, collected here for the next one. The release-candidate change moves them into `ai-<version>.md` and empties this file. Nothing here is published until then.

## Tokenizer

Every package that tokenizes now pins `Starlings.Tokenizers [0.3.5]` (0.3.4 in `0.5.0`), and every manifest's `processing.tokenizer.version` moved with the pin. `0.3.5` adds the `WhitespaceSplit` pre-tokenizer, the o200k `Split` regex and three experimental profiles (`phi4-bpe-chat`, `xlmr-unigram-paraphrase-multilingual`, `modernbert-bpe-gte-reranker`), none used by a package yet. Outputs of existing profiles are unchanged: the conformance fixtures they have pass unchanged, the profiles without fixtures reach no code that changes their output, and no model was rerun (`docs/support-policy.md`, "Tokenizer bumps", option B). A tokenizer with an `rstrip` added token and, in the same matching pass, an added token that starts with whitespace now fails at load instead of diverging from upstream; no pinned asset has that combination.

## Generation fields in the capability record

`ModelProfileCapabilities.Generation` (`ProfileGeneration`, new and Experimental API) carries tool-call mode, structured output, state reuse, caching style and max output tokens for the seven generation profiles, generated from the new manifest section `behavior.generation` (#177). The fields are informational until a release note promotes them (`docs/kickoff.md`, "Capability record fields"). No capability status changes. The validator and the contract tests hold each value to its capability row, so `untested` appears where a row is untested or absent: tool calls and structured output on Qwen3.5 int4, structured output on the QwenChat profiles and max output tokens on LFM3. An `untested` value also blocks Stable, so these appear as `generation.<field>` in the catalog's Stable readiness column; those profiles are all Experimental, and no Stable profile has one.

## Starlings.Models.QwenChat.OnnxGenAI: hosting revision r2 (#278)

`QwenChatProfile.Qwen3` and `QwenChatProfile.Qwen25` pin new `model.onnx` digests, and `QwenChatModelAssets.Qwen3HostingRevision` and `Qwen25HostingRevision` move to revision `r2` of `Starlinghq/qwen3-0.6b-ortgenai` (`00278275`) and `Starlinghq/qwen2.5-0.5b-instruct-ortgenai` (`265459af`). Only `model.onnx` changes, and only in its `producer_version` header: `0.17.1+83de55a`, the builder's own commit, instead of `0.17.1+1bada25`, the Git HEAD of the checkout that built r1. The weights and the other five bundle files are the same bytes, the sizes are the same, and the maturity of both profiles stays Experimental.

`build_bundle.py` hides Git from the builder since #283, so a local build with the hash-locked builder gives the pinned digests again: two builds per profile on 2026-11-04, one from an environment inside a Git work tree and one from outside any, on one machine. With the new pins the onboarding `dotnet` stage, which failed with exit 70 in qualification, passes for both profiles, and development parity passes on all 14 corpus rows. Evidence: `docs/evidence/qwenchat-hosting-r2-2026-11-04/README.md`.

**Upgrading:** r1 stays hosted at its commits (`c24cd59c`, `6b6957c8`), so packages up to 0.5.0 keep working unchanged. After upgrading the package, a cache acquired from r1 fails `QwenChatModelAssets.OpenLocal` with `ModelAssetIntegrityException` on `model.onnx`, at the default SHA-256 verification; call `DownloadAsync` once, which replaces that one 150 to 250 KB file. A cache opened with `CacheVerification` set to size only passes `OpenLocal`, because the size is unchanged, and `QwenChatModel.Load` then refuses the bundle, so run `DownloadAsync` with SHA-256 verification. An r1 bundle kept in a directory of its own is refused by `Load`.

Both revisions use one cache path, because the path is keyed on the upstream revision. So an app on 0.5.0 and a newer app that share a cache directory invalidate each other, and rolling back hits the same thing: each `DownloadAsync` replaces `model.onnx` with its own revision, and the other package then fails as above until it downloads again. Give the two apps separate cache directories.

The same stamp exposure in the Qwen3.5 int4 graph (`c98b9fc`) is not part of this change.
