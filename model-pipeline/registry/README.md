# Model registry

One directory per model family, one subdirectory per profile, each holding a `manifest.yaml` that validates against [`../schemas/model-manifest.schema.json`](../schemas/model-manifest.schema.json).

```
registry/
  <family>/
    <profile>/
      manifest.yaml
```

The public catalog, [`docs/catalog/`](../../docs/catalog/README.md), is generated from these manifests by `model-pipeline catalog` (P-021); there is no second hand-maintained support list. Every capability carries an explicit status, and a missing value never implies support.

Validate with `model-pipeline validate registry`. After changing a manifest, regenerate the catalog with `model-pipeline catalog`; CI fails when it is stale.

A profile whose `behavior.tasks` has `generation` also needs `behavior.generation` (tool-call mode, structured output, state reuse, caching style, max output tokens). The validator holds each value to its row in `capabilities`: `untested` when the row is untested or absent, `none` when it is unsupported. `max_output_tokens` is the exception: it has no `none`, so an unsupported row gives `untested` too. An `untested` value blocks Stable. The `model-manifest.schema.json` descriptions give the rest.

Quote every digest and revision in YAML. An unquoted all-digit SHA parses as a number and fails validation.
