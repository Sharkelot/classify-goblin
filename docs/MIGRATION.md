# classify-goblin migration

`classify-goblin` is the active product, Python package, command-line name, and
repository name. The typed-decision wire contract is unchanged: `/v1/systemone`,
`Choice`, `Score`, `Noul`, the response envelope, and loopback-only transport keep
their existing shapes.

## Local migration

| Previous surface | Current surface |
|---|---|
| repository directory `jev-laya-free` | `classify-goblin` |
| Python package `jev_laya_free` | `classify_goblin` |
| distribution `jev-laya-free` | `classify-goblin` |
| checkpoint command `goblin-jev` | `classify-goblin` |
| server command `jev-laya-free` | `classify-goblin-server` |
| checkpoint config `jev_laya_config.json` | `classify_goblin_config.json` |
| checkpoint format `jev-laya-distilbert-v1` | `classify-goblin-distilbert-v1` |

Update imports, installed entry points, and service supervisors to the current
names. The old Python package and old command names are intentionally not
shipped in the renamed release.

## Environment variables

New deployments use the `CLASSIFY_GOBLIN_*` namespace, for example:

```bash
CLASSIFY_GOBLIN_LOCAL_API_KEY=...
CLASSIFY_GOBLIN_ARTIFACT_ROOTS=/absolute/artifacts
CLASSIFY_GOBLIN_QWEN_BASE_URL=http://127.0.0.1:8080
CLASSIFY_GOBLIN_QWEN_TIMEOUT=60
```

The runtime accepts the corresponding `JEV_*` names only as a temporary
read-only fallback, and only when the new variable is absent. New names win
when both are set. Secret values are never written to logs or manifests.

Existing DistilBERT checkpoints using the old configuration filename and format
remain readable. New manifests and exported checkpoints must use the current
`classify_goblin_config.json` filename and `classify-goblin-distilbert-v1`
format.

## Benchmark compatibility

The rename is branding and packaging only. The protocol endpoint, typed question
schema, probability fields, loopback security checks, and deterministic guard
behavior remain compatible with existing Benchmark Heaven-compatible clients.
Historical evidence IDs and external artifact provenance are not used as active
package names; the repository's current evidence files use the `CG-MM-*`
internal IDs.
