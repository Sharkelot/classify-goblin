# Offline dataset conversion and provenance

Run from the repository root (no downloads or live services):

```sh
python -m jev_laya_free.trainer.conversion \
  --external-root /home/coreys/models/jev-laya-free/data/external \
  --out outputs/converted-public
```

Add `--include-local` to include Hermes, optionally with `--hermes-path FILE`.
The entire resulting build is then **local-only**. Outputs use the existing
`DecisionExample` format, consumable by `load_examples`; train, validation and
test JSONL files are accompanied by a deterministic manifest with input/output
SHA-256 hashes, license evidence hashes, source restrictions, episode counts,
label coverage and explicit unavailable/excluded statuses. No synthetic fallback
is silently introduced. Config is `configs/dataset_sources.json`.

## Source policy

- **AgentHazard:** local README declares CC-BY-4.0. Attribute AgentHazard dataset
  authors and retain the source card when redistributing derivatives. Preserve
  official train/val/test partitions. Predict edit outcome at k using only edits
  before k. True means error. Never expose final resolution, total edits or future
  prefixes as features; hidden benchmark labels are not read. This is an auxiliary
  observable outcome task, not a claim that an error implies blocked or looping.
- **ETO:** local README declares Apache-2.0. Attribute Yifan Song et al., *Expert
  Trajectories for ETO*, retain applicable notices. Only a single explicit Action
  line and preceding human/environment observation are retained. Assistant prose,
  Thought blocks and setup instructions are discarded. The auxiliary target is
  exact repeated action/observation, not a fabricated progress label. Environment
  assets retain upstream terms; this build does not redistribute images.
- **Hermes:** local-only, explicitly opted in. Only allowlisted observable event
  state is retained. `gold.loop_state` supplies progress, repeat_without_progress,
  blocked and terminal labels; stale repeat metadata is ignored. Group by board
  and task hash, falling back to the event ID's task prefix. Task titles, private
  rationales and arbitrary metadata are not copied.
- **Sentinel:** available but manifest-only. LICENSING.md distinguishes Apache
  code/format and Qwen outputs from Llama community/Gemini terms. No blanket Apache
  assumption is made. A future Qwen adapter must establish row-level model and
  upstream provenance before public use. All current rows are excluded.
- **Pi sessions:** available but manifest-only. Card declares MIT, outside this
  task's public allowlist. Tree branches, assistant thinking and embedded images
  are not converted. Local-only designation is not a license override.
- **Visual/GUI:** ScreenSpot, ScreenSpot Pro, ShowUI, GUI World, Mind2Web, AITW and
  Android Control have explicit discovery entries. Missing files are reported;
  even if added later, they remain excluded pending schema/license review.
  No pixel training examples are claimed by this converter.

Public eligibility is based on the locally inspected source card, not a legal
opinion or permission for unrelated upstream assets. Changed cards without the
expected license marker fail closed. Source cards and content hashes should be
retained alongside any derivative; no network license verification is performed.

## Causality and reproducibility

All non-official partitions use stable SHA-256 episode hashes (80/20), never row
shuffling or fallback row movement. A single episode may produce an empty split.
ETO uses domain plus game_file when present, otherwise source episode ID. AgentHazard
retains official benchmark partitions. Features contain only present/past
observations; no end-of-episode success or total length is injected. These splits
prevent episode leakage, but do not claim disjoint semantic task templates across
sources. Re-running against identical files and config produces identical outputs.

## Local artifact policy

Raw input, converted JSONL, manifests, checkpoints and reports stay under ignored
`data/`, `outputs/`, or `reports/`. Do not commit them. Local reports may reveal
source file names or counts, and redaction is best-effort rather than a guarantee
of anonymization. Reasoning-named fields and mixed reasoning text are removed;
credential-shaped keys and common tokens are scrubbed. The converter never reads
assistant free-form reasoning into training features. Fixture tests contain only
small synthetic rows embedded in Python.

```sh
PYTHONPATH=src python -m unittest discover -s tests -p 'test_conversion.py'
```
