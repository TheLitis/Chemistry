# V3 implementation - approved R03-R05 continuation, 2026-09-16

Start from a9c3008. Preserve the existing R03 confidence implementation and R04
COCONUT research in main; do not overwrite them with the older standalone ZIP.
The live ChemistryPC preflight confirmed COMPLETE submissions 56268669 (0.147)
and 56269464 (0.112). The original model remains the best official public result.
Two attempts already consume the agreed rolling-24h cap. This task performs NO
Kaggle writes or additional submissions.

## Sealed design and comparison

Use a train-only new cache. Preserve the original 4104 input coordinates and add
4096 fractional-mass moments (8200 total). These moments do not losslessly retain
every peak. Targets: tautomer-canonical, nonchiral Morgan radius 2 / 2048,
Morgan radius 3 / 4096, and atom-pair / 8192 bits. RDKit 2026.03.3.

Warm-start only from the fixed 90%-training holdout checkpoint. Compare frozen
v1; old-input Morgan control with 10 extra epochs; high-resolution Morgan with
the same 10 epochs; high-resolution three-target model with the same 10 epochs.
Keep hidden=512, batch=256, learning_rate=.0005, seed=26091605. Multihead ranking
compares only fixed weights (1,0,0), (.5,.25,.25), (1,1,1). Select on 800 fresh
calibration keys, then evaluate once on 1600 other unused molecular keys.
Exclude every previously evaluated key found in research artifacts. Model
fitting excludes ALL old validation keys, not just the selected 2400.

Seal exact keys, input hashes, variants, settings and holdout-checkpoint hash
before training. Save selection before the audit. Primary measurement is
all-spectrum aggregated, inclusive-catalog MRR@25. Instrument/polarity/NP
subgroups are diagnostic and cannot select a different model. Require a positive
molecule-bootstrap lower bound vs frozen R01 before building a candidate bundle.
Then refit the selected architecture on all training rows using the old all-data
model; never evaluate that final refit on the previous held-out examples.

Inference supports neural, fixed-R03 confidence (.95/.05), and legacy mixture
modes. No routing gain is claimed from a neural-only audit. Group ALL current
test spectra by actual IDs, whitelist feature columns, ignore test labels,
and never hardcode the visible sample IDs. Save numeric-only model files and
verified hashes. The notebook installs dependencies offline and writes only its
submission/report to the output directory; temporary code/dependencies are
removed after use. Existing champion files are never overwritten.

## Verification and execution

Local TDD: 8 new contracts failed on missing modules before implementation.
After implementation, 156 tests passed including real-Parquet cache creation,
all-spectrum inference, leading-zero IDs, stale-template handling, tampering,
CPU training/export parity, and generated notebook compilation. These are
software checks, not molecular accuracy. The full ChemistryPC execution must
provide the actual training, validation and delivery evidence.

## Acceptance boundary

A completed implementation is not score 1. This remains a catalog model: absent
graphs need a separately evaluated external-catalog or de novo workflow. Oracle
descriptors are not predicted descriptors. Local selection does not make the
candidate a new official champion. The best confirmed public score remains
0.147 until a new server-scored result actually exceeds it.
