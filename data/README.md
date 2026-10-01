# GeoLife processed caches

The `geolife_development.npz`, `geolife_validation.npz`, and
`geolife_test.npz` files are immutable processed caches produced from the
recorded GeoLife source archive. They contain the original user-ID split
candidates (23 development, 17 validation, 67 test); they are not, by
themselves, the effective analysis split.

Two distinct audits are maintained. Exact float64 GPS-coordinate fingerprints
are retained for provenance and identify literal duplicate windows. Holdout
isolation, however, follows the representation actually consumed by the
models: the complete 48-slot 8x8 state sequence. A state-path group is assigned
to the earliest split in development -> validation -> test order. All members
of that group in the owning split are retained, while later-split members are
excluded even when their GPS coordinates differ.

This rule prevents a model-input group from appearing in both construction
and holdout splits, while preserving repeated observations within its owning
split. It deliberately does **not** claim that equal coarse paths
are the same physical trip; rather, they are indistinguishable to the fitted
and replay models used in this paper. The current effective counts, dropped
later-split windows, cross-split state groups, and literal coordinate duplicate
clusters are generated in `results/geolife_decontamination.json`.

All analysis code that claims isolated GeoLife evidence must load through
`src.geolife`. Future regeneration through `src.prepare_data` preserves the
raw deterministic user-ID caches and records both coordinate and state-path
fingerprints; effective isolation remains an analysis-time operation.
