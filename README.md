# Branch-Safe Participation (BSP) — experimental reproducibility

Research artifacts for **Branch-Safe Participation for Location Privacy in Sequential Mobile Crowdsensing** (IEEE TMC manuscript).

This repository separates the paper's experimental implementation from the manuscript repository. It includes the BSP/R-BSP/DF-BSP/RDF-BSP implementations, comparator adapters (including PML-T and PRIVIC-T), experiment configurations, analysis scripts, tests, and frozen **aggregate and per-run numerical evidence**. The source manuscript is maintained at [js110/Bpaper](https://github.com/js110/Bpaper/tree/tmc).

**Provenance:** this export is based on `js110/Bpaper` at the immutable source commit `2cc94ff6e7c4696dc364392dc19c8137d0cb7e65` (2026-10-01). The original frozen event-level logs remain accessible at that source commit because the complete historical set is hundreds of megabytes. Never interpret a partial evidence export as the complete raw log archive.

## Scope and claims

The experiments study a **linkable, truthful report/silence** mobile crowdsensing channel. BSP bounds the report- and silence-branch posterior peaks given the declared model; R-BSP intersects a finite set of such constraints. DF-BSP/RDF-BSP explicitly relax the report-side cap to address the unavoidable truthful-positive disclosure floor, while retaining the silence constraint. The guarantee is *conditional on the assumed observation channel and model set*, not differential privacy or a deployment-wide guarantee.

GeoLife contributes **mobility traces only**, not task participation, availability, device timing, or production service measurements. Simulated task processes and the explored policies should not be interpreted as real-world completion performance. The recent PML-T and PRIVIC-T comparisons are *task-channel adaptations*, not claims of superiority over the original mechanisms on their native release interfaces. Certain GeoLife extensions and operating-point interpolation are exploratory/post-hoc.

## Environment and entry points

Requires Python 3.12, install `python -m pip install -r requirements-lock.txt`.

- `python -m unittest discover -s tests -v` — unit/regression tests. Tests requiring locally reconstructed GeoLife data or the archived frozen event fixture require those assets to be prepared first; the default clean checkout reports those explicitly as skipped.
- `python -m src.prepare_data` — reproducible preprocessing after independently obtaining Microsoft GeoLife (see `data/README.md`).
- `python -m src.reproduce --output results/new_reproduction` — re-run primary experiments to a **new**, non-frozen directory. This can be compute-intensive.
- `python -m src.analyze_recent --reference-run results/final --recent-run results/recent_baselines --extra-run results/recent_privic_thinning --output results/new_recent_analysis` — rebuild the recent-comparator analysis from the included per-run rows.
- `python -m src.analyze_reviewer_stress` — regenerate reviewer-stress summaries from included rows.
- `python -m src.analyze_extensions --run results/data_driven_extension` — recompute the extension summary when GeoLife-derived inputs are available.

**Do not overwrite `results/` frozen records.** Pass distinct output directories where supported or copy configs and change `output`.

## Data and reproducibility

The Microsoft GeoLife archive is not redistributed. Download it at the source URL and verify the SHA-256 in `data/manifest.json`; place it at `data/geolife.zip` and run `python -m src.prepare_data`. See `data/README.md` for the 48-slot state-path decontamination rule. The effective development, validation and test split counts were 23, 12 and 53, respectively, in the frozen manuscript analysis.

Selected aggregate/per-run evidence is retained under `results/`, including configurations, analysis summaries, bootstrap provenance, and the source rows for the paper's quantitative comparisons. The complete historical event-level `*.jsonl.gz` archive and source-specific evidence audits are retained at [the frozen Bpaper source commit](https://github.com/js110/Bpaper/tree/2cc94ff6e7c4696dc364392dc19c8137d0cb7e65/results). `docs/evidence_map.md` maps the new repository's evidence to manuscript claims, with this archival distinction made explicit.

The comparator-conformance fixture `literature/recent/privic_functions.py` is original authors' material retained under its own MIT license (see `literature/recent/PRIVIC_LICENSE`). Third-party article full texts are not included.

## Citation

When referencing the code and supporting evidence, cite the **immutable release commit of this repository**, not a moving branch. The exact export commit is recorded in the paper's reproducibility statement after migration.
