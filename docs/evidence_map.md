# Evidence map and archival boundary

This repository holds the numerical inputs **actually used** for the BSP/R-BSP/DF-BSP/RDF-BSP IEEE TMC manuscript. The immutable historical source is [Bpaper at `2cc94ff6e7c4696dc364392dc19c8137d0cb7e65`](https://github.com/js110/Bpaper/tree/2cc94ff6e7c4696dc364392dc19c8137d0cb7e65). A separate machine-checkable list of byte-identical transferred files is in [`docs/source_manifest.json`](source_manifest.json).

| Manuscript evidence | Exported numerical inputs | Processing / notes |
|---|---|---|
| Primary BSP synthetic/GeoLife, recent comparator baselines | `results/final/rows.csv`, `results/kl_baseline/rows.csv`, `results/recent_baselines/rows.csv`, `results/recent_privic_thinning/rows.csv`, `results/combined/rows.csv` | `src/analyze.py` and `src/analyze_recent.py`; matched results in `results/recent_comparison/{summary.csv,matched_utility.json,provenance.json}`. |
| Finite-model robust stress | `results/robust_informed/rows.csv` and `analysis/{summary.csv,provenance.json}` | `src/analyze_robust.py`. |
| Outside-set robustness boundary | `results/robust_boundary/rows.csv` and `analysis/{summary.csv,provenance.json}` | Finite ambiguity set, with explicit outside-set tests. |
| Sensitivity and corrected reset semantics | `results/sensitivity/rows.csv`, `results/reset_sensitivity_postfix/rows.csv` | `results/reset_sensitivity_postfix/legacy_comparison.json` reports 360/360 equal keys and no difference for the equal-prior historical configuration; distinct-prior case remains unit tested. |
| Post-review spatial availability, nonstationarity, and probe planning | `results/tmc_reviewer_stress/rows.csv` and `analysis/{summary.csv,provenance.json}` | `src/analyze_reviewer_stress.py`. |
| GeoLife development/validation fitting and disclosure-floor extension | `results/data_driven_ambiguity/models.json`, `results/data_driven_extension/rows.csv`, `analysis/{summary.csv,report.json}` | Effective state-path isolation; extensions are post-hoc/exploratory, not an independent confirmatory holdout. |
| Additional holdout/replay diagnostics | `results/review1_diagnostics/`, `results/geolife_decontamination.json` | See source-input hashes in `results/review1_diagnostics/provenance.json`. |
| Frozen table and numerical macro snapshots | `paper/*_table.tex`, `paper/*_numbers.tex` | Derived *table assets only*. The manuscript's `main.tex`, supplement and bibliographic sources are in Bpaper. |

**Not redistributed:** Microsoft's full GeoLife archive, processed trajectory caches containing coordinates, and hundreds of megabytes of frozen event-level `*.jsonl.gz`. GeoLife acquisition and deterministic preprocessing are documented in `data/README.md` and `data/manifest.json`. The original event logs remain accessible at the [fixed historical source result tree](https://github.com/js110/Bpaper/tree/2cc94ff6e7c4696dc364392dc19c8137d0cb7e65/results). The published aggregate rows and bootstrap provenance here are sufficient to re-evaluate the manuscript's aggregate analyses, but auditing *every individual logged event* requires the original archived logs.

**Reproducing:** Do not overwrite the frozen `results/` directories. For a completely new independent full run, obtain GeoLife, regenerate local caches with `python -m src.prepare_data`, then run `python -m src.reproduce --output results/new_reproduction`. This is computationally intensive. The unit suite explicitly skips the two tests that require assets intentionally excluded from the lightweight checkout; it does not claim those tests passed without the external data.

**Licensing:** The third-party PRIVIC conformance fixture is copied under `literature/recent/PRIVIC_LICENSE` (MIT, original authors). No license for the original BSP code is implicitly inferred by publishing this repository; the repository owner should separately choose an appropriate license if broader reuse rights are intended.
