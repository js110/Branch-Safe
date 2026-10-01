"""Reproduce review-1 descriptive diagnostics from frozen traces and event logs.

No simulation, training, or attack is rerun. Test positions evaluate a predictor
chosen using development positions only. Existing result files are read-only.
"""
import argparse
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .analyze_recent import lower_hull
from .geolife import effective_geolife_splits,state_path_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ['static', 'walk', 'commute', 'geolife']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='results/review1_diagnostics')
    ap.add_argument('--paper-assets', action='store_true')
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sources = []

    def source(name):
        p = ROOT / name
        sources.append(p)
        return p

    for split in ['development','validation','test']:
        source(f'data/geolife_{split}.npz')
    splits, _ = effective_geolife_splits()
    assert all(not set(splits[a]['users']) & set(splits[b]['users'])
               for a, b in [('development', 'test'), ('development', 'validation'), ('validation', 'test')])
    development = splits['development']['paths']
    test = splits['test']['paths']
    counts = np.bincount(development.ravel(), minlength=64)
    modes = np.flatnonzero(counts == counts.max())
    per_user = np.isin(test, modes).mean(axis=1) / len(modes)
    test_groups={}
    for i,path in enumerate(test):
        test_groups.setdefault(state_path_fingerprint(path),[]).append(i)
    group_members=list(test_groups.values())
    rng = np.random.default_rng(170926)
    draws=[]
    for selected in rng.integers(len(group_members), size=(1000, len(group_members))):
        idx=np.concatenate([np.asarray(group_members[j],dtype=int) for j in selected])
        draws.append(float(per_user[idx].mean()))
    draws=np.asarray(draws,dtype=float)
    prior = dict(fit_split='development', evaluate_split='test', map_cells=modes.tolist(),
                 development_peak=float(counts.max() / counts.sum()),
                 hit=float(per_user.mean()), low=float(np.quantile(draws, .025)),
                 high=float(np.quantile(draws, .975)), users=len(test), state_path_groups=len(group_members), slots=test.shape[1],
                 uniform_hit=1 / 64, bootstrap_seed=170926, bootstrap_replicates=1000,
                 interval='window-weighted point estimate with percentile state-path-group cluster bootstrap conditional on fixed development predictor; pointwise',
                 interpretation='No task outputs; fixed cell predictor; not a rerun of an adaptive attack')
    mobility = {s: dict(users=len(a['paths']), static_windows=int(np.all(np.diff(a['paths']) == 0, axis=1).sum()),
                       unique_cells_median=float(np.median([len(set(row)) for row in a['paths']])),
                       changes=int((np.diff(a['paths']) != 0).sum()),
                       adjacent_pairs=int(len(a['paths']) * (a['paths'].shape[1] - 1))) for s, a in splits.items()}
    with (out / 'prior_users.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['user', 'no_output_hit'])
        w.writerows(zip(splits['test']['users'], per_user))

    bins = [(1, 8), (9, 31), (32, 64)]
    area_rows = []
    exact_areas = []
    totals = {}
    for sc in SCENARIOS:
        grouped = {b: [0, 0, 0] for b in bins}
        with gzip.open(source(f'results/final/{sc}_adaptive_bsp_0.1.jsonl.gz'), 'rt') as f:
            for line in f:
                row = json.loads(line)
                for e in row['events']:
                    if not e['legitimate']:
                        continue
                    x0, x1, y0, y1 = e['rectangle']
                    area = (x1 - x0) * (y1 - y0)
                    opportunity = int(e['eligible'] and e['available'])
                    completed = int(e['reported'])
                    assert completed <= opportunity
                    group = next(b for b in bins if b[0] <= area <= b[1])
                    grouped[group] = [x + y for x, y in zip(grouped[group], [1, opportunity, completed])]
                    exact_areas.append(dict(scenario=sc, seed=row['seed'], user=row['user'], slot=e['slot'],
                                            area=area, opportunity=opportunity, completed=completed))
        for (lo, hi), (tasks, opportunities, completed) in grouped.items():
            area_rows.append(dict(scenario=sc, area_low=lo, area_high=hi, tasks=tasks,
                                  opportunities=opportunities, completed=completed,
                                  retention=completed / opportunities if opportunities else None))
        tasks, opportunities, completed = map(sum, zip(*grouped.values()))
        totals[sc] = dict(tasks=tasks, opportunities=opportunities, completed=completed,
                          retention=completed / opportunities, raw_completion=completed / tasks,
                          large_region_completion_share=grouped[(32, 64)][2] / completed)
    for name, rows in [('area_events.csv', exact_areas), ('task_area.csv', area_rows)]:
        with (out / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    summaries = list(csv.DictReader(source('results/combined/analysis/summary.csv').open()))
    main_rows = list(csv.DictReader(source('results/final/rows.csv').open()))
    lines = [r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'Scenario & Policy & Retention & Raw comp. & MAP hit & Error & 95\% cover \\', r'\midrule']
    main_numeric = []
    names = {'none': 'None', 'random': 'Random .5', 'rate': 'Period 2', 'coarse': 'Grid 2',
             'positive_only': 'Success only', 'bsp': 'BSP .1'}
    for sc in SCENARIOS:
        for method, par in [('none', 1), ('random', .5), ('rate', 2), ('coarse', 2), ('positive_only', .1), ('bsp', .1)]:
            s = next(s for s in summaries if s['scenario'] == sc and s['attack'] == 'adaptive'
                     and s['method'] == method and float(s['param']) == par)
            rr = [r for r in main_rows if r['condition'] == s['condition']]
            completed = sum(int(r['complete']) for r in rr)
            tasks = sum(int(r['legitimate']) for r in rr)
            opportunities = sum(int(r['opportunities']) for r in rr)
            raw = completed / tasks
            assert abs(completed / opportunities - float(s['utility'])) < 1e-12
            if method == 'bsp':
                assert (tasks, opportunities, completed) == tuple(totals[sc][k] for k in ['tasks', 'opportunities', 'completed'])
            main_numeric.append(dict(condition=s['condition'], tasks=tasks, opportunities=opportunities,
                                     completed=completed, raw_completion=raw))
            lines.append(f"{sc.capitalize()} & {names[method]} & {float(s['utility']):.3f} & {raw:.3f} & "
                         f"{float(s['hit']):.3f} & {float(s['error_cells']):.2f} & {float(s['covered95']):.3f} " + r'\\')
        if sc != SCENARIOS[-1]:
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'main_table_revision.tex').write_text('\n'.join(lines) + '\n')

    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Scenario & Area (cells) & Tasks & Opportunities & Completed & Retention \\', r'\midrule']
    for r in area_rows:
        u = '--' if r['retention'] is None else f"{r['retention']:.3f}"
        lines.append(f"{r['scenario'].capitalize()} & {r['area_low']}--{r['area_high']} & {r['tasks']} & "
                     f"{r['opportunities']} & {r['completed']} & {u} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'task_area_table.tex').write_text('\n'.join(lines) + '\n')
    lines = [r'\begin{tabular}{lrr}', r'\toprule', r'No-output predictor & Hit (\%) & 95\% interval \\', r'\midrule',
             r'Uniform random cell & 1.56 & Exact expectation \\',
             f"Development modal cell & {100*prior['hit']:.2f} & [{100*prior['low']:.2f}, {100*prior['high']:.2f}] " + r'\\',
             r'\bottomrule', r'\end{tabular}']
    (out / 'prior_diagnostic_table.tex').write_text('\n'.join(lines) + '\n')
    macros = {'PopulationPriorHit': f"{100*prior['hit']:.2f}", 'PopulationPriorPeak': f"{100*prior['development_peak']:.2f}",
              'ReplayStaticWindows': str(mobility['test']['static_windows']),
              'ReplayMedianCells': f"{mobility['test']['unique_cells_median']:.0f}",
              'LargeTaskShareMin': f"{100*min(s['large_region_completion_share'] for s in totals.values()):.1f}",
              'LargeTaskShareMax': f"{100*max(s['large_region_completion_share'] for s in totals.values()):.1f}",
              'RawCompletionMin': f"{100*min(s['raw_completion'] for s in totals.values()):.2f}",
              'RawCompletionMax': f"{100*max(s['raw_completion'] for s in totals.values()):.2f}"}
    (out / 'review_numbers.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k, v in macros.items()) + '\n')

    recent = list(csv.DictReader(source('results/recent_comparison/summary.csv').open()))
    styles = {'bsp': ('BSP', '#D55E00', '^'), 'pml': ('PML-T', '#009E73', 's'),
              'privic': ('PRIVIC-T', '#0072B2', 'D'), 'random': ('Random participation', '#555555', 'o')}
    selected = [s for s in recent if s['scenario'] == 'commute' and s['method'] in styles]
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300}):
        fig, ax = plt.subplots(figsize=(5.8, 3.6), layout='constrained')
        for method, (label, color, marker) in styles.items():
            rows = [s for s in selected if s['method'] == method]
            xy = [(float(s['utility']), float(s['hit'])) for s in rows]
            hull = lower_hull(xy)
            ax.scatter(*zip(*xy), color=color, marker=marker, alpha=.28, s=18)
            ax.plot(*zip(*hull), color=color, marker=marker, ms=4, lw=1.2, label=label)
            for x, y in hull:
                s = next(s for s in rows if float(s['utility']) == x and float(s['hit']) == y)
                ax.errorbar(x, y, yerr=[[max(0, y-float(s['hit_low']))], [max(0, float(s['hit_high'])-y)]],
                            fmt='none', ecolor=color, lw=.7, alpha=.55)
        upper = np.ceil((max(float(s['hit_high']) for s in selected) + .005) * 100) / 100
        ax.axhline(1/64, ls=':', color='#777777', lw=1)
        ax.set(xlabel='Legitimate opportunity retention', ylabel='MAP cell hit rate',
               xlim=(0, 1.03), ylim=(0, upper), title='Commute: expanded vertical scale')
        ax.grid(alpha=.15)
        ax.legend(ncol=2, loc='upper left', fontsize=8)
        fig.savefig(out / 'recent_commute_zoom.pdf')
        fig.savefig(out / 'recent_commute_zoom.png')
        plt.close(fig)

    result = dict(prior=prior, mobility=mobility, task_area=area_rows, totals=totals,
                  main_table=main_numeric, numeric_macros=macros,
                  study_status='post-review descriptive diagnostics; unchanged simulation outputs')
    (out / 'diagnostics.json').write_text(json.dumps(result, indent=2) + '\n')
    sources.append(Path(__file__).resolve())
    provenance = dict(inputs={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                      region_bins=bins, region_condition='adaptive BSP rho=0.1; not envelope mixtures',
                      raw_completion='sum completions / sum all normal tasks; not mean per-user ratio',
                      prior_interpretation=prior['interpretation'], prior_estimator='effective test-window/position-weighted point estimate; complete state-path-group cluster bootstrap', test_use='descriptive evaluation only; predictor fitted on development',
                      zoom='same saved points and envelope as main recent-method figure; expanded vertical scale',
                      numpy=np.__version__, matplotlib=matplotlib.__version__)
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    assets = ['task_area_table.tex', 'recent_commute_zoom.pdf', 'review_numbers.tex']
    if args.paper_assets:
        for name in assets:
            shutil.copyfile(out / name, ROOT / 'paper' / name)
    print(json.dumps(dict(prior=prior, mobility=mobility['test'], totals=totals), indent=2))


if __name__ == '__main__':
    main()
