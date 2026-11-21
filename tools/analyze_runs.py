"""
Aggregate the Oxford-IIIT Pet study runs and apply the pre-registered decision rule
(docs/experiment_design.md).

Expected layout: one folder per run, named <arm>_s<seed>, e.g.
    <root>/mal_dense_s0/{match_iou.csv, log.txt, eval_test.json}
with arm in {mal_dense, vfl_dense, mal_plain, vfl_plain, ref_imagenet}.

Usage:
    python tools/analyze_runs.py --root outputs --out outputs/analysis
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

from scipy import stats

ARMS = ['mal_dense', 'vfl_dense', 'mal_plain', 'vfl_plain', 'ref_imagenet']
METRICS = ['AP', 'AP50', 'AP75', 'AP_M', 'AP_L']
DELTA_F_MAX_PP = 5.0     # test (b): Dense O2O raises f by less than this many percentage points
REF_RATIO_MAX = 0.5      # check (a): f(COCO init) <= this * f(ImageNet-backbone init)
COCO_MAL_GAIN = 0.3      # the paper's smallest reported MAL gain on COCO, in AP points


def mean(xs):
    return sum(xs) / len(xs)


def load_run(path):
    run = {'iou': {}, 'val_ap': {}, 'time': {}, 'mem': {}, 'test': None}
    p = os.path.join(path, 'match_iou.csv')
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):   # a resumed run can repeat an epoch: the last row wins
                run['iou'][int(r['epoch'])] = {k: float(v) for k, v in r.items()}
    p = os.path.join(path, 'log.txt')
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                run['val_ap'][r['epoch']] = r['test_coco_eval_bbox'][0]
                if 'train_time_s' in r:
                    run['time'][r['epoch']] = r['train_time_s']
                    run['mem'][r['epoch']] = r.get('peak_gpu_mem_mb', 0.)
    p = os.path.join(path, 'eval_test.json')
    if os.path.exists(p):
        with open(p) as f:
            run['test'] = json.load(f)
    return run


def window_stats(run, lo, hi):
    """Pooled over all matches in epochs [lo, hi): low-IoU fraction f, mean IoU, targets per image."""
    rows = [r for e, r in run['iou'].items() if lo <= e < hi]
    n = sum(r['num_matches'] for r in rows)
    if not rows or n == 0:
        return None
    return {
        'epochs': len(rows),
        'matches': int(n),
        'f': sum(r['frac_iou_lt_0.5'] * r['num_matches'] for r in rows) / n,
        'f03': sum(r['frac_iou_lt_0.3'] * r['num_matches'] for r in rows) / n,
        'mean_iou': sum(r['mean_iou'] * r['num_matches'] for r in rows) / n,
        'targets_per_image': mean([r['mean_targets_per_image'] for r in rows]),
    }


def paired_ci(diffs):
    """Mean of paired differences with a two-sided 95% t interval."""
    n = len(diffs)
    m = mean(diffs)
    if n < 2:
        return m, None, None
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    half = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, m - half, m + half


def gap_verdict(lo, hi):
    if lo is None:
        return 'too few pairs for an interval'
    if lo > 0:
        return 'MAL helps (interval excludes 0)'
    if hi < 0:
        return 'MAL hurts (interval excludes 0)'
    if hi < COCO_MAL_GAIN:
        return f'smaller gain than on COCO (interval entirely below +{COCO_MAL_GAIN})'
    return 'unresolved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='outputs')
    parser.add_argument('--out', default=None, help='default: <root>/analysis')
    parser.add_argument('--window', type=int, nargs=2, default=[4, 29], help='Dense O2O epochs [lo, hi)')
    args = parser.parse_args()
    out_dir = args.out or os.path.join(args.root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)
    lo, hi = args.window

    runs = defaultdict(dict)   # arm -> seed -> run
    for name in sorted(os.listdir(args.root)):
        m = re.fullmatch(r'(%s)_s(\d+)' % '|'.join(ARMS), name)
        if m:
            runs[m.group(1)][int(m.group(2))] = load_run(os.path.join(args.root, name))

    L = []   # report lines
    result = {'window': [lo, hi], 'runs': {}}

    # ---- per-run table
    L += [f'# Oxford-IIIT Pet study: results', '',
          f'Dense O2O window: epochs {lo}-{hi - 1}. f = fraction of final-layer Hungarian matches with IoU < 0.5, '
          f'pooled over the window.', '', '## Runs', '',
          '| run | window epochs | f (%) | mean IoU | targets/img | s/epoch | peak mem (MB) | best val AP | '
          + ' | '.join(f'test {k}' for k in METRICS) + ' |',
          '|---|---|---|---|---|---|---|---|' + '---|' * len(METRICS)]
    ws = defaultdict(dict)
    for arm in ARMS:
        for seed, run in sorted(runs[arm].items()):
            w = window_stats(run, lo, hi)
            ws[arm][seed] = w
            t = mean(list(run['time'].values())) if run['time'] else None
            mem = max(run['mem'].values()) if run['mem'] else None
            best = max(run['val_ap'].values()) if run['val_ap'] else None
            test = run['test']
            cells = [f'{arm}_s{seed}',
                     str(w['epochs']) if w else '-',
                     f"{100 * w['f']:.2f}" if w else '-',
                     f"{w['mean_iou']:.3f}" if w else '-',
                     f"{w['targets_per_image']:.2f}" if w else '-',
                     f'{t:.0f}' if t else '-',
                     f'{mem:.0f}' if mem else '-',
                     f'{100 * best:.2f}' if best is not None else '-']
            cells += [f'{100 * test[k]:.2f}' if test else '-' for k in METRICS]
            L.append('| ' + ' | '.join(cells) + ' |')
            result['runs'][f'{arm}_s{seed}'] = {'window': w, 'sec_per_epoch': t, 'best_val_ap': best, 'test': test}
    L.append('')

    # ---- test (b): Dense O2O effect on the low-IoU fraction
    L += ['## Test (b): does Dense O2O raise the low-IoU match fraction?', '',
          f'Delta f = f(Dense O2O on, seed s) - mean f(Dense O2O off), same loss. Passes if below {DELTA_F_MAX_PP} pp.', '',
          '| loss | seed | f on (%) | mean f off (%) | delta f (pp) | pass |', '|---|---|---|---|---|---|']
    b_pass = []
    for loss in ('mal', 'vfl'):
        off = [w['f'] for w in ws[f'{loss}_plain'].values() if w]
        for seed, w in sorted(ws[f'{loss}_dense'].items()):
            if not w or not off:
                continue
            d = 100 * (w['f'] - mean(off))
            b_pass.append(d < DELTA_F_MAX_PP)
            L.append(f"| {loss.upper()} | {seed} | {100 * w['f']:.2f} | {100 * mean(off):.2f} | {d:+.2f} | {'yes' if b_pass[-1] else 'no'} |")
    L.append('')

    # ---- check (a): COCO init vs ImageNet-backbone-only reference
    L += ['## Sanity check (a): COCO init vs ImageNet-backbone reference', '']
    a_pass = None
    coco = [w['f'] for w in ws['mal_dense'].values() if w]
    ref = [w['f'] for w in ws['ref_imagenet'].values() if w]
    if coco and ref:
        ratio = mean(coco) / mean(ref)
        a_pass = ratio <= REF_RATIO_MAX
        L.append(f'f(COCO init, MAL + Dense O2O) = {100 * mean(coco):.2f}%, f(reference) = {100 * mean(ref):.2f}%, '
                 f"ratio = {ratio:.2f} (needs <= {REF_RATIO_MAX}): {'pass' if a_pass else 'fail'}.")
    else:
        L.append('Not available yet (needs mal_dense and ref_imagenet runs).')
    L.append('')

    # ---- H1 verdict
    if not b_pass:
        verdict = 'not available yet'
    elif not any(b_pass):
        verdict = 'H1 rejected'
    elif all(b_pass) and a_pass:
        verdict = 'H1 supported'
    elif all(b_pass) and a_pass is None:
        verdict = 'pending: (b) holds in every run, sanity check (a) not available yet'
    else:
        verdict = 'inconclusive'
    L += ['## Decision', '', f'**{verdict}** ((b) passes in {sum(b_pass)}/{len(b_pass)} Dense O2O runs).', '']
    result['test_b_pass'] = b_pass
    result['check_a_pass'] = a_pass
    result['verdict'] = verdict

    # ---- secondary: MAL - VFL gap on the test split
    L += ['## Secondary: MAL - VFL on the test split (paired by seed, AP points)', '',
          '| Dense O2O | metric | pairs | per-seed differences | mean | 95% CI | reading |', '|---|---|---|---|---|---|---|']
    result['gaps'] = {}
    for setting in ('dense', 'plain'):
        a, b = runs[f'mal_{setting}'], runs[f'vfl_{setting}']
        seeds = [s for s in sorted(a) if s in b and a[s]['test'] and b[s]['test']]
        if not seeds:
            continue
        for k in METRICS:
            diffs = [100 * (a[s]['test'][k] - b[s]['test'][k]) for s in seeds]
            m, cl, ch = paired_ci(diffs)
            ci = f'[{cl:+.2f}, {ch:+.2f}]' if cl is not None else '-'
            reading = gap_verdict(cl, ch) if k in ('AP', 'AP75') else ''
            L.append(f"| {'on' if setting == 'dense' else 'off'} | {k} | {len(seeds)} | "
                     f"{', '.join(f'{d:+.2f}' for d in diffs)} | {m:+.2f} | {ci} | {reading} |")
            result['gaps'][f'{setting}_{k}'] = {'seeds': seeds, 'diffs': diffs, 'mean': m, 'ci': [cl, ch]}
    L.append('')

    with open(os.path.join(out_dir, 'results.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print('\n'.join(L))

    # ---- figures
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: skipping figures')
        return
    colors = {'mal_dense': 'tab:red', 'vfl_dense': 'tab:blue', 'mal_plain': 'tab:orange',
              'vfl_plain': 'tab:cyan', 'ref_imagenet': 'tab:gray'}
    for fname, ylabel, getter in [
        ('low_iou_fraction.png', 'matches with IoU < 0.5 (%)', lambda r: {e: 100 * v['frac_iou_lt_0.5'] for e, v in r['iou'].items()}),
        ('val_ap.png', 'val AP (%)', lambda r: {e: 100 * v for e, v in r['val_ap'].items()}),
    ]:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for arm in ARMS:
            series = [getter(r) for r in runs[arm].values()]
            series = [s for s in series if s]
            if not series:
                continue
            for s in series:
                ax.plot(sorted(s), [s[e] for e in sorted(s)], color=colors[arm], alpha=0.25, lw=1)
            epochs = sorted(set.intersection(*[set(s) for s in series]))
            ax.plot(epochs, [mean([s[e] for s in series]) for e in epochs], color=colors[arm], lw=2,
                    label=f'{arm} (n={len(series)})')
        ax.axvspan(lo - 0.5, hi - 0.5, color='k', alpha=0.05, lw=0)
        ax.set_xlabel('epoch (shaded: Dense O2O window)')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=200)
        plt.close(fig)
    print(f'wrote {out_dir}/results.md, results.json and figures')


if __name__ == '__main__':
    main()
