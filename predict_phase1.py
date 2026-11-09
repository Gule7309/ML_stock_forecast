"""Replay the frozen C26 + C21 Phase 1 route; never trains or uploads."""
import argparse
import hashlib
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm_baseline.experiments.asof_predict import load_history
from lightgbm_baseline.experiments.rank_ablation import ranked_asof
from lightgbm_baseline.experiments.static_candidate import info_and_template


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-zip', required=True)
    ap.add_argument('--template', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    base = Path(__file__).parent / 'lightgbm_baseline/experiments/next_stage/RANK20261104_V1/candidate'
    manifest = json.loads((base / 'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256(Path(args.data_zip).read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Official data ZIP differs from the scored version')
    info, template, dates = info_and_template(args.data_zip, args.template)
    if hashlib.sha256(Path(args.template).read_bytes()).hexdigest() != manifest['template_sha256']:
        raise ValueError('Official template differs from the scored version')
    history, world, stocks = load_history(args.data_zip)
    models = {}
    for h in range(1, 21):
        path = base / f'models/h{h:02}.txt'
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['files'][f'models/h{h:02}.txt']:
            raise ValueError('Model hash mismatch')
        models[h] = lgb.Booster(model_file=str(path))
    snapshot = ranked_asof(history, world, '2026-09-01', stocks)
    result = template.copy()
    for date in dates:
        h = 1 if date < pd.Timestamp('2026-09-01') else dates[dates >= pd.Timestamp('2026-09-01')].get_loc(date) + 1
        frame = ranked_asof(history, world, date, stocks) if date < pd.Timestamp('2026-09-01') else snapshot
        features = manifest['h1_features'] if h == 1 else manifest['later_horizon_features']
        if models[h].feature_name() != features:
            raise ValueError('Feature order mismatch')
        forecast = models[h].predict(frame[features], num_threads=4)
        if not np.isfinite(forecast).all():
            raise ValueError('Nonfinite prediction')
        result[date.strftime('%Y-%m-%d')] = pd.Series((forecast > 0).astype(int), index=frame.stock_id).reindex(stocks).to_numpy()
    reference = pd.read_csv(Path(__file__).parent / 'submissions/phase1_lgbm_C26_C21_20261108.csv', dtype=str)
    pd.testing.assert_frame_equal(result.astype(str), reference)
    result.to_csv(output, index=False)
    print(f'PASS: all 4200 values match the scored C26+C21 CSV; saved {output}')


if __name__ == '__main__':
    main()
