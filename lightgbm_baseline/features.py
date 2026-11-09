import numpy as np
import pandas as pd

from .config import FEATURE_MAPPING, FEATURES, STOCK_COLUMNS


def build_features(raw, global_data):
    raw, global_data = raw.copy(), global_data.copy()
    for frame in (raw, global_data):
        frame['date'] = pd.to_datetime(frame.date, errors='raise')
        if frame.date.isna().any() or not frame.date.eq(frame.date.dt.normalize()).all():
            raise ValueError('Dates must be non-null calendar dates')
    duplicates = int(raw.duplicated(['stock_id', 'date']).sum())
    if duplicates or global_data.date.duplicated().any():
        raise ValueError(f'Duplicate stock/date keys: {duplicates}, or duplicate global dates')
    raw = raw.sort_values(['stock_id', 'date']).reset_index(drop=True)
    for col in [*STOCK_COLUMNS, 'y']:
        raw[col] = pd.to_numeric(raw[col], errors='raise')
    global_data['^VIX'] = pd.to_numeric(global_data['^VIX'], errors='raise')
    report = {'raw_rows': len(raw), 'duplicate_keys': duplicates,
              'global_duplicate_dates': int(global_data.date.duplicated().sum()),
              'official_numeric_dtypes': {c: str(raw[c].dtype) for c in STOCK_COLUMNS},
              'vix_numeric_dtype': str(global_data['^VIX'].dtype),
              'raw_missing': raw.isna().sum().to_dict(),
              'global_missing': global_data.isna().sum().to_dict(),
              'raw_nonfinite': {c: int((raw[c].notna() & ~np.isfinite(raw[c])).sum()) for c in STOCK_COLUMNS}}
    # Holdout/static rows never enter feature construction or diagnostics of labels.
    raw = raw[raw.date.between('2020-01-01', '2025-12-31')].copy()
    global_data = global_data[global_data.date < pd.Timestamp('2026-01-01')].copy()
    if not raw.y.isin([0, 1]).all():
        raise ValueError('Official y must be binary and nonmissing')
    valid_return = (raw['etl:adj_open'] > 0) & np.isfinite(raw['etl:adj_open']) & np.isfinite(raw['etl:adj_close'])
    raw['actual_return'] = ((raw['etl:adj_close'] - raw['etl:adj_open']) / raw['etl:adj_open']).where(valid_return)
    report['label_mismatches'] = int((valid_return & raw.y.ne((raw.actual_return > 0).astype(int))).sum())
    report['invalid_returns'] = int((~valid_return).sum())
    grouped = raw.groupby('stock_id', sort=False)
    shifted = grouped[STOCK_COLUMNS].shift(1).rename(columns=FEATURE_MAPPING)
    result = raw[['date', 'stock_id', 'y', 'actual_return']].join(shifted)
    result['stock_source_date'] = grouped.date.shift(1)
    result['previous_direction'] = grouped.y.shift(1)
    report['shifted_missing'] = shifted.isna().sum().to_dict()
    vix = global_data.rename(columns={'date': 'vix_source_date', '^VIX': 'vix'})
    result = pd.merge_asof(result.sort_values('date'), vix.sort_values('vix_source_date'),
                           left_on='date', right_on='vix_source_date',
                           direction='backward', allow_exact_matches=False)
    for prefix in ('stock', 'vix'):
        source = result[f'{prefix}_source_date']
        age = (result.date - source).dt.days
        report[f'{prefix}_timing_violations'] = int((source.notna() & (source >= result.date)).sum())
        report[f'{prefix}_stale_over_7_days'] = int((age > 7).sum())
        report[f'{prefix}_max_age_days'] = None if age.dropna().empty else int(age.max())
        report[f'{prefix}_stale_events'] = [
            {'stock_id': row.stock_id, 'date': str(row.date.date()),
             'source_date': str(row[f'{prefix}_source_date'].date())}
            for _, row in result.loc[age > 7].iterrows()]
    report['aligned_missing'] = result[FEATURES].isna().sum().to_dict()
    complete = np.isfinite(result[FEATURES]).all(axis=1) & np.isfinite(result.actual_return)
    report['excluded_rows'] = int((~complete).sum())
    report['retained_rows'] = int(complete.sum())
    report['exclusions_by_year'] = result.assign(excluded=~complete).groupby(result.date.dt.year).excluded.agg(['sum', 'count']).to_dict('index')
    if report['label_mismatches']:
        error = ValueError(f"Official labels disagree with adjusted OHLC: {report['label_mismatches']}; inspect data before training")
        error.quality_report = report
        raise error
    return result[complete].copy(), report
