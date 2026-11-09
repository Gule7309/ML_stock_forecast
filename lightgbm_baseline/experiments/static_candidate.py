"""Official-only, cutoff-safe data rebuilding and an offline Static candidate."""
from pathlib import Path
from zipfile import ZipFile
import hashlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from lightgbm_baseline.config import STOCK_COLUMNS
from lightgbm_baseline.features import build_features
from .asof_predict import FLOW_COLUMNS, load_history, build_asof
from .submission_format import format_predictions

FLOW_PREFIX='institutional_investors_trading_summary:'
OWNERSHIP='foreign_investors_shareholding:全體外資及陸資持股比率'
RANK_SOURCES=['rank_intraday_source','rank_foreign5_source','rank_trust5_source']
RANK_FEATURES=['xs_rank_previous_intraday','xs_rank_foreign_flow_5','xs_rank_trust_flow_5']
REQUIRED=['date',*STOCK_COLUMNS,*[FLOW_PREFIX+x for x in FLOW_COLUMNS.values()]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def keys_sha(frame,columns):
    return hashlib.sha256(pd.util.hash_pandas_object(frame[columns],index=False).values.tobytes()).hexdigest()


def info_and_template(zip_path,template_path):
    with ZipFile(zip_path) as z:
        bad=z.testzip()
        if bad:raise ValueError(f'CRC error: {bad}')
        info=json.loads(z.read('data/info.json'))
        for key,n in [('Top50@Y',50),('Top100@Y',100)]:
            ids=info[key]
            if len(ids)!=n or len(set(ids))!=n or not all(isinstance(s,str) and s and '/' not in s and '\\' not in s for s in ids):
                raise ValueError(f'Invalid {key}')
        if not set(info['Top50@Y']).issubset(info['Top100@Y']):raise ValueError('Top50 not subset of Top100')
    template=pd.read_csv(template_path,dtype=str)
    if template.shape!=(50,85) or template.iloc[:,0].tolist()!=info['Top50@Y']:
        raise ValueError('Expected exact50-stock84-date official template and official stock order')
    dates=pd.DatetimeIndex(pd.to_datetime(template.columns[1:],errors='raise'))
    if dates.has_duplicates or not dates.is_monotonic_increasing or dates.min()!=pd.Timestamp('2026-06-01') or dates.max()!=pd.Timestamp('2026-09-30'):
        raise ValueError('Unexpected Static date headers')
    return info,template,dates


def read_training(zip_path,stocks,end):
    """A date-only pass fixes nrows BEFORE any y column is requested."""
    end=pd.Timestamp(end)
    if end>=pd.Timestamp('2026-01-01'):raise ValueError('Training may not read2026 targets')
    records=[];frames=[]
    with ZipFile(zip_path) as z:
        for stock in stocks:
            member=f'data/market/{stock}.csv'
            date=pd.to_datetime(pd.read_csv(z.open(member),usecols=['date']).date,errors='raise')
            if date.isna().any() or not date.is_monotonic_increasing or date.duplicated().any():
                raise ValueError(f'Unsorted/null/duplicate dates: {stock}')
            count=int(date.le(end).sum())
            if not count:raise ValueError(f'No training prefix: {stock}')
            # nrows protects the label field even when later y values contain arbitrary strings.
            frame=pd.read_csv(z.open(member),nrows=count,usecols=[*REQUIRED,'y'])
            frame['date']=pd.to_datetime(frame.date)
            if frame.date.max()>end:raise ValueError('Prefix cutoff error')
            for col in [*STOCK_COLUMNS,*[FLOW_PREFIX+x for x in FLOW_COLUMNS.values()],'y']:
                frame[col]=pd.to_numeric(frame[col],errors='raise')
            frame['stock_id']=str(stock)
            if not frame.y.isin([0,1]).all():raise ValueError(f'Invalid training labels: {stock}')
            valid=(frame['etl:adj_open']>0)&np.isfinite(frame['etl:adj_open'])&np.isfinite(frame['etl:adj_close'])
            ret=(frame['etl:adj_close']-frame['etl:adj_open'])/frame['etl:adj_open']
            mismatch=int((valid & frame.y.ne((ret>0).astype(int))).sum())
            if mismatch:raise ValueError(f'Training label mismatch: {stock}: {mismatch}')
            volume=frame['price:成交股數']
            excess={name:int((frame[FLOW_PREFIX+column].abs()>volume).sum()) for name,column in FLOW_COLUMNS.items()}
            if volume.lt(0).any():raise ValueError(f'Negative volume: {stock}')
            records.append(dict(stock_id=str(stock),rows=count,label_last_date=str(frame.date.max().date()),
                                label_mismatches=mismatch,invalid_target_rows=int((~valid).sum()),zero_volume_rows=int(volume.eq(0).sum()),
                                absolute_flow_exceeds_volume=excess,
                                numeric_dtypes={col:str(frame[col].dtype) for col in REQUIRED if col!='date'}))
            frames.append(frame)
        global_data=pd.read_csv(z.open('data/global/global_data.csv'),usecols=['date','^VIX'])
    global_data['date']=pd.to_datetime(global_data.date)
    global_data=global_data[global_data.date<=end].copy()
    global_data['^VIX']=pd.to_numeric(global_data['^VIX'],errors='raise')
    if global_data.date.duplicated().any():raise ValueError('Duplicate global dates')
    return pd.concat(frames,ignore_index=True),global_data,records


def canonical(raw,world,features):
    """Preserve C4/C10's original price-warmup mask without adding price inputs."""
    base,quality=build_features(raw[['date','stock_id','y',*STOCK_COLUMNS]],world)
    frames=[]
    for stock,r in raw.groupby('stock_id',sort=False):
        r=r.sort_values('date').reset_index(drop=True)
        o,h,l,c=[r[f'etl:adj_{name}'] for name in ['open','high','low','close']]
        volume=r['price:成交股數']
        def ratio(a,b):return a/b.replace(0,np.nan)
        x=pd.DataFrame(index=r.index)
        # The historical canonical experiment required these8 columns for row eligibility.
        x['warm_intraday']=ratio(c-o,o);x['warm_close_return']=c.pct_change(fill_method=None)
        x['warm_overnight']=ratio(o-c.shift(),c.shift());x['warm_range']=ratio(h-l,o)
        for w in [5,20]:
            x[f'warm_momentum_{w}']=ratio(c,c.shift(w))-1
            x[f'warm_volatility_{w}']=c.pct_change(fill_method=None).rolling(w,min_periods=w).std(ddof=1)
        for name,col in FLOW_COLUMNS.items():
            flow=r[FLOW_PREFIX+col]
            x[name+'_flow_ratio']=ratio(flow,volume)
            x[name+'_flow_ratio_5']=ratio(flow.rolling(5,min_periods=5).sum(),volume.rolling(5,min_periods=5).sum())
        x['ownership_change_5']=r[OWNERSHIP].diff(5)
        ret=(c-o)/o
        x=x.shift(1).replace([np.inf,-np.inf],np.nan)
        for lag in range(1,21):x[f'raw_intraday_lag_{lag}']=ret.shift(lag)
        x['date']=r.date;x['stock_id']=str(stock)
        frames.append(x)
    extra=pd.concat(frames,ignore_index=True)
    data=base.merge(extra,on=['date','stock_id'],validate='one_to_one')
    warm=[col for col in extra if col.startswith('warm_')]
    valid=np.isfinite(data[features+warm]).all(axis=1)
    quality['canonical_warmup_or_feature_exclusions']=int((~valid).sum())
    data=data.loc[valid,['date','stock_id','y','actual_return','stock_source_date','vix_source_date',*features]].copy()
    data[features]=data[features].astype('float64').round(10)
    data=data.sort_values(['date','stock_id']).reset_index(drop=True)
    assert (data.stock_source_date<data.date).all() and (data.vix_source_date<data.date).all()
    quality['canonical_retained_rows']=len(data)
    return data,quality


def horizon_pairs(data,features,horizon,end):
    if horizon<1:raise ValueError('Invalid horizon')
    history=data[data.date<=pd.Timestamp(end)].copy()
    days=pd.DatetimeIndex(sorted(history.date.unique()))
    mapping=dict(zip(days[:len(days)-horizon+1],days[horizon-1:]))
    history['target_date']=history.date.map(mapping)
    targets=history[['date','stock_id','actual_return']].rename(columns={'date':'target_date','actual_return':'target_return'})
    pairs=history[['date','stock_id','target_date',*features]].merge(targets,on=['target_date','stock_id'],validate='many_to_one')
    pairs=pairs.sort_values(['date','stock_id']).reset_index(drop=True)
    if pairs.empty or not (pairs.date<=pairs.target_date).all() or pairs.target_date.max()>pd.Timestamp(end):
        raise ValueError('Invalid horizon training pairs')
    return pairs


def rank_frame(history,stocks,calendar):
    """Compute past-only ranks for all Top50 BEFORE joining any target/eligibility mask."""
    if 'y' in history or 'actual_return' in history:raise ValueError('Rank inputs must be label-free')
    targets=pd.DataFrame({'date':pd.DatetimeIndex(calendar).sort_values().unique()})
    frames=[]
    for stock in stocks:
        r=history[history.stock_id==stock].sort_values('date')
        v=r['price:成交股數']
        source=pd.DataFrame({'rank_source_date':r.date,
            RANK_SOURCES[0]:(r['etl:adj_close']-r['etl:adj_open'])/r['etl:adj_open'].replace(0,np.nan),
            RANK_SOURCES[1]:r[FLOW_PREFIX+FLOW_COLUMNS['foreign']].rolling(5,min_periods=5).sum()/v.rolling(5,min_periods=5).sum().replace(0,np.nan),
            RANK_SOURCES[2]:r[FLOW_PREFIX+FLOW_COLUMNS['trust']].rolling(5,min_periods=5).sum()/v.rolling(5,min_periods=5).sum().replace(0,np.nan)})
        source[RANK_SOURCES]=source[RANK_SOURCES].replace([np.inf,-np.inf],np.nan)
        out=pd.merge_asof(targets,source.sort_values('rank_source_date'),left_on='date',right_on='rank_source_date',direction='backward',allow_exact_matches=False)
        out['stock_id']=str(stock);frames.append(out)
    out=pd.concat(frames,ignore_index=True)
    for source,name in zip(RANK_SOURCES,RANK_FEATURES):
        out[name]=out.groupby('date')[source].rank(method='average',pct=True)
    assert (out.rank_source_date.dropna()<out.loc[out.rank_source_date.notna(),'date']).all()
    return out[['date','stock_id','rank_source_date',*RANK_FEATURES]].sort_values(['date','stock_id']).reset_index(drop=True)


def rolling_train(data,year):
    return data[data.date.between(f'{year-2}-01-01',f'{year-1}-12-31')].copy()


def check_static_keys(pred,stocks,dates):
    expected=pd.MultiIndex.from_product([stocks,pd.DatetimeIndex(dates).strftime('%Y-%m-%d')],names=['stock_id','date'])
    actual=pd.MultiIndex.from_frame(pred.assign(date=pd.to_datetime(pred.date).dt.strftime('%Y-%m-%d'))[['stock_id','date']])
    if len(pred)!=len(expected) or actual.has_duplicates or len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Static key coverage mismatch')
    if not pred.pred_label.isin([0,1]).all():raise ValueError('Nonbinary Static prediction')


def fit_saved(train,features,target,params,path,replay,repeat=False):
    if Path(path).exists():raise FileExistsError(path)
    model=lgb.LGBMRegressor(**params).fit(train[features],target)
    model.booster_.save_model(str(path))
    restored=lgb.Booster(model_file=str(path))
    if restored.feature_name()!=features or restored.num_trees()!=params['n_estimators']:raise ValueError('Native model mismatch')
    a=model.predict(replay[features]);b=restored.predict(replay[features])
    np.testing.assert_array_equal(a,b)
    repeat_error=None
    if repeat:
        second=lgb.LGBMRegressor(**params).fit(train[features],target)
        repeat_error=float(np.max(np.abs(a-second.predict(replay[features]))))
        assert repeat_error==0.
    return restored,dict(model_sha256=sha(path),n_train=len(train),n_features=len(features),native_reload_max_error=0.,
                         repeat_max_error=repeat_error,training_keys_sha256=keys_sha(train,[c for c in ['date','stock_id','target_date'] if c in train]))


def package(run_dir,zip_path,template_path,data,config):
    output=run_dir/'delivery';output.mkdir(exist_ok=False)
    models=output/'models';models.mkdir()
    info,template,dates=info_and_template(zip_path,template_path)
    history,world,stocks=load_history(zip_path)  # usecols never includes y
    cutoff=min([history.loc[history.stock_id==s,'date'].max() for s in stocks]+[world.date.max()])
    future=dates[dates>cutoff]
    if cutoff!=pd.Timestamp('2026-08-31') or len(future)!=20:raise ValueError('Unexpected data cutoff/template; do not change routing silently')
    features=config['features'];params=config['params']
    snapshot=build_asof(history,world,future.min(),stocks)
    model1,check=fit_saved(data,features,data.actual_return,params,models/'h01.txt',snapshot,repeat=True)
    model_catalog=[dict(horizon=1,**check,training_end=str(data.date.max().date()),reused_for='C10 daily and C21 h1')]
    pd.DataFrame({'feature':features,'gain':model1.feature_importance('gain')}).to_csv(output/'gain_h01.csv',index=False)
    rows=[];lineage=[]
    for date in dates[dates<=cutoff]:
        frame=build_asof(history,world,date,stocks)
        pred=model1.predict(frame[features])
        p=frame[['date','stock_id','stock_source_date','vix_source_date']].copy()
        p['origin_date']=date;p['horizon']=1;p['route']='C10-daily';p['model']='models/h01.txt'
        p['pred_return']=pred;p['pred_probability']=np.nan;p['pred_label']=(pred>0).astype(int);rows.append(p)
    for h,date in enumerate(future,1):
        if h==1:model=model1
        else:
            pairs=horizon_pairs(data,features,h,'2025-12-31')
            model,check=fit_saved(pairs,features,pairs.target_return,params,models/f'h{h:02d}.txt',snapshot,repeat=h==2)
            model_catalog.append(dict(horizon=h,**check,training_feature_end=str(pairs.date.max().date()),training_end=str(pairs.target_date.max().date())))
        pred=model.predict(snapshot[features])
        p=snapshot[['stock_id','stock_source_date','vix_source_date']].copy()
        p['date']=date;p['origin_date']=future.min();p['horizon']=h;p['route']='C21-frozen';p['model']=f'models/h{h:02d}.txt'
        p['pred_return']=pred;p['pred_probability']=np.nan;p['pred_label']=(pred>0).astype(int);rows.append(p)
        print(f'package horizon={h} complete',flush=True)
    pred=pd.concat(rows,ignore_index=True).sort_values(['date','stock_id']).reset_index(drop=True)
    check_static_keys(pred,stocks,dates)
    assert (pred.stock_source_date<pred.origin_date).all() and (pred.vix_source_date<pred.origin_date).all()
    pred['stock_source_age_days']=(pred.date-pred.stock_source_date).dt.days
    pred['vix_source_age_days']=(pred.date-pred.vix_source_date).dt.days
    pred.to_csv(output/'predictions_long.csv',index=False,float_format='%.17g')
    submission=format_predictions(template_path,output/'predictions_long.csv',output/'static_submission.csv')
    result=pd.read_csv(submission,dtype={template.columns[0]:str})
    assert result.shape==(50,85) and result.columns.tolist()==template.columns.tolist()
    assert result.iloc[:,0].tolist()==stocks and result.iloc[:,1:].isin([0,1]).all().all()
    save(output/'model_catalog.json',model_catalog)
    save(output/'config.json',{**config,'training_cutoff':'2025-12-31','decision_rule':'pred_return>0','future_calendar':'official template headers only'})
    quality=dict(rows=4200,stocks=50,dates=84,duplicate_keys=0,missing_predictions=0,
                 c10_daily_rows=int(pred.route.eq('C10-daily').sum()),c21_frozen_rows=int(pred.route.eq('C21-frozen').sum()),
                 snapshot_dates=65,cutoff=str(cutoff.date()),frozen_origin=str(future.min().date()),
                 max_stock_age_at_forecast_days=int(pred.stock_source_age_days.max()),max_vix_age_at_forecast_days=int(pred.vix_source_age_days.max()),
                 labels_2026_read=False,training_2026_used=False,local_2026_score_computed=False,
                 publication_timestamp_verified=False,warning='Institution timing is previous-row approximation; adjusted-history vintage not verified. Frozen source age grows intentionally with horizon.')
    save(output/'data_quality_report.json',quality)
    manifest=dict(status='offline_draft_pending_user_approval',models=20,fit_count=22,features=features,params=params,
                  final_training_period=[str(data.date.min().date()),str(data.date.max().date())],
                  official_score=None,evaluation_score=None,template_sha256=sha(template_path),data_sha256=sha(zip_path),
                  submission_sha256=sha(submission),routing='64 dates daily C10;20 dates frozen C21 from2026-09-01 origin',
                  files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},
                  competition_submission=False,baseline_results_inherited=False)
    save(output/'manifest.json',manifest)
    return manifest


def verify_package(run_dir,zip_path,template_path,features):
    """Replay every saved delivery prediction; no outcome columns are read."""
    output=run_dir/'delivery'
    manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        if sha(output/name)!=digest:raise ValueError(f'Changed delivery artifact: {name}')
    info,template,dates=info_and_template(zip_path,template_path)
    p=pd.read_csv(output/'predictions_long.csv',dtype={'stock_id':str},
                  parse_dates=['date','origin_date','stock_source_date','vix_source_date'],float_precision='round_trip')
    assert not {'actual_y','actual_return','y'}&set(p.columns)
    check_static_keys(p,info['Top50@Y'],dates)
    assert p.pred_probability.isna().all() and p.pred_label.eq((p.pred_return>0).astype(int)).all()
    history,world,stocks=load_history(zip_path)
    snapshot=build_asof(history,world,'2026-09-01',stocks).sort_values('stock_id').reset_index(drop=True)
    models={h:lgb.Booster(model_file=str(output/'models'/f'h{h:02d}.txt')) for h in range(1,21)}
    for m in models.values():assert m.feature_name()==features and m.num_trees()==200
    max_error=0.
    for date,g in p.groupby('date',sort=True):
        g=g.sort_values('stock_id').reset_index(drop=True)
        if date<pd.Timestamp('2026-09-01'):
            frame=build_asof(history,world,date,stocks).sort_values('stock_id').reset_index(drop=True)
            h=1;assert g.route.eq('C10-daily').all() and g.origin_date.eq(date).all()
        else:
            frame=snapshot;h=int(dates[dates>pd.Timestamp('2026-08-31')].get_loc(date))+1
            assert g.route.eq('C21-frozen').all() and g.origin_date.eq(pd.Timestamp('2026-09-01')).all() and g.horizon.eq(h).all()
        assert g.stock_id.tolist()==frame.stock_id.tolist()
        np.testing.assert_array_equal(g.stock_source_date,frame.stock_source_date)
        np.testing.assert_array_equal(g.vix_source_date,frame.vix_source_date)
        error=float(np.max(np.abs(models[h].predict(frame[features])-g.pred_return.to_numpy())))
        assert error==0.;max_error=max(max_error,error)
    catalog=json.loads((output/'model_catalog.json').read_text(encoding='utf-8'))
    assert len(catalog)==20 and all(pd.Timestamp(x['training_end'])<pd.Timestamp('2026-01-01') for x in catalog)
    assert catalog[0]['repeat_max_error']==0. and catalog[1]['repeat_max_error']==0.
    result=pd.read_csv(output/'static_submission.csv',dtype={template.columns[0]:str})
    assert result.shape==(50,85) and result.columns.tolist()==template.columns.tolist()
    assert result.iloc[:,0].tolist()==stocks
    for date in template.columns[1:]:
        expected=p[p.date.eq(pd.Timestamp(date))].set_index('stock_id').pred_label.reindex(stocks)
        np.testing.assert_array_equal(result[date],expected)
    assert manifest['official_score'] is None and manifest['evaluation_score'] is None
    return dict(status='passed',models=20,predictions=4200,source_snapshots=65,max_native_csv_replay_error=max_error,
                seed_repeat_checks=2,template_order_verified=True,labels_2026_read=False,local_2026_score_computed=False)
