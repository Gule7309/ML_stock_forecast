"""Small hypothesis-led cycles; no pretrained downloads or confirmation evaluation."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error

from lightgbm_baseline.config import FEATURES, PARAMS
from lightgbm_baseline.load_data import load_data
from lightgbm_baseline.features import build_features
from lightgbm_baseline.evaluate import average_return
from .feature_groups import augment
from .compare_experiments import ROOT, table

BASE=ROOT.parent/'results_official_v1_20261031_200052'
ZIP=Path('C:/Users/USER/Downloads/fin_pred_data.zip')
LAB=ROOT/'cycles'


def save(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare():
    config=json.loads((BASE/'baseline_config.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(ZIP.read_bytes()).hexdigest()==config['source_sha256']
    raw,glob,_=load_data(ZIP)
    raw=raw[pd.to_datetime(raw.date)<'2025-01-01'].copy()
    glob=glob[pd.to_datetime(glob.date)<'2025-01-01'].copy()
    data,_=build_features(raw,glob)
    data,extra,_=augment(data,ZIP,'FEAT-03+FEAT-01')
    data=data.dropna(subset=FEATURES+extra).sort_values(['date','stock_id']).reset_index(drop=True)
    price=['intraday','close_return','overnight','range','momentum_5','volatility_5','momentum_20','volatility_20']
    inst=[c for c in extra if c not in price]
    return data, FEATURES+inst, FEATURES+extra


def preregister(cycle,content):
    LAB.mkdir(exist_ok=True)
    path=LAB/cycle
    path.mkdir(exist_ok=False)
    content['registered_utc']=datetime.now(timezone.utc).isoformat()
    save(path/'preregistration.json',content)
    return path


def training_target(returns, winsor_quantile=None):
    """Fit a symmetric return cap using only the supplied training labels."""
    values=returns.copy()
    if not len(values) or not np.isfinite(values.to_numpy()).all():
        raise ValueError('Training returns must be nonempty and finite')
    if winsor_quantile is None:
        return values,None
    if not 0<winsor_quantile<1:
        raise ValueError('Winsor quantile must lie strictly between zero and one')
    cap=float(values.abs().quantile(winsor_quantile))
    if cap<=0:
        raise ValueError('Training return cap must be positive')
    return values.clip(-cap,cap),cap


def train_run(identifier,data,features,overrides=None,parent='OBJ-00',note='',winsor_quantile=None):
    output=ROOT/'results'/identifier
    output.mkdir(exist_ok=False)
    models=ROOT/'models'/identifier
    models.mkdir(exist_ok=False)
    params={**PARAMS,'objective':'regression',**(overrides or {})}
    git_result=subprocess.run(['git','rev-parse','--verify','HEAD'],capture_output=True,text=True)
    config=dict(experiment_id=identifier,parent_experiment=parent,features=features,params=params,
                note=note,seed=42,python=platform.python_version(),
                versions={k:importlib.metadata.version(k) for k in ['numpy','pandas','scikit-learn','lightgbm']},
                source_sha256=hashlib.sha256(ZIP.read_bytes()).hexdigest(),
                git_commit=git_result.stdout.strip() if git_result.returncode==0 else None,
                source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                folds=[2023,2024],confirmation_used=False,holdout_used=False,
                training_target=dict(name='actual_return' if winsor_quantile is None else 'symmetric_winsorized_actual_return',
                                     quantile=winsor_quantile,fit_scope='each outer training fold only',
                                     validation_return='original unmodified actual_return'))
    save(output/'config.json',config)
    save(ROOT/'configs'/f'{identifier}.json',config)
    ref=pd.read_csv(BASE/'baseline_predictions.csv',dtype={'stock_id':str},parse_dates=['date'])
    parent_file=ROOT/'results'/parent/'predictions.csv'
    parent_pred=pd.read_csv(parent_file,dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip') if parent_file.exists() else None
    rows=[];predictions=[]
    for fold,year in enumerate([2023,2024],1):
        train=data[data.date<pd.Timestamp(f'{year}-01-01')].copy()
        val=data[data.date.between(f'{year}-01-01',f'{year}-12-31')].copy()
        if 'upstream_pred_return' in features and f'outer_{year}_pred' in val:
            assert val[f'outer_{year}_pred'].notna().all()
            val['upstream_pred_return']=val[f'outer_{year}_pred']
        assert train.date.max()<val.date.min()
        r=ref[ref.fold==fold].merge(val[['date','stock_id']],on=['date','stock_id'],validate='one_to_one')
        assert len(r)==len(val)
        target,cap=training_target(train.actual_return,winsor_quantile)
        model=lgb.LGBMRegressor(**params)
        model.fit(train[features],target)
        pred_return=model.predict(val[features]);label=(pred_return>0).astype(int)
        p=val[['date','stock_id','actual_return']].copy()
        p['actual_y']=val.y;p['fold']=fold;p['experiment_id']=identifier
        p['pred_label']=label;p['pred_probability']=np.nan;p['pred_return']=pred_return
        p['signed_return']=(2*label-1)*val.actual_return
        ar=average_return(label,val.actual_return)
        direct_delta=None
        if parent_pred is not None:
            direct=parent_pred[parent_pred.fold==fold].merge(val[['date','stock_id']],on=['date','stock_id'],validate='one_to_one')
            assert len(direct)==len(val)
            direct_delta=ar-direct.signed_return.mean()
        rows.append(dict(experiment_id=identifier,parent_experiment=parent,feature_set='cycle-specific',
                         objective=params['objective'],params=json.dumps(params),seed=42,fold=fold,
                         train_period=f'{train.date.min().date()}:{train.date.max().date()}',val_period=f'{year}-01-01:{year}-12-31',
                         n_train=len(train),n_val=len(val),ar=ar,delta_ar=ar-r.signed_return.mean(),direct_delta_ar=direct_delta,
                         training_return_cap=cap,training_returns_clipped=int((train.actual_return!=target).sum()),
                         accuracy=accuracy_score(val.y,label),log_loss=None,status='completed',
                         mse=mean_squared_error(val.actual_return,pred_return),mae=mean_absolute_error(val.actual_return,pred_return)))
        model.booster_.save_model(str(models/f'fold_{fold}.txt'))
        pd.DataFrame({'feature':features,'gain':model.booster_.feature_importance(importance_type='gain')}).to_csv(output/f'gain_fold_{fold}.csv',index=False)
        train[['date','stock_id']].to_csv(output/f'training_universe_fold_{fold}.csv',index=False)
        predictions.append(p)
    pd.concat(predictions).to_csv(output/'predictions.csv',index=False,float_format='%.17g')
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    registry=ROOT/'experiment_registry.csv'
    pd.concat([pd.read_csv(registry),pd.DataFrame(rows)],ignore_index=True).to_csv(registry,index=False)
    return output


def paired(candidate,control):
    def read(name):
        return pd.read_csv(ROOT/'results'/name/'predictions.csv',dtype={'stock_id':str},parse_dates=['date'])
    a,b=read(candidate),read(control)
    x=a.merge(b,on=['date','stock_id','fold'],suffixes=('_a','_b'),validate='one_to_one')
    assert len(x)==len(a)==len(b)
    np.testing.assert_allclose(x.actual_return_a,x.actual_return_b,atol=1e-15,rtol=0)
    x['delta']=x.signed_return_a-x.signed_return_b
    annual={str(f):float(g.delta.mean()) for f,g in x.groupby('fold')}
    rng=np.random.default_rng(42); boot=[]
    groups=[g.groupby('date').agg(total=('delta','sum'),n=('delta','size')).to_numpy() for _,g in x.groupby('fold')]
    # Circular moving blocks within each year, preserving the cross-stock date cluster.
    for _ in range(1000):
        total=0.;count=0
        for z in groups:
            starts=rng.integers(0,len(z),size=int(np.ceil(len(z)/10)))
            indices=((starts[:,None]+np.arange(10))%len(z)).ravel()[:len(z)]
            total+=z[indices,0].sum();count+=z[indices,1].sum()
        boot.append(total/count)
    return dict(candidate=candidate,control=control,n=len(x),delta_ar=float(x.delta.mean()),
                annual_delta=annual,bootstrap_95_percentile=[float(v) for v in np.quantile(boot,[.025,.975])],
                method='Exploratory paired circular date-block bootstrap, block=10 trading dates, B=1000, seed=42, stratified by year; not adjusted for prior model selection')


def cycle1():
    out=preregister('C1',dict(hypotheses=[
        {'id':'H1','reason':'Price+institution improved pooled AR, but previous training masks differed.','test':'C1-PRICE versus C1-CONTROL, identical rows; require positive delta in both years for stability.'},
        {'id':'H2','reason':'Binary rate=.1 / trees=100 improved both development years. Transfer to institutional regression is untested.','test':'C1-LR versus C1-CONTROL on identical rows; no parameter grid.'}],
        fixed=dict(mask='complete price+institution rows',years=[2023,2024],seed=42),
        uncertainty='10-date blocks, 1000 bootstrap resamples, seed 42; exploratory, no significance claim after selection',
        next_rule='Select no new setting unless both annual deltas are positive; then investigate cheap forward-OOF linear sequence features before costly foundation models.'))
    data,inst,all_features=prepare()
    train_run('C1-CONTROL',data,inst,note='Institution only, complete price+institution training mask')
    train_run('C1-PRICE',data,all_features,parent='C1-CONTROL')
    train_run('C1-LR',data,inst,{'learning_rate':.1,'n_estimators':100},parent='C1-CONTROL')
    results=[paired(x,'C1-CONTROL') for x in ['C1-PRICE','C1-LR']]
    save(out/'results.json',results)
    print(json.dumps(results,indent=2))


def cycle2():
    prior=json.loads((LAB/'C1/results.json').read_text())
    stable=[r for r in prior if all(d>0 for d in r['annual_delta'].values())]
    selected=max(stable,key=lambda r:min(r['annual_delta'].values()))['candidate'] if stable else 'C1-CONTROL'
    out=preregister('C2',dict(hypothesis='Forward-OOF linear forecasts from 20 historical intraday returns add complementary information to LightGBM.',
        reason='C1 results now observed; use a cheap linear upstream as a control before considering pretrained Kronos or larger sequence networks.',
        selected_from_cycle1=selected,selection_rule='Both annual deltas positive, then maximum minimum annual delta; otherwise C1-CONTROL',
        evidence=['https://arxiv.org/abs/2205.13504','https://doi.org/10.1016/S0893-6080(05)80023-1'],
        scope='Inspired by linear-baseline and stacking literature; not a reproduction of DLinear and no claim those papers prove Taiwan intraday alpha.',
        upstream=dict(model='StandardScaler + Ridge(alpha=1.0)',lags=20,target='intraday actual_return',inner_blocks='6 calendar months',first_fit='before 2021-01-01',fit_stocks='official Top50 only'),
        tests=['C2-STACK versus C2-CONTROL on identical OOF rows','C2-RAWLAGS versus C2-CONTROL as raw-history control'],
        stop_rule='If stacking fails either annual increment or does not add beyond raw lags, do not automatically scale to more upstream models.',
        bootstrap=dict(block=10,resamples=1000,seed=42)))
    data,inst,all_features=prepare()
    selected_cfg=json.loads((ROOT/'results'/selected/'config.json').read_text())
    features=selected_cfg['features'];params=selected_cfg['params']
    raw,_,_=load_data(ZIP)
    raw['date']=pd.to_datetime(raw.date)
    raw=raw[raw.date<'2025-01-01'].sort_values(['stock_id','date'])
    raw['r']=(raw['etl:adj_close']-raw['etl:adj_open'])/raw['etl:adj_open']
    lag_names=[]
    for lag in range(1,21):
        name=f'upstream_intraday_lag_{lag}';lag_names.append(name)
        raw[name]=raw.groupby('stock_id').r.shift(lag)
    data=data.merge(raw[['stock_id','date',*lag_names]],on=['stock_id','date'],validate='one_to_one').dropna(subset=lag_names)
    data['upstream_pred_return']=np.nan
    data['upstream_fit_cutoff']=pd.NaT
    fitted=[]
    for start in pd.date_range('2021-01-01','2023-07-01',freq='6MS'):
        end=start+pd.DateOffset(months=6)
        train=data[data.date<start]
        target=(data.date>=start)&(data.date<end)
        assert len(train)>0 and train.date.max()<start
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        model.fit(train[lag_names],train.actual_return)
        joblib.dump(model,out/f'upstream_inner_{start.date()}.joblib')
        data.loc[target,'upstream_pred_return']=model.predict(data.loc[target,lag_names])
        data.loc[target,'upstream_fit_cutoff']=train.date.max()
        fitted.append(dict(start=str(start.date()),fit_cutoff=str(train.date.max().date()),n_train=len(train),n_output=int(target.sum())))
    history_all=data.copy()
    known=data.upstream_fit_cutoff.notna()
    assert (data.loc[known,'upstream_fit_cutoff']<data.loc[known,'date']).all()
    # 2023 and 2024 output blocks must use a single model fitted before the outer year,
    # even for the second half; no adaptation on that validation year's labels.
    for year in [2023,2024]:
        start=pd.Timestamp(f'{year}-01-01')
        history=data[(data.date<start)]
        # Reconstruct full pre-year data, including 2020 warmup, for the validation encoder.
        all_history=history_all[history_all.date<start]
        model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
        model.fit(all_history[lag_names],all_history.actual_return)
        joblib.dump(model,out/f'upstream_outer_{year}.joblib')
        target=data.date.dt.year==year
        # Save outer-validation values separately. Training OOF for 2024 must retain
        # genuine 2023 half-year OOF values, so do not overwrite training OOF in data.
        data[f'outer_{year}_pred']=np.nan
        data.loc[target,f'outer_{year}_pred']=model.predict(data.loc[target,lag_names])
        if year==2024:
            data.loc[target,'upstream_pred_return']=data.loc[target,f'outer_{year}_pred']
            data.loc[target,'upstream_fit_cutoff']=all_history.date.max()
    data=data.dropna(subset=['upstream_pred_return']).copy()
    assert (data.upstream_fit_cutoff<data.date).all()
    save(out/'upstream_oof_blocks.json',fitted)
    data.to_csv(out/'upstream_features.csv',index=False,float_format='%.17g')
    joblib.dump(data,out/'upstream_features_exact.joblib')
    overrides={k:v for k,v in params.items() if k!='objective'}
    train_run('C2-CONTROL',data,features,overrides,parent=selected)
    train_run('C2-RAWLAGS',data,features+lag_names,overrides,parent='C2-CONTROL')
    train_run('C2-STACK',data,features+['upstream_pred_return'],overrides,parent='C2-CONTROL',note='Forward half-year OOF upstream; outer validation fit before year start')
    results=[paired(x,'C2-CONTROL') for x in ['C2-RAWLAGS','C2-STACK']]+[paired('C2-STACK','C2-RAWLAGS')]
    save(out/'results.json',results)
    print(json.dumps(results,indent=2))


def cycle3():
    prior=json.loads((LAB/'C2/results.json').read_text())
    out=preregister('C3',dict(hypothesis='The raw-history gain may be captured by recent lags or low-dimensional distribution summaries rather than a trained linear upstream.',
        reason='C2 raw lags improved both years; learned linear forecast failed one year and lost to raw lags in both years. Do not scale the failed upstream automatically.',
        prior_results=prior,
        tests=['C3-LAG5: only last 5 historical intraday returns','C3-SUMMARY: latest return, mean and sample std of 20 historical returns'],
        controls=['C2-CONTROL','C2-RAWLAGS'],
        fixed='same exact training/validation rows and original regressor parameters as C2',
        stop_rule='This execution budget ends after two candidates; retain hypotheses, not declare a winner solely by highest development AR. No 2025/2026 evaluation.'))
    data=pd.read_csv(LAB/'C2/upstream_features.csv',dtype={'stock_id':str},parse_dates=['date'])
    cfg=json.loads((ROOT/'results/C2-CONTROL/config.json').read_text())
    features=cfg['features'];params={k:v for k,v in cfg['params'].items() if k!='objective'}
    lag_names=[f'upstream_intraday_lag_{i}' for i in range(1,21)]
    data['return_mean_20']=data[lag_names].mean(axis=1)
    data['return_std_20']=data[lag_names].std(axis=1,ddof=1)
    train_run('C3-LAG5',data,features+lag_names[:5],params,parent='C2-RAWLAGS')
    train_run('C3-SUMMARY',data,features+[lag_names[0],'return_mean_20','return_std_20'],params,parent='C2-RAWLAGS')
    results=[paired(a,b) for a in ['C3-LAG5','C3-SUMMARY'] for b in ['C2-CONTROL','C2-RAWLAGS']]
    save(out/'results.json',results)
    print(json.dumps(results,indent=2))


def precision_control():
    out=preregister('C3-PRECISION',dict(reason='Model replay found CSV decimal parsing changes tree routing for 14 validation predictions in C2-RAWLAGS fold 1. C3 had consumed that CSV, so add the same CSV input-path 20-lag control before interpreting C3 differences.',
        hypothesis='The qualitative raw-lag versus summary comparison should survive identical numeric serialization.',
        test='C3-RAW20-CSV with same training/validation rows and lag values as C3 candidates; compare both candidates to this control.',
        scope='Integrity correction, not a new parameter search; keep previous outputs'))
    data=pd.read_csv(LAB/'C2/upstream_features.csv',dtype={'stock_id':str},parse_dates=['date'])
    cfg=json.loads((ROOT/'results/C2-CONTROL/config.json').read_text())
    lags=[f'upstream_intraday_lag_{i}' for i in range(1,21)]
    train_run('C3-RAW20-CSV',data,cfg['features']+lags,{k:v for k,v in cfg['params'].items() if k!='objective'},parent='C2-RAWLAGS')
    results=[paired(a,'C3-RAW20-CSV') for a in ['C3-LAG5','C3-SUMMARY','C2-RAWLAGS']]
    save(out/'results.json',results)
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cycle',choices=['C1','C2','C3','C3-PRECISION'])
    a=parser.parse_args()
    {'C1':cycle1,'C2':cycle2,'C3':cycle3,'C3-PRECISION':precision_control}[a.cycle]()
