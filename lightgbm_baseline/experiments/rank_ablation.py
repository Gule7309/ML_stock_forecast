"""Locked single-rank ablations followed by a separate offline candidate draft."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error

from lightgbm_baseline.evaluate import average_return
from .next_stage import ROOT,BASE,protected_files,assert_protected,append_registry
from .static_candidate import (RANK_FEATURES,sha,save,keys_sha,fit_saved,read_training,rank_frame,
    info_and_template,load_history,build_asof,check_static_keys,format_predictions)

FAMILIES=['C27-RANK-RETURN','C28-RANK-FOREIGN5','C29-RANK-TRUST5']
FULL='C26-XSRANK__STATIC20261103_V1'


def read_predictions(identifier):
    return pd.read_csv(ROOT/'results'/identifier/'predictions.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip').sort_values(['date','stock_id']).reset_index(drop=True)


def select_candidate(contrasts,ids,full=FULL):
    """A simplified model must beat FULL in BOTH years, then fixed tie-breaking."""
    eligible=[]
    for name in ids:
        c=next(x for x in contrasts if x['candidate']==name and x['control']==full)
        if c['delta_ar']>0 and all(v>0 for v in c['annual_delta'].values()):eligible.append(c)
    eligible.sort(key=lambda c:(-min(c['annual_delta'].values()),-c['delta_ar'],c['candidate']))
    return eligible[0]['candidate'] if eligible else full


def contrast_family(pairs):
    matrices=[];results=[]
    for candidate,control,a,b in pairs:
        if a.duplicated(['date','stock_id','fold']).any() or b.duplicated(['date','stock_id','fold']).any():
            raise ValueError('Duplicate comparison key')
        p=a.merge(b,on=['date','stock_id','fold'],suffixes=('_a','_b'),validate='one_to_one')
        if len(p)!=len(a) or len(p)!=len(b):raise ValueError('Mismatched comparison coverage')
        if not np.array_equal(p.actual_return_a.to_numpy(),p.actual_return_b.to_numpy()):raise ValueError('Mismatched realized returns')
        p['delta']=p.signed_return_a-p.signed_return_b;p['month']=p.date.dt.to_period('M')
        m=p.groupby(['fold','month']).agg(total=('delta','sum'),n=('delta','size')).sort_index()
        if matrices:pd.testing.assert_frame_equal(m[['n']],matrices[0][['n']])
        matrices.append(m)
        concentration=[]
        for fold,g in p.groupby('fold'):
            by_stock=g.groupby('stock_id').delta.agg(['sum','size'])
            top=by_stock.nlargest(3,'sum').index.tolist()
            remainder=g[~g.stock_id.isin(top)]
            concentration.append(dict(fold=int(fold),positive_delta_stocks=int(by_stock['sum'].gt(0).sum()),
                posthoc_top3=top,delta_without_top3=float(remainder.delta.mean()) if len(remainder) else None))
        results.append(dict(candidate=candidate,control=control,n=len(p),delta_ar=float(p.delta.mean()),
            annual_delta={str(f):float(g.delta.mean()) for f,g in p.groupby('fold')},months_improved=int(m.total.gt(0).sum()),concentration=concentration))
    rng=np.random.default_rng(42);boot=[];estimates=np.array([r['delta_ar'] for r in results])
    for _ in range(1000):
        total=np.zeros(len(matrices));count=0
        for fold in [1,2]:
            groups=[m.xs(fold,level=0).to_numpy() for m in matrices]
            idx=rng.integers(0,len(groups[0]),len(groups[0]));count+=groups[0][idx,1].sum()
            for k,g in enumerate(groups):total[k]+=g[idx,0].sum()
        boot.append(total/count)
    boot=np.array(boot)
    width=float(np.quantile(np.abs(boot-estimates).max(axis=1),.95))
    for k,r in enumerate(results):
        r['bootstrap95']=[float(v) for v in np.quantile(boot[:,k],[.025,.975])]
        r['simultaneous95']=[float(estimates[k]-width),float(estimates[k]+width)]
    return dict(contrasts=results,method='Whole-month paired clusters within each year;1000 common draws;seed42;centered max-absolute-deviation simultaneous intervals across6 preregistered contrasts',
                limitation='Exploratory2023/2024 development;not adjusted for prior adaptive history;top3-stock exclusion is diagnostic only.')


def prepare(args):
    if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):raise ValueError('Invalid run ID')
    source=ROOT/'next_stage/STATIC20261103_V1';run=ROOT/'next_stage'/args.run_id
    base=json.loads((source/'run_config.json').read_text(encoding='utf-8'))
    if sha(args.data_zip)!=base['data_sha256'] or sha(args.template)!=base['template_sha256']:raise ValueError('Data/template version changed')
    audit=json.loads((source/'audit_complete.json').read_text(encoding='utf-8'))
    assert sha(source/'canonical_top50_through2025.joblib')==audit['canonical_sha256']
    old_manifest=json.loads((source/'delivery/manifest.json').read_text(encoding='utf-8'))
    for name,digest in old_manifest['files'].items():assert sha(source/'delivery'/name)==digest
    if args.stage=='package':
        config=json.loads((run/'config.json').read_text(encoding='utf-8'))
        if not (run/'research_complete.json').exists():raise ValueError('Research must finish before candidate packaging')
        assert config['driver_sha256']==sha(__file__)
        return run,source,config
    if run.exists():raise FileExistsError(f'Refusing overwrite: {run}')
    ids=[f+'__'+args.run_id for f in FAMILIES]
    for name in ids:
        for p in [ROOT/'results'/name,ROOT/'models'/name,ROOT/'configs'/f'{name}.json']:
            if p.exists():raise FileExistsError(p)
    # Protect the previous competition draft as well as baseline/research artifacts.
    protected=protected_files()
    protected.update({str(p):sha(p) for p in source.rglob('*') if p.is_file()})
    run.mkdir(exist_ok=False)
    config={**base,'run_id':args.run_id,'source_run':str(source),'data_zip':str(Path(args.data_zip).resolve()),'template':str(Path(args.template).resolve()),
        'experiment_ids':ids,'driver_sha256':sha(__file__),'rank_cache_sha256':sha(source/'top50_past_only_ranks.joblib'),
        'training_cache_sha256':sha(source/'canonical_top50_through2025.joblib'),'fit_budget':11,'official_score':None,
        'resources':dict(device='local CPU',n_jobs=4,sequential=True,total_fit_budget=11)}
    save(run/'config.json',config);shutil.copyfile(__file__,run/'driver_source.py')
    save(run/'preregistration.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypotheses={f:dict(Hypothesis='This single past-only rank may preserve useful relative information without the other two ranks.',
            Reason='C26 improved both years but its3-rank contribution has not been isolated;2023 incremental gains remained stock-concentrated.',
            Baseline='C10-MINCHILD200',Change='Add only '+rank,SecondaryControl=FULL) for f,rank in zip(FAMILIES,RANK_FEATURES)},
        parameters=base['params'],features=base['features'],seed=42,research_years=[2023,2024],
        fit_budget='9 research fits:3 candidates x2 folds plus fold1 repeat each;2 final fits:one selected h1 model plus a seed repeat. No additional model search.',
        validation='All24049 original stock/date keys,same training universes as C10/C26; ranking source universe is Top50 before labels.',
        selection='A single-rank model must beat full C26 in BOTH years and pooled. If multiple qualify, maximize minimum annual delta,then pooled delta,then lexicographic ID. Otherwise retain full C26. Record choice before2026 inputs.',
        inference='Create a separate candidate only;selected h1 replaces daily64 dates and September h1. Reuse unchanged original C21 h2..20. No untested transfer of rank features to later horizons.',
        boundaries='2025 only final fit.2026 input history allowed;2026 y/local scores forbidden. No external data,weights,competition submission,push or publication.',
        uncertainty='Six fixed contrasts:each single rank versus C10 and versus C26;month-cluster shared bootstrap and simultaneous intervals. Not independent confirmation.',
        protected_before=protected))
    return run,source,config


def research(run,source,config):
    if (run/'research_started.json').exists():raise FileExistsError('Research already started')
    save(run/'research_started.json',dict(utc=datetime.now(timezone.utc).isoformat()))
    data=joblib.load(source/'canonical_top50_through2025.joblib');data=data[data.date<'2025-01-01'].copy()
    ranks=joblib.load(source/'top50_past_only_ranks.joblib')
    frame=data.merge(ranks,on=['date','stock_id'],validate='one_to_one')
    baseline=read_predictions('C10-MINCHILD200');full=read_predictions(FULL)
    original_baseline=pd.read_csv(BASE/'baseline_predictions.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip')
    pd.testing.assert_frame_equal(baseline[['date','stock_id']],full[['date','stock_id']])
    params=config['params'];features=config['features'];predictions={};metrics=[]
    for family,rank,name in zip(FAMILIES,RANK_FEATURES,config['experiment_ids']):
        names=features+[rank];output=ROOT/'results'/name;modeldir=ROOT/'models'/name
        output.mkdir(exist_ok=False);modeldir.mkdir(exist_ok=False)
        conf=dict(experiment_id=name,family=family,parent_experiment='C10-MINCHILD200',secondary_control=FULL,
            features=names,params=params,seed=42,versions=config['versions'],data_sha256=config['data_sha256'],
            driver_sha256=config['driver_sha256'],git_commit=config['git_commit'],registration=str(run/'preregistration.json'),
            confirmation_used=False,holdout_used=False)
        save(output/'config.json',conf);save(ROOT/'configs'/f'{name}.json',conf)
        pieces=[];rows=[]
        for fold,year in enumerate([2023,2024],1):
            train=frame[frame.date<pd.Timestamp(f'{year}-01-01')].sort_values(['date','stock_id']).reset_index(drop=True)
            keys=baseline.loc[baseline.fold==fold,['date','stock_id']]
            val=frame.merge(keys,on=['date','stock_id'],validate='one_to_one').sort_values(['date','stock_id']).reset_index(drop=True)
            if not np.isfinite(train[names]).all().all() or not np.isfinite(val[names]).all().all():raise ValueError('Rank coverage changed;do not silently alter samples')
            expected=pd.read_csv(ROOT/'results/C10-MINCHILD200'/f'training_universe_fold_{fold}.csv',dtype={'stock_id':str},parse_dates=['date'])
            pd.testing.assert_frame_equal(train[['date','stock_id']],expected)
            print(f'{family} fold={fold} train={len(train)} val={len(val)}',flush=True)
            model,check=fit_saved(train,names,train.actual_return,params,modeldir/f'fold_{fold}.txt',val,repeat=fold==1)
            save(output/f'fold_{fold}_metadata.json',{**check,'training_start':str(train.date.min().date()),'training_end':str(train.date.max().date())})
            train[['date','stock_id']].to_csv(output/f'training_universe_fold_{fold}.csv',index=False)
            pd.DataFrame({'feature':names,'gain':model.feature_importance('gain')}).to_csv(output/f'gain_fold_{fold}.csv',index=False)
            forecast=model.predict(val[names]);label=(forecast>0).astype(int)
            p=val[['date','stock_id','actual_return']].copy();p['fold']=fold;p['experiment_id']=name;p['actual_y']=val.y
            p['pred_probability']=np.nan;p['pred_return']=forecast;p['pred_label']=label;p['signed_return']=(2*label-1)*val.actual_return
            p['stock_source_date']=val.stock_source_date;p['vix_source_date']=val.vix_source_date;p['rank_source_date']=val.rank_source_date
            pieces.append(p)
            ref=baseline[baseline.fold==fold];reference_full=full[full.fold==fold];ar=average_return(label,val.actual_return)
            original=original_baseline.merge(keys,on=['date','stock_id'],validate='one_to_one')
            rows.append(dict(experiment_id=name,parent_experiment='C10-MINCHILD200',feature_set=family,objective='regression',params=json.dumps(params),seed=42,fold=fold,
                train_period=f'{train.date.min().date()}:{train.date.max().date()}',val_period=f'{val.date.min().date()}:{val.date.max().date()}',
                n_train=len(train),n_val=len(val),ar=ar,delta_ar=ar-original.signed_return.mean(),direct_delta_ar=ar-ref.signed_return.mean(),secondary_delta_ar=ar-reference_full.signed_return.mean(),
                accuracy=float(accuracy_score(val.y,label)),log_loss=None,mse=float(mean_squared_error(val.actual_return,forecast)),mae=float(mean_absolute_error(val.actual_return,forecast)),
                n_stocks=50,n_dates=int(val.date.nunique()),pred_up_ratio=float(label.mean()),status='completed'))
        p=pd.concat(pieces,ignore_index=True).sort_values(['date','stock_id']).reset_index(drop=True)
        assert len(p)==24049 and not p.duplicated(['date','stock_id']).any()
        p.to_csv(output/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
        reread=read_predictions(name);np.testing.assert_array_equal(reread.pred_return,p.pred_return)
        predictions[name]=reread;metrics.extend(rows)
    pairs=[]
    for name,p in predictions.items():
        pairs.extend([(name,'C10-MINCHILD200',p,baseline),(name,FULL,p,full)])
    family=contrast_family(pairs);save(run/'comparisons.json',family)
    selected=select_candidate(family['contrasts'],config['experiment_ids'])
    selected_features=json.loads((ROOT/'results'/selected/'config.json').read_text(encoding='utf-8'))['features']
    save(run/'selection.json',dict(selected_experiment=selected,features=selected_features,selection_rule='Both annual increments vs full C26 positive;fixed tie-break;otherwise C26',
        selected_utc=datetime.now(timezone.utc).isoformat(),before_2026_inference=True,official_score=None))
    report(run,config,predictions,baseline,full,metrics,family,selected)
    assert_protected(json.loads((run/'preregistration.json').read_text(encoding='utf-8'))['protected_before'])
    append_registry(metrics,run)
    save(run/'research_complete.json',dict(status='passed',new_model_fits=9,n_validation_rows=24049,selected=selected))


def ranked_asof(history,world,date,stocks):
    if 'y' in history or 'actual_return' in history:raise ValueError('Inference must be label-free')
    date=pd.Timestamp(date);past=history[history.date<date].copy()
    frame=build_asof(past,world[world.date<date],date,stocks)
    ranks=rank_frame(past,stocks,[date])
    out=frame.merge(ranks,on=['date','stock_id'],validate='one_to_one')
    if len(out)!=len(stocks) or not np.isfinite(out[RANK_FEATURES]).all().all():raise ValueError('Incomplete rank snapshot')
    assert (out.rank_source_date<date).all()
    return out


def make_candidate(run,source,config):
    if (run/'package_started.json').exists():raise FileExistsError('Candidate already started')
    save(run/'package_started.json',dict(utc=datetime.now(timezone.utc).isoformat()))
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    names=selection['features'];base_features=config['features'];output=run/'candidate';output.mkdir(exist_ok=False)
    models=output/'models';models.mkdir()
    data=joblib.load(source/'canonical_top50_through2025.joblib')
    info,template,dates=info_and_template(config['data_zip'],config['template'])
    raw,_,_=read_training(config['data_zip'],info['Top50@Y'],'2025-12-31')
    label_free=raw.drop(columns='y');ranks=rank_frame(label_free,info['Top50@Y'],label_free.date.unique())
    old_ranks=joblib.load(source/'top50_past_only_ranks.joblib')
    earlier=ranks[ranks.date<'2025-01-01'].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier,old_ranks,check_exact=True)
    train=data.merge(ranks,on=['date','stock_id'],validate='one_to_one').sort_values(['date','stock_id']).reset_index(drop=True)
    assert len(train)==71874 and np.isfinite(train[names]).all().all() and train.date.max()<pd.Timestamp('2026-01-01')
    history,world,stocks=load_history(config['data_zip'])
    snapshot=ranked_asof(history,world,'2026-09-01',stocks)
    model,check=fit_saved(train,names,train.actual_return,config['params'],models/'h01.txt',snapshot,repeat=True)
    for h in range(2,21):
        origin=source/'delivery/models'/f'h{h:02d}.txt';target=models/origin.name
        shutil.copyfile(origin,target);assert sha(origin)==sha(target)
    shutil.copyfile(source/'delivery/model_catalog.json',output/'original_C21_catalog.json')
    rows=[]
    for date in dates[dates<=pd.Timestamp('2026-08-31')]:
        frame=ranked_asof(history,world,date,stocks)
        forecast=model.predict(frame[names])
        p=frame[['date','stock_id','stock_source_date','vix_source_date','rank_source_date']].copy()
        p['origin_date']=date;p['horizon']=1;p['route']='selected-daily';p['model']='models/h01.txt'
        p['pred_return']=forecast;p['pred_probability']=np.nan;p['pred_label']=(forecast>0).astype(int);rows.append(p)
    future=dates[dates>pd.Timestamp('2026-08-31')]
    for h,date in enumerate(future,1):
        booster=model if h==1 else lgb.Booster(model_file=str(models/f'h{h:02d}.txt'))
        features=names if h==1 else base_features
        assert booster.feature_name()==features
        forecast=booster.predict(snapshot[features])
        cols=['stock_id','stock_source_date','vix_source_date']+(['rank_source_date'] if h==1 else [])
        p=snapshot[cols].copy();p['date']=date;p['origin_date']=pd.Timestamp('2026-09-01');p['horizon']=h
        p['route']='selected-frozen-h1' if h==1 else 'original-C21-frozen';p['model']=f'models/h{h:02d}.txt'
        p['pred_return']=forecast;p['pred_probability']=np.nan;p['pred_label']=(forecast>0).astype(int);rows.append(p)
    predictions=pd.concat(rows,ignore_index=True).sort_values(['date','stock_id']).reset_index(drop=True)
    check_static_keys(predictions,stocks,dates)
    assert len(predictions)==4200 and (predictions.stock_source_date<predictions.origin_date).all() and (predictions.vix_source_date<predictions.origin_date).all()
    predictions.to_csv(output/'predictions_long.csv',index=False,float_format='%.17g')
    format_predictions(config['template'],output/'predictions_long.csv',output/'static_submission.csv')
    original=pd.read_csv(source/'delivery/predictions_long.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip')
    a=predictions[predictions.route=='original-C21-frozen'].sort_values(['date','stock_id'])
    b=original[original.date>pd.Timestamp('2026-09-01')].sort_values(['date','stock_id'])
    np.testing.assert_array_equal(a.pred_return,b.pred_return)
    joined=predictions.merge(original,on=['date','stock_id'],validate='one_to_one',suffixes=('_new','_old'))
    manifest=dict(status='additional_offline_candidate_pending_approval',selected_experiment=selection['selected_experiment'],
        h1_features=names,later_horizon_features=base_features,params=config['params'],training_end='2025-12-31',n_train=len(train),
        new_final_model_fits=2,reused_models=19,total_models=20,h1_verification=check,
        routing='Selected h1 model for64 daily dates plus Sep1;original C21 h2..20 reused without rank features.',
        data_sha256=config['data_sha256'],template_sha256=config['template_sha256'],versions=config['versions'],driver_sha256=config['driver_sha256'],
        predictions=4200,changed_labels_vs_original=int(joined.pred_label_new.ne(joined.pred_label_old).sum()),
        unchanged_later_horizon_predictions=950,official_score=None,evaluation_score=None,labels_2026_read=False,
        replaces_previous_draft=False,competition_submission=False,submission_sha256=sha(output/'static_submission.csv'),
        files={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    save(output/'manifest.json',manifest)
    verify_candidate(run,source,config)
    save(run/'package_complete.json',dict(status='passed',new_model_fits=2,official_score=None))


def verify_candidate(run,source,config):
    output=run/'candidate';manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():assert sha(output/name)==digest
    p=pd.read_csv(output/'predictions_long.csv',dtype={'stock_id':str},parse_dates=['date','origin_date','stock_source_date','vix_source_date','rank_source_date'],float_precision='round_trip')
    assert not {'y','actual_y','actual_return'}&set(p.columns)
    assert p.pred_probability.isna().all() and p.pred_label.eq((p.pred_return>0).astype(int)).all()
    assert (p.stock_source_date<p.origin_date).all() and (p.vix_source_date<p.origin_date).all()
    info,template,dates=info_and_template(config['data_zip'],config['template']);check_static_keys(p,info['Top50@Y'],dates)
    history,world,stocks=load_history(config['data_zip']);snapshot=ranked_asof(history,world,'2026-09-01',stocks)
    models={h:lgb.Booster(model_file=str(output/'models'/f'h{h:02d}.txt')) for h in range(1,21)}
    for h in range(2,21):assert sha(output/'models'/f'h{h:02d}.txt')==sha(source/'delivery/models'/f'h{h:02d}.txt')
    for date,g in p.groupby('date',sort=True):
        g=g.sort_values('stock_id').reset_index(drop=True)
        if date<pd.Timestamp('2026-09-01'):frame=ranked_asof(history,world,date,stocks);h=1
        else:frame=snapshot;h=int(dates[dates>pd.Timestamp('2026-08-31')].get_loc(date))+1
        frame=frame.sort_values('stock_id').reset_index(drop=True)
        features=manifest['h1_features'] if h==1 else manifest['later_horizon_features']
        assert models[h].feature_name()==features and models[h].num_trees()==200
        np.testing.assert_array_equal(g.stock_source_date,frame.stock_source_date)
        np.testing.assert_array_equal(g.vix_source_date,frame.vix_source_date)
        if h==1:np.testing.assert_array_equal(g.rank_source_date,frame.rank_source_date)
        np.testing.assert_array_equal(models[h].predict(frame[features]),g.pred_return)
    wide=pd.read_csv(output/'static_submission.csv',dtype={template.columns[0]:str})
    assert wide.shape==(50,85) and wide.columns.tolist()==template.columns.tolist() and wide.iloc[:,0].tolist()==stocks
    for date in template.columns[1:]:np.testing.assert_array_equal(wide[date],p[p.date==pd.Timestamp(date)].set_index('stock_id').pred_label.reindex(stocks))
    save(run/'candidate_verification.json',dict(status='passed',predictions=4200,native_csv_replay_error=0.,seed_repeat_error=manifest['h1_verification']['repeat_max_error'],
        original_models_reused_exactly=19,labels_2026_read=False,local_2026_score_computed=False))


def report(run,config,predictions,baseline,full,metrics,comparison,selected):
    blocks=[dict(style='HEADING_1',text='C27–C29：C26排名特徵的單一增量研究（2026-11-04）'),
        dict(style='NORMAL_TEXT',text='沿用既有官方資料與C10固定參數，先鎖定三個單一rank、兩個對照與六個統計對比；9次研究fit。2025只供後續最終重訓，2026不讀y或本地評分。原Static草稿保留。')]
    rows=[]
    for name,p in predictions.items():
        family=name.split('__')[0];rank=RANK_FEATURES[FAMILIES.index(family)]
        c=next(x for x in comparison['contrasts'] if x['candidate']==name and x['control']=='C10-MINCHILD200')
        f=next(x for x in comparison['contrasts'] if x['candidate']==name and x['control']==FULL)
        m=[x for x in metrics if x['experiment_id']==name];ar=float(p.signed_return.mean())
        status='符合簡化取代門檻' if all(v>0 for v in f['annual_delta'].values()) and f['delta_ar']>0 else '未能兩年皆超過完整C26'
        row=dict(experiment_id=name,family=family,rank_feature=rank,ar_2023_bp=m[0]['ar']*10000,ar_2024_bp=m[1]['ar']*10000,
            pooled_ar=ar,pooled_percent=ar*100,pooled_bp=ar*10000,fold_mean_bp=float(p.groupby('fold').signed_return.mean().mean()*10000),
            daily_equal_bp=float(p.groupby('date').signed_return.mean().mean()*10000),delta_vs_C10_bp=c['delta_ar']*10000,
            delta_vs_C26_bp=f['delta_ar']*10000,accuracy=float((p.pred_label==p.actual_y).mean()),conclusion=status)
        rows.append(row)
        record=dict(Hypothesis='The single rank '+rank+' may provide useful relative signal without the other two.',
            Reason='C26 group improved development AR but its components and interaction were not isolated.',Baseline='C10-MINCHILD200',
            Change='Add one rank only;secondary comparison against full C26.',Parameters=config['params'],Validation='2023/2024;24049 matched rows;no2025 retest',
            AR=row,DeltaAR={'vs_C10':c,'vs_C26':f},Conclusion=status,Next='No automatic pairwise combinations or horizon-specific policy selection.')
        save(ROOT/'results'/name/'research_record.json',record)
        blocks.extend([
            dict(style='HEADING_2',text=family+'：'+status),
            dict(style='NORMAL_TEXT',text='Hypothesis：單獨加入'+rank+'可能保留有效相對訊號。提出理由：C26三個排名整組有增益，但尚未拆開確認是哪個訊號、是否需要交互作用。'),
            dict(style='NORMAL_TEXT',text='Baseline：C10；直接簡化對照為完整C26。Change：只新增一個rank，共41個模型輸入。Parameters：LR=.05、trees200、leaves15、min_child_samples200、L2=1、seed42；同原設定。'),
            dict(style='NORMAL_TEXT',text='Validation：2023/2024，訓練及驗證鍵與C10/C26相同，24049筆；rank母體在接標籤之前固定，2025不重測。'),
            dict(style='NORMAL_TEXT',text=f"AR：2023={row['ar_2023_bp']:.6f}bp；2024={row['ar_2024_bp']:.6f}bp；Pooled={ar:.12f}（{ar*100:.9f}%，{ar*10000:.6f}bp）；Fold-mean={row['fold_mean_bp']:.6f}bp；Daily-equal={row['daily_equal_bp']:.6f}bp。"),
            dict(style='NORMAL_TEXT',text=f"ΔAR：vs C10={row['delta_vs_C10_bp']:+.6f}bp；vs C26={row['delta_vs_C26_bp']:+.6f}bp；六對比同步95%區間vs C26=[{f['simultaneous95'][0]*10000:+.6f},{f['simultaneous95'][1]*10000:+.6f}]bp。Conclusion：{status}；Accuracy={row['accuracy']:.4%}。")])
    pd.DataFrame(rows).to_csv(run/'comparison.csv',index=False)
    blocks.extend([dict(style='HEADING_2',text='固定選擇與後續'),
        dict(style='NORMAL_TEXT',text='事前規則選中：'+selected+'。只有單一rank在兩年均勝過完整C26才簡化；否則保留完整C26。這是已多次使用的開發資料，區間與排行都不代表官方成績。'),
        dict(style='NORMAL_TEXT',text='另外建立獨立候選草稿：只把每日模型及9月h1改為選中配置，後續h2–20完全重用原C21；不把單日rank效益直接轉移至長距模型。原Static檔案與模型保留，不自動提交、Push或發布。')])
    save(run/'google_doc_payload.json',dict(blocks=blocks))
    text='\n\n'.join(('# '+b['text']) if b['style']=='HEADING_1' else ('## '+b['text']) if b['style']=='HEADING_2' else b['text'] for b in blocks)
    (run/'findings.md').write_text(text+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['research','package','all'],required=True)
    parser.add_argument('--data-zip',required=True);parser.add_argument('--template',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();run,source,config=prepare(args)
    if args.stage in ['research','all']:research(run,source,config)
    if args.stage in ['package','all']:make_candidate(run,source,config)
    assert_protected(json.loads((run/'preregistration.json').read_text(encoding='utf-8'))['protected_before'])
    save(run/'status.json',dict(research_complete=(run/'research_complete.json').exists(),candidate_complete=(run/'package_complete.json').exists(),
        original_artifacts_unchanged=True,official_score=None,competition_submission=False))
    print(str(run),flush=True)


if __name__=='__main__':main()
