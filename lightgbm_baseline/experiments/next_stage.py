"""One offline entrypoint for an immutable Static draft and three locked studies."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,mean_squared_error,mean_absolute_error

from lightgbm_baseline.evaluate import average_return
from .static_candidate import (sha,save,keys_sha,info_and_template,read_training,canonical,
    rank_frame,RANK_FEATURES,rolling_train,fit_saved,package,verify_package,load_history,build_asof)

ROOT=Path(__file__).parent
BASE=ROOT.parent/'results_official_v1_20261031_200052'
FAMILIES=['C24-ROLL2Y','C25-TOP100','C26-XSRANK']


def protected_files():
    paths=[*BASE.rglob('*')]
    for folder in ['models','results','delivery','configs']:
        paths.extend((ROOT/folder).rglob('*'))
    return {str(p):sha(p) for p in paths if p.is_file()}


def assert_protected(snapshot):
    for path,digest in snapshot.items():
        if sha(path)!=digest:raise ValueError(f'Original artifact changed: {path}')


def hypotheses():
    return {
        FAMILIES[0]:dict(Hypothesis='Recent two-year training may adapt better than expanding history.',Reason='Recent market relationships may differ from older observations; three-year rolling is identical to the2023 control, so two years gives two informative comparisons.',Baseline='C10-MINCHILD200',Change='Only training origin dates:2021-2022 for2023;2022-2023 for2024.'),
        FAMILIES[1]:dict(Hypothesis='Additional official stocks may improve the shared return mapping.',Reason='The ZIP contains100 labelled stock series; all prior C10 training used only Top50. Transfer can help or introduce distribution mismatch.',Baseline='C10-MINCHILD200',Change='Train on Top100@Y, but evaluate only the original Top50; same40 inputs, no stock ID or weights.'),
        FAMILIES[2]:dict(Hypothesis='Relative cross-stock position may transfer better than absolute feature values.',Reason='Earlier institution ablations and C10 cross-stock contribution diagnostics motivate a small relative-feature group, rather than adding common-day context again.',Baseline='C10-MINCHILD200',Change='Add3 Top50 past-only average-percentile ranks: previous intraday return,foreign5-day flow/volume,trust5-day flow/volume. Rank before label join.')}


def prepare_run(args):
    if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):raise ValueError('Invalid run ID')
    info,template,dates=info_and_template(args.data_zip,args.template)
    parent=ROOT/'next_stage';parent.mkdir(exist_ok=True)
    ignore=parent/'.gitignore'
    if not ignore.exists():ignore.write_text('*\n!.gitignore\n',encoding='utf-8')
    run=parent/args.run_id
    config=json.loads((ROOT/'configs/C10-MINCHILD200.json').read_text(encoding='utf-8'))
    if sha(args.data_zip)!=config['source_sha256']:raise ValueError('Official ZIP differs from audited C10 version')
    request=dict(run_id=args.run_id,data_zip=str(Path(args.data_zip).resolve()),template=str(Path(args.template).resolve()),
                 data_sha256=sha(args.data_zip),template_sha256=sha(args.template))
    if run.exists():
        if args.stage=='all':raise FileExistsError(f'Run ID already exists: {run}')
        saved=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
        for k,v in request.items():
            if saved[k]!=v:raise ValueError(f'Cannot change an existing run: {k}')
        assert saved['code_sha256']=={p.name:sha(p) for p in [Path(__file__),ROOT/'static_candidate.py',ROOT/'asof_predict.py',ROOT/'submission_format.py']}
    else:
        run.mkdir(exist_ok=False)
        git=subprocess.run(['git','-C',str(ROOT.parent.parent),'rev-parse','--verify','HEAD'],capture_output=True,text=True)
        saved={**request,'created_utc':datetime.now(timezone.utc).isoformat(),'features':config['features'],'params':config['params'],
               'python':platform.python_version(),'versions':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm']},
               'git_commit':git.stdout.strip() if git.returncode==0 else None,
               'code_sha256':{p.name:sha(p) for p in [Path(__file__),ROOT/'static_candidate.py',ROOT/'asof_predict.py',ROOT/'submission_format.py']},
               'final_training_cutoff':'2025-12-31','research_years':[2023,2024],
               'scope':'2026 historical input allowed;2026 y and local scores forbidden;2025 final fit only,not candidate validation.',
               'resources':dict(device='local CPU',n_jobs=4,sequential=True,total_fit_budget=31),
               'official_score':None,'competition_upload_allowed':False}
        if saved['versions']['lightgbm']!='4.6.0':raise ValueError('Use the existing pinned LightGBM4.6.0 environment')
        save(run/'run_config.json',saved)
        sources=run/'sources';sources.mkdir()
        for p in [Path(__file__),ROOT/'static_candidate.py',ROOT/'asof_predict.py',ROOT/'submission_format.py']:
            shutil.copyfile(p,sources/p.name)
        save(run/'preregistration.json',dict(registered_utc=saved['created_utc'],hypotheses=hypotheses(),
            final_configuration='Fixed C10 daily64 dates plus C21 frozen20 dates;2020-2025 Top50;40 features;no research replacement.',
            research_budget='3 fixed candidates x2 folds +first-fold repeat each =9 fits; final20 models +2 repeats =22 fits.',
            rank_formula="groupby(date).rank(method='average',pct=True) over source-available Top50, before any target join",
            validation='Freeze original24049 keys intersected with source-available new ranks, then recompute every matched control on that same set.',
            decision='Positive direct delta both years and pooled; show three-contrast simultaneous month-cluster intervals. Crossing0 leaves uncertainty unresolved. No automatic combinations or follow-up search.',
            constraints=saved['scope'],protected_before=protected_files()))
    return run,saved,info,dates


def begin_stage(run,name):
    if (run/f'{name}_started.json').exists():raise FileExistsError(f'Stage already started: {name}; no silent overwrite')
    save(run/f'{name}_started.json',dict(utc=datetime.now(timezone.utc).isoformat()))


def audit(run,config,info,dates):
    begin_stage(run,'audit')
    raw,world,quality=read_training(config['data_zip'],info['Top50@Y'],'2025-12-31')
    data,dq=canonical(raw,world,config['features'])
    old=joblib.load(ROOT/'cycles/C4/canonical_features.joblib').sort_values(['date','stock_id']).reset_index(drop=True)
    earlier=data[data.date<'2025-01-01'].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier[['date','stock_id']],old[['date','stock_id']])
    np.testing.assert_array_equal(earlier[config['features']].to_numpy(),old[config['features']].to_numpy())
    np.testing.assert_array_equal(earlier.actual_return,old.actual_return)
    expected=read_control()
    for fold,year in enumerate([2023,2024],1):
        v=earlier[earlier.date.dt.year==year]
        ref=expected[expected.fold==fold]
        pd.testing.assert_frame_equal(v[['date','stock_id']].reset_index(drop=True),ref[['date','stock_id']].reset_index(drop=True))
        booster=lgb.Booster(model_file=str(ROOT/'models/C10-MINCHILD200'/f'fold_{fold}.txt'))
        np.testing.assert_array_equal(booster.predict(v[config['features']]),ref.pred_return)
    joblib.dump(data,run/'canonical_top50_through2025.joblib')
    # Labels never enter the rank builder. Calendar is obtained from raw date metadata.
    history=raw.drop(columns='y');history=history[history.date<'2025-01-01'].copy()
    ranks=rank_frame(history,info['Top50@Y'],history.date.unique())
    joblib.dump(ranks,run/'top50_past_only_ranks.joblib')
    history,world,stocks=load_history(config['data_zip'])
    cutoff=min([history.loc[history.stock_id==s,'date'].max() for s in stocks]+[world.date.max()])
    if cutoff!=pd.Timestamp('2026-08-31'):raise ValueError('Data cutoff changed')
    future=dates[dates>cutoff]
    checks=[]
    for date in list(dates[dates<=cutoff])+[future.min()]:
        frame=build_asof(history,world,date,stocks)
        assert len(frame)==50 and np.isfinite(frame[config['features']]).all().all()
        checks.append(dict(origin_date=str(date.date()),stocks=50,
            stock_max_age=int((date-frame.stock_source_date).dt.days.max()),vix_max_age=int((date-frame.vix_source_date).dt.days.max())))
    save(run/'audit_data_quality.json',dict(training_stock_reports=quality,canonical_quality=dq,
        historical_rebuild_rows=len(earlier),max_historical_feature_difference=0.,max_historical_prediction_difference=0.,
        final_training_rows=len(data),final_training_last_date=str(data.date.max().date()),
        label_reader='date-only pass,then y prefix nrows through2025',labels_2026_read=False,
        static_snapshot_checks=checks,all_65_snapshots_complete=len(checks)==65,
        limitation='Institutional publication time uses previous stock observation; corporate-adjustment vintages not verified.'))
    save(run/'audit_complete.json',dict(status='passed',canonical_sha256=sha(run/'canonical_top50_through2025.joblib'),replayed_predictions=24049))
    print('audit complete:59734 historical rows exact;71874 final rows;65 input snapshots valid',flush=True)


def read_control():
    return pd.read_csv(ROOT/'results/C10-MINCHILD200/predictions.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip').sort_values(['date','stock_id']).reset_index(drop=True)


def family_intervals(predictions,control):
    matrix=[];results=[]
    for name,p in predictions.items():
        x=p.merge(control,on=['date','stock_id','fold'],validate='one_to_one',suffixes=('_a','_b'))
        assert len(x)==len(p)==len(control)
        np.testing.assert_array_equal(x.actual_return_a,x.actual_return_b)
        x['delta']=x.signed_return_a-x.signed_return_b;x['month']=x.date.dt.to_period('M')
        month=x.groupby(['fold','month']).agg(total=('delta','sum'),n=('delta','size')).sort_index()
        if matrix:pd.testing.assert_frame_equal(month[['n']],matrix[0][['n']])
        matrix.append(month)
        result=dict(experiment_id=name,delta_ar=float(x.delta.mean()),annual_delta={str(f):float(g.delta.mean()) for f,g in x.groupby('fold')},months_improved=int(month.total.gt(0).sum()))
        stock=x.groupby(['fold','stock_id']).delta.agg(['sum','size']).reset_index()
        concentration=[]
        for fold,g in stock.groupby('fold'):
            top=g.nlargest(3,'sum').stock_id.tolist();remaining=x[(x.fold==fold)&~x.stock_id.isin(top)]
            concentration.append(dict(fold=int(fold),positive_delta_stocks=int(g['sum'].gt(0).sum()),stocks=len(g),
                posthoc_excluded_top3_increment_stocks=top,delta_without_top3=float(remaining.delta.mean())))
        result['concentration']=concentration;results.append(result)
    rng=np.random.default_rng(42);boot=[]
    estimates=np.array([r['delta_ar'] for r in results])
    for _ in range(1000):
        sums=np.zeros(len(matrix));count=0
        for fold in [1,2]:
            groups=[m.xs(fold,level=0).to_numpy() for m in matrix]
            idx=rng.integers(0,len(groups[0]),len(groups[0]));count+=groups[0][idx,1].sum()
            for j,g in enumerate(groups):sums[j]+=g[idx,0].sum()
        boot.append(sums/count)
    boot=np.array(boot);width=float(np.quantile(np.abs(boot-estimates).max(axis=1),.95))
    for j,r in enumerate(results):
        r['bootstrap95']=[float(v) for v in np.quantile(boot[:,j],[.025,.975])]
        r['simultaneous95']=[float(estimates[j]-width),float(estimates[j]+width)]
        r['nominate']=bool(all(v>0 for v in r['annual_delta'].values()) and r['delta_ar']>0)
    return dict(comparisons=results,method='Paired whole-month clusters within year,B1000,seed42; centered max-absolute-deviation simultaneous intervals over completed candidates',
        limitation='Exploratory;2023/2024 reused. No correction for earlier adaptive choices. Top3 exclusion is a post-hoc sensitivity diagnostic, not a deployed stock filter.')


def append_registry(rows,run):
    path=ROOT/'experiment_registry.csv';before=path.read_bytes()
    with path.open(encoding='utf-8-sig',newline='') as f:fields=next(csv.reader(f))
    existing=pd.read_csv(path)
    if existing.experiment_id.isin([r['experiment_id'] for r in rows]).any():raise FileExistsError('Experiment IDs already in registry')
    with path.open('a',encoding='utf-8',newline='') as f:
        if before and not before.endswith(b'\n'):f.write('\n')
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n')
        writer.writerows(rows)
    if path.read_bytes()[:len(before)]!=before:raise ValueError('Registry historic bytes changed')
    save(run/'registry_append_receipt.json',dict(original_bytes=len(before),original_sha256=hashlib.sha256(before).hexdigest(),rows_appended=len(rows),historic_prefix_unchanged=True))


def research(run,config,info):
    begin_stage(run,'research')
    features=config['features'];params=config['params']
    data=joblib.load(run/'canonical_top50_through2025.joblib');data=data[data.date<'2025-01-01'].copy()
    ranks=joblib.load(run/'top50_past_only_ranks.joblib')
    ranked=data.merge(ranks,on=['date','stock_id'],how='left',validate='one_to_one')
    eligible=ranked[np.isfinite(ranked[features+RANK_FEATURES]).all(axis=1)][['date','stock_id']]
    control=read_control().merge(eligible,on=['date','stock_id'],validate='one_to_one').sort_values(['date','stock_id']).reset_index(drop=True)
    if control.empty:raise ValueError('No common validation rows')
    control.to_csv(run/'common_control_predictions.csv',index=False,float_format='%.17g')
    save(run/'common_coverage.json',dict(original_rows=24049,common_rows=len(control),newly_excluded=24049-len(control),
        rank_universe='Source-available Top50 before labels; same fixed common validation mask for all3 experiments'))
    top100_error=None;top100=None
    try:
        raw,world,reports=read_training(config['data_zip'],info['Top100@Y'],'2024-12-31')
        top100,dq=canonical(raw,world,features)
        save(run/'top100_quality.json',dict(stock_reports=reports,canonical_quality=dq,training_stock_ids=info['Top100@Y'],unit_definition='Institution net flows and trading volume both shares; ownership percentage points'))
        joblib.dump(top100,run/'canonical_top100_through2024.joblib')
        target=top100[top100.stock_id.isin(info['Top50@Y'])].sort_values(['date','stock_id']).reset_index(drop=True)
        pd.testing.assert_frame_equal(target[['date','stock_id']],data[['date','stock_id']].reset_index(drop=True))
        np.testing.assert_array_equal(target[features].to_numpy(),data[features].to_numpy())
    except (ValueError,KeyError,AssertionError) as exc:
        top100_error=str(exc);save(run/'top100_failure.json',dict(status='failed_data_validation',error=top100_error,stock_pool_reduced=False))
    base=pd.read_csv(BASE/'baseline_predictions.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip')
    metrics=[];predictions={};failed=[]
    for family in FAMILIES:
        name=family+'__'+config['run_id']
        if family==FAMILIES[1] and top100_error:
            failed.append(dict(experiment_id=name,parent_experiment='C10-MINCHILD200',status='failed_data_validation',error=top100_error));continue
        output=ROOT/'results'/name;models=ROOT/'models'/name
        output.mkdir(exist_ok=False);models.mkdir(exist_ok=False)
        names=features+RANK_FEATURES if family==FAMILIES[2] else features
        conf=dict(experiment_id=name,family=family,parent_experiment='C10-MINCHILD200',features=names,params=params,seed=42,
                  hypothesis=hypotheses()[family],versions=config['versions'],git_commit=config['git_commit'],
                  data_sha256=config['data_sha256'],code_sha256=config['code_sha256'],registration=str(run/'preregistration.json'),
                  confirmation_used=False,holdout_used=False,validation_years=[2023,2024])
        save(output/'config.json',conf);save(ROOT/'configs'/f'{name}.json',conf)
        outputs=[];rows=[]
        for fold,year in enumerate([2023,2024],1):
            if family==FAMILIES[0]:train=rolling_train(data,year)
            elif family==FAMILIES[1]:train=top100[top100.date<pd.Timestamp(f'{year}-01-01')].copy()
            else:train=ranked[(ranked.date<pd.Timestamp(f'{year}-01-01'))&np.isfinite(ranked[names]).all(axis=1)].copy()
            source=ranked if family==FAMILIES[2] else data
            keys=control.loc[control.fold==fold,['date','stock_id']]
            val=source.merge(keys,on=['date','stock_id'],validate='one_to_one').sort_values(['date','stock_id']).reset_index(drop=True)
            train=train.sort_values(['date','stock_id']).reset_index(drop=True)
            assert train.date.max()<val.date.min() and set(val.stock_id)<=set(info['Top50@Y'])
            if family==FAMILIES[1] and set(train.stock_id)!=set(info['Top100@Y']):raise ValueError('Top100 pool changed')
            print(f'{family} fold={fold} train={len(train)} val={len(val)} features={len(names)}',flush=True)
            model,check=fit_saved(train,names,train.actual_return,params,models/f'fold_{fold}.txt',val,repeat=fold==1)
            save(output/f'fold_{fold}_metadata.json',{**check,'training_start':str(train.date.min().date()),'training_end':str(train.date.max().date()),'train_stocks':int(train.stock_id.nunique()),'val_stocks':int(val.stock_id.nunique())})
            train[['date','stock_id']].to_csv(output/f'training_universe_fold_{fold}.csv',index=False)
            pd.DataFrame({'feature':names,'gain':model.feature_importance('gain')}).to_csv(output/f'gain_fold_{fold}.csv',index=False)
            forecast=model.predict(val[names]);label=(forecast>0).astype(int)
            p=val[['date','stock_id','actual_return']].copy();p['fold']=fold;p['experiment_id']=name;p['actual_y']=val.y
            p['pred_label']=label;p['pred_probability']=np.nan;p['pred_return']=forecast;p['signed_return']=(2*label-1)*val.actual_return
            p['stock_source_date']=val.stock_source_date;p['vix_source_date']=val.vix_source_date
            if family==FAMILIES[2]:p['rank_source_date']=val.rank_source_date
            outputs.append(p)
            ref=control[control.fold==fold];historical=base.merge(keys,on=['date','stock_id'],validate='one_to_one')
            ar=average_return(label,val.actual_return)
            rows.append(dict(experiment_id=name,parent_experiment='C10-MINCHILD200',feature_set=family,objective='regression',params=json.dumps(params),seed=42,fold=fold,
                train_period=f'{train.date.min().date()}:{train.date.max().date()}',val_period=f'{val.date.min().date()}:{val.date.max().date()}',
                n_train=len(train),n_val=len(val),ar=ar,delta_ar=ar-historical.signed_return.mean(),direct_delta_ar=ar-ref.signed_return.mean(),
                accuracy=float(accuracy_score(val.y,label)),log_loss=None,mse=float(mean_squared_error(val.actual_return,forecast)),
                mae=float(mean_absolute_error(val.actual_return,forecast)),n_stocks=int(val.stock_id.nunique()),n_dates=int(val.date.nunique()),
                pred_up_ratio=float(label.mean()),actual_up_ratio=float(val.y.mean()),status='completed'))
        p=pd.concat(outputs,ignore_index=True).sort_values(['date','stock_id']).reset_index(drop=True)
        assert len(p)==len(control) and not p.duplicated(['date','stock_id']).any()
        p.to_csv(output/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
        reread=pd.read_csv(output/'predictions.csv',dtype={'stock_id':str},parse_dates=['date'],float_precision='round_trip')
        np.testing.assert_array_equal(reread.pred_return,p.pred_return)
        for fold,year in enumerate([2023,2024],1):
            q=reread[reread.fold==fold]
            source=ranked if family==FAMILIES[2] else data
            v=source.merge(q[['date','stock_id']],on=['date','stock_id'],validate='one_to_one').sort_values(['date','stock_id']).reset_index(drop=True)
            booster=lgb.Booster(model_file=str(models/f'fold_{fold}.txt'))
            np.testing.assert_array_equal(booster.predict(v[names]),q.pred_return)
            np.testing.assert_array_equal(v.actual_return,q.actual_return)
        predictions[name]=p;metrics.extend(rows)
    family=family_intervals(predictions,control)
    save(run/'research_intervals.json',family)
    make_report(run,config,predictions,metrics,family,failed)
    assert_protected(json.loads((run/'preregistration.json').read_text(encoding='utf-8'))['protected_before'])
    append_registry(metrics+failed,run)
    save(run/'research_complete.json',dict(completed_candidates=len(predictions),failed=failed,fit_count=len(predictions)*3,common_validation_rows=len(control)))


def make_report(run,config,predictions,metrics,family,failed):
    rows=[];blocks=[dict(style='HEADING_1',text='Static交付與C24–C26研究：'+config['run_id']),
        dict(style='NORMAL_TEXT',text='固定交付使用C10＋C21、只在2020–2025最終訓練；2026只讀歷史特徵，不讀y或計分。研究仍只評估2023/2024。三項改動獨立測試，不自動組合或替換交付包。')]
    for name,p in predictions.items():
        family_id=name.split('__')[0];h=hypotheses()[family_id]
        r=next(x for x in family['comparisons'] if x['experiment_id']==name)
        ms=[m for m in metrics if m['experiment_id']==name]
        pooled=float(p.signed_return.mean())
        conclusion='保留為待獨立驗證候選' if r['nominate'] else '未通過兩年一致改善，暫不保留'
        row=dict(experiment_id=name,family=family_id,ar_2023_bp=ms[0]['ar']*10000,ar_2024_bp=ms[1]['ar']*10000,
                 pooled_ar=pooled,pooled_percent=pooled*100,pooled_bp=pooled*10000,
                 fold_mean_bp=float(p.groupby('fold').signed_return.mean().mean()*10000),daily_equal_bp=float(p.groupby('date').signed_return.mean().mean()*10000),
                 delta_2023_bp=r['annual_delta']['1']*10000,delta_2024_bp=r['annual_delta']['2']*10000,delta_pooled_bp=r['delta_ar']*10000,
                 accuracy=float((p.pred_label==p.actual_y).mean()),pred_up_ratio=float(p.pred_label.mean()),conclusion=conclusion)
        rows.append(row)
        record=dict(**h,Parameters=config['params'],Validation='2023/2024; common source-eligible Top50 keys',AR=row,DeltaAR=r,Conclusion=conclusion,
                    Observation='Report both annual deltas,month counts,stock concentration and uncertainty. These are adaptive development results.',Next='No automatic combination or additional search;review stability and fixed deployment separately.')
        save(ROOT/'results'/name/'research_record.json',record)
        blocks.extend([
            dict(style='HEADING_2',text=family_id+'：'+conclusion),
            dict(style='NORMAL_TEXT',text='Hypothesis：'+h['Hypothesis']+' 提出理由：'+h['Reason']),
            dict(style='NORMAL_TEXT',text='Baseline：C10-MINCHILD200。Change：'+h['Change']),
            dict(style='NORMAL_TEXT',text='Parameters：LR=.05、trees200、leaves15、min_child_samples200、L2=1、seed42；同原C10。Validation：2023/2024，固定共同股票日期，不使用2025作候選驗證。'),
            dict(style='NORMAL_TEXT',text=f"AR：2023={row['ar_2023_bp']:.6f}bp；2024={row['ar_2024_bp']:.6f}bp；Pooled={pooled:.12f}（{pooled*100:.9f}%，{pooled*10000:.6f}bp）；Fold-mean={row['fold_mean_bp']:.6f}bp；Daily-equal={row['daily_equal_bp']:.6f}bp。"),
            dict(style='NORMAL_TEXT',text=f"ΔAR vs C10共同樣本：2023={row['delta_2023_bp']:+.6f}bp；2024={row['delta_2024_bp']:+.6f}bp；Pooled={row['delta_pooled_bp']:+.6f}bp；改善月份={r['months_improved']}/24。三對比同步95%區間=[{r['simultaneous95'][0]*10000:+.6f},{r['simultaneous95'][1]*10000:+.6f}]bp。"),
            dict(style='NORMAL_TEXT',text=f"Conclusion：{conclusion}。Accuracy={row['accuracy']:.4%}；預測1比例={row['pred_up_ratio']:.4%}。區間跨0時不能宣稱穩定泛化；Gain只作輔助解釋。")])
    pd.DataFrame(rows).to_csv(run/'research_comparison.csv',index=False)
    for item in failed:blocks.append(dict(style='NORMAL_TEXT',text='資料稽核失敗實驗：'+json.dumps(item,ensure_ascii=False)))
    blocks.extend([
        dict(style='HEADING_2',text='部署邊界與後續'),
        dict(style='NORMAL_TEXT',text='Static草稿包含3200筆每日C10、1000筆固定起點C21，共4200筆；9月只使用8/31前已知快照，h1重用C10。共20個最終模型與2次重跑。所有官方分數為null，尚未上傳。研究成績不繼承給最終重訓模型。'),
        dict(style='NORMAL_TEXT',text='仍有法人發布時間以前一股票日近似、復權vintage未逐筆確認及開發集多次選模等限制。Top3股票排除只作事後診斷，不作部署篩選。後續依各實驗年度穩定性與來源風險另立假說，不自動增加網格或組合。')])
    save(run/'google_doc_payload.json',dict(blocks=blocks))
    text='\n\n'.join(('# '+b['text']) if b['style']=='HEADING_1' else ('## '+b['text']) if b['style']=='HEADING_2' else b['text'] for b in blocks)
    (run/'research_findings.md').write_text(text+'\n',encoding='utf-8')


def finish(run,config):
    registration=json.loads((run/'preregistration.json').read_text(encoding='utf-8'))
    assert_protected(registration['protected_before'])
    stages={name:(run/f'{name}_complete.json').exists() for name in ['audit','package','research']}
    if stages['package']:
        verification=verify_package(run,config['data_zip'],config['template'],config['features'])
        save(run/'static_verification.json',verification)
    if stages['research']:
        completed=json.loads((run/'research_complete.json').read_text(encoding='utf-8'))
        save(run/'research_verification.json',dict(status='passed',native_csv_predictions_equal=True,
            completed_candidates=completed['completed_candidates'],fit_count=completed['fit_count'],
            original_registry_bytes_preserved=True,validation_years=[2023,2024],holdout_2026_used=False))
    save(run/'run_status.json',dict(stages=stages,original_artifacts_unchanged=True,official_score=None,competition_submission=False))
    command=f"& '{ROOT.parent/'.venv/Scripts/python.exe'}' -B -m lightgbm_baseline.experiments.next_stage --stage all --data-zip '{config['data_zip']}' --template '{config['template']}' --run-id NEW_UNIQUE_ID"
    (run/'README.md').write_text('# Static候選與下一批實驗\n\n'
        '使用既有Paper專案與venv；從Paper工作目錄執行以下PowerShell命令。每次選新Run ID；既有all執行不覆寫。\n\n```powershell\n'+command+'\n```\n\n'
        '階段為audit→package→research。可分別指定stage延續同一個已稽核且設定相同的Run ID；已開始的stage不自動覆寫或重跑。\n\n'
        'delivery/static_submission.csv是待批准的官方格式草稿；delivery/predictions_long.csv保存完整路由與來源。'
        '研究模型與預測在experiments/models及results的C24/C25/C26__RUN_ID；本目錄保存比較、品質報告、程式快照與設定。'
        '模型、快取與交付檔均被既有或本目錄的.gitignore排除，沒有自動Git操作。\n\n'
        '2025僅供最終fit；2026 y未讀、沒有本地2026評分或競賽上傳。\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['audit','package','research','all'],required=True)
    parser.add_argument('--data-zip',required=True);parser.add_argument('--template',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    run,config,info,dates=prepare_run(args)
    if not (run/'audit_complete.json').exists():audit(run,config,info,dates)
    elif args.stage=='audit':raise FileExistsError('Audit already completed')
    if args.stage in ['package','all']:
        begin_stage(run,'package')
        data=joblib.load(run/'canonical_top50_through2025.joblib')
        manifest=package(run,config['data_zip'],config['template'],data,config)
        save(run/'package_complete.json',dict(status='passed',models=manifest['models'],fit_count=22,submission_sha256=manifest['submission_sha256']))
    if args.stage in ['research','all']:research(run,config,info)
    finish(run,config)
    print(str(run),flush=True)


if __name__=='__main__':main()
