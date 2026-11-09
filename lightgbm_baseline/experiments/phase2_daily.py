"""Offline Phase2 daily audit/draft using frozen C26;never downloads, trains or uploads."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import importlib.metadata
from pathlib import Path
import re,shutil
from zipfile import ZipFile
import lightgbm as lgb
import numpy as np
import pandas as pd
from .next_stage import ROOT
from .static_candidate import sha,save,REQUIRED
from .rank_ablation import ranked_asof,FULL
from .submission_format import format_predictions

TAIPEI=timezone(timedelta(hours=8))
LOCKED=ROOT/'next_stage/RANK20261104_V1/candidate'
TARGET_FIELDS={'y','actual_y','actual_return','target_return'}


def daily_template(path,stocks):
    t=pd.read_csv(path,dtype=str)
    if t.shape!=(50,2) or t.columns[0]!='id' or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',t.columns[1]):raise ValueError('Expected official Phase2 id+one-date 50-row template')
    if t.id.isna().any() or t.id.duplicated().any() or t.id.tolist()!=stocks:raise ValueError('Official stock IDs/order mismatch')
    date=pd.Timestamp(t.columns[1])
    if date.strftime('%Y-%m-%d')!=t.columns[1]:raise ValueError('Invalid session date')
    return t,date


def deadline_check(date,now):
    if now.tzinfo is None:raise ValueError('Timezone-aware clock required')
    close=datetime.combine(pd.Timestamp(date).date(),datetime.min.time(),TAIPEI)+timedelta(hours=9)
    if now.astimezone(TAIPEI)>=close:raise ValueError('Session deadline passed;do not prepare a current submission')
    return close


def read_feature_zip(path,stocks,cutoff):
    """Parse only feature usecols;future price rows are removed before feature calculations."""
    frames=[];cutoff=pd.Timestamp(cutoff)
    with ZipFile(path) as z:
        if z.testzip():raise ValueError('ZIP CRC failure')
        for stock in stocks:
            p=pd.read_csv(z.open(f'data/market/{stock}.csv'),usecols=REQUIRED)
            p['date']=pd.to_datetime(p.date,errors='raise')
            if p.date.isna().any() or p.date.duplicated().any() or not p.date.is_monotonic_increasing:raise ValueError('Unsorted/duplicate stock dates:'+stock)
            p=p[p.date<cutoff].copy();p['stock_id']=str(stock)
            for col in REQUIRED:
                if col!='date':p[col]=pd.to_numeric(p[col],errors='raise')
            frames.append(p)
        world=pd.read_csv(z.open('data/global/global_data.csv'),usecols=['date','^VIX'])
    world['date']=pd.to_datetime(world.date,errors='raise');world=world[world.date<cutoff].copy()
    if world.date.duplicated().any():raise ValueError('Duplicate global dates')
    world['^VIX']=pd.to_numeric(world['^VIX'],errors='raise')
    return pd.concat(frames,ignore_index=True),world


def combine_history(a,b,keys):
    p=pd.concat([a,b],ignore_index=True)
    if p.duplicated(keys).any():
        for _,g in p[p.duplicated(keys,keep=False)].groupby(keys):
            first=g.iloc[[0]].reset_index(drop=True)
            for i in range(1,len(g)):pd.testing.assert_frame_equal(first,g.iloc[[i]].reset_index(drop=True),check_dtype=False)
        p=p.drop_duplicates(keys)
    return p.sort_values(keys).reset_index(drop=True)


def freshness(history,world,date,expected,status=None,live_start=None):
    """Never infer that -1 signed flow is missing;require declared live completeness."""
    if TARGET_FIELDS&set(history):raise ValueError('Inference history must be label-free')
    target=pd.Timestamp(date);expected=pd.Timestamp(expected)
    if expected>=target:raise ValueError('Expected source date must precede session')
    if (target-expected).days>7:raise ValueError('Expected stock source older than7 days;manual review required')
    history=history[history.date<target].copy();world=world[world.date<target].copy();excluded=[]
    states=(status or {}).get('dates',{})
    if live_start is not None:
        for d,state in states.items():
            if state.get('other_features')=='closed_incomplete':
                day=pd.Timestamp(d);excluded.append(d);history=history[history.date!=day];world=world[world.date!=day]
    quality=[]
    for stock,g in history.groupby('stock_id',sort=False):
        g=g.sort_values('date');tail=g.tail(20)
        if len(tail)!=20:raise ValueError('Insufficient20 valid prior rows:'+stock)
        if tail.date.max()!=expected:raise ValueError(f'Stale stock source:{stock},latest={tail.date.max().date()},expected={expected.date()}')
        if live_start is not None:
            for d in tail.loc[tail.date>=pd.Timestamp(live_start),'date']:
                if states.get(d.strftime('%Y-%m-%d'),{}).get('other_features')!='complete':raise ValueError('Missing official completeness evidence:'+str(d.date()))
        if not np.isfinite(tail[REQUIRED[1:]]).all().all():raise ValueError('Nonfinite live inputs:'+stock)
        for c in ['etl:adj_open','etl:adj_high','etl:adj_low','etl:adj_close','SMA']:
            if tail[c].le(0).any():raise ValueError('Invalid/placeholder positive feature:'+c)
        for c in ['price:成交股數','price:成交金額','price:成交筆數']:
            if tail[c].lt(0).any():raise ValueError('Invalid/placeholder activity:'+c)
        if not tail.RSI.between(0,100).all() or not tail['foreign_investors_shareholding:全體外資及陸資持股比率'].between(0,100).all():raise ValueError('Invalid/placeholder bounded feature')
        quality.append(dict(stock_id=str(stock),latest_source=str(tail.date.max().date()),source_age_days=int((target-tail.date.max()).days)))
    world=world.sort_values('date')
    if world.empty or not np.isfinite(world.iloc[-1]['^VIX']) or world.iloc[-1]['^VIX']<=0:raise ValueError('Invalid/placeholder prior VIX')
    age=int((target-world.iloc[-1].date).days)
    if age>7:raise ValueError('Prior VIX older than7 calendar days;manual review required')
    if live_start is not None and world.iloc[-1].date>=pd.Timestamp(live_start):
        if states.get(world.iloc[-1].date.strftime('%Y-%m-%d'),{}).get('other_features')!='complete':raise ValueError('Missing VIX completeness evidence')
    return history,world,dict(stock_sources=quality,vix_source_date=str(world.iloc[-1].date.date()),vix_age_days=age,
        explicitly_excluded_closed_dates=excluded,signed_minus_one_flows_retained=True,
        official_group_completion_is_not_field_publication_timestamp=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-zip',required=True);ap.add_argument('--live-zip');ap.add_argument('--availability');ap.add_argument('--template',required=True)
    ap.add_argument('--expected-source-date',required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--stage',choices=['audit','draft'],default='audit');args=ap.parse_args()
    if not re.fullmatch('[A-Za-z0-9_-]+',args.run_id):raise ValueError('Invalid run ID')
    run=ROOT/'next_stage'/args.run_id
    if run.exists():raise FileExistsError('Existing run ID;never overwrite')
    run.mkdir();shutil.copyfile(__file__,run/'driver_source.py')
    conf=dict(run_id=args.run_id,mode=args.stage,created_utc=datetime.now(timezone.utc).isoformat(),data_zip=str(Path(args.data_zip).resolve()),data_sha256=sha(args.data_zip),
        live_zip=args.live_zip,live_sha256=sha(args.live_zip) if args.live_zip else None,template=str(Path(args.template).resolve()),template_sha256=sha(args.template),
        expected_source_date=args.expected_source_date,model=str(LOCKED/'models/h01.txt'),model_sha256=sha(LOCKED/'models/h01.txt'),
        training_cutoff='2025-12-31',new_training_fits=0,labels_2026_read=False,local_2026_score_computed=False,upload_allowed=False,official_score=None)
    conf['code_sha256']={str(p):sha(p) for p in [Path(__file__),ROOT/'rank_ablation.py',ROOT/'asof_predict.py',ROOT/'submission_format.py']}
    conf['versions']={p:importlib.metadata.version(p) for p in ['lightgbm','numpy','pandas','scikit-learn']}
    save(run/'config.json',conf)
    try:
        original=json.loads((ROOT/'results'/FULL/'config.json').read_text(encoding='utf-8'))
        if conf['data_sha256']!=original['data_sha256']:raise ValueError('Original official ZIP version mismatch')
        with ZipFile(args.data_zip) as z:stocks=json.loads(z.read('data/info.json'))['Top50@Y']
        template,date=daily_template(args.template,stocks);close=deadline_check(date,datetime.now(timezone.utc));conf['deadline']=close.isoformat()
        if args.live_zip and not args.availability:raise ValueError('Live ZIP requires official completeness evidence JSON')
        history,world=read_feature_zip(args.data_zip,stocks,date);live_start=None;status=None
        if args.live_zip:
            extra,extra_world=read_feature_zip(args.live_zip,stocks,date);live_start=extra.date.min();status=json.loads(Path(args.availability).read_text(encoding='utf-8'))
            observed=datetime.fromisoformat(status['observed_at'])
            if observed.tzinfo is None or observed>datetime.now(timezone.utc)+timedelta(minutes=1):raise ValueError('Invalid evidence observation timestamp')
            conf['availability_sha256']=sha(args.availability);history=combine_history(history,extra,['stock_id','date']);world=combine_history(world,extra_world,['date'])
        history,world,quality=freshness(history,world,date,args.expected_source_date,status,live_start)
        manifest=json.loads((LOCKED/'manifest.json').read_text(encoding='utf-8'))
        if conf['model_sha256']!=manifest['files']['models/h01.txt']:raise ValueError('Frozen C26 model SHA mismatch')
        model=lgb.Booster(model_file=str(LOCKED/'models/h01.txt'));features=original['features']
        if model.feature_name()!=features or len(features)!=43:raise ValueError('Frozen model feature mismatch')
        frame=ranked_asof(history,world,date,stocks)
        if not frame.stock_source_date.eq(pd.Timestamp(args.expected_source_date)).all():raise ValueError('Snapshot freshness mismatch')
        if not all(frame[c].lt(date).all() for c in ['stock_source_date','vix_source_date','rank_source_date']):raise ValueError('Future feature source')
        if not np.isfinite(frame[features]).all().all():raise ValueError('Nonfinite43 inputs')
        joblib_path=run/'features_exact.csv';frame.to_csv(joblib_path,index=False,float_format='%.17g')
        save(run/'data_quality_report.json',dict(status='passed',**quality,session=str(date.date()),n_stocks=50,n_features=43,labels_2026_read=False,local_score=None,
            limitations='Institution timing and adjusted-history vintage remain inherited assumptions;group update log is not per-field historical publication proof.'))
        if args.stage=='draft':
            pred=model.predict(frame[features],num_threads=4)
            if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
            p=frame[['date','stock_id','stock_source_date','vix_source_date','rank_source_date']].copy();p['experiment_id']='C26-FROZEN-2025-DAILY';p['actual_y']=np.nan;p['actual_return']=np.nan;p['pred_probability']=np.nan;p['pred_return']=pred;p['pred_label']=(pred>0).astype(int)
            p.to_csv(run/'predictions_long.csv',index=False,float_format='%.17g');format_predictions(args.template,run/'predictions_long.csv',run/'phase2_draft.csv')
            q=pd.read_csv(run/'phase2_draft.csv',dtype=str)
            if q.shape!=(50,2) or q.columns.tolist()!=template.columns.tolist() or q.id.tolist()!=template.id.tolist() or not q.iloc[:,1].isin(['0','1']).all():raise ValueError('Daily CSV readback mismatch')
            conf['draft_sha256']=sha(run/'phase2_draft.csv')
        save(run/'config_complete.json',conf);save(run/'status.json',dict(status='audited' if args.stage=='audit' else 'draft_ready_for_review',new_submissions=0,approval_required=True))
        print(json.dumps(dict(status='passed',run=str(run),stage=args.stage,submission_performed=False),ensure_ascii=False))
    except (ValueError,KeyError,AssertionError,FileNotFoundError) as exc:
        save(run/'status.json',dict(status='blocked_data_quality',reason=str(exc),new_submissions=0,submission_allowed=False))
        print(json.dumps(dict(status='blocked_data_quality',reason=str(exc),run=str(run)),ensure_ascii=False));raise SystemExit(2)


if __name__=='__main__':main()
