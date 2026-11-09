"""Label-free, one-date inference from strictly earlier official history."""
import json
from pathlib import Path
from zipfile import ZipFile
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm_baseline.config import FEATURE_MAPPING,STOCK_COLUMNS
from .representation_cycle import extract

FLOW_COLUMNS=dict(foreign='外陸資買賣超股數(不含外資自營商)',trust='投信買賣超股數',
                  dealer='自營商買賣超股數(自行買賣)',hedging='自營商買賣超股數(避險)')


def load_history(zip_path):
    with ZipFile(zip_path) as archive:
        damaged=archive.testzip()
        if damaged:raise ValueError(f'ZIP CRC failure: {damaged}')
        stocks=json.loads(archive.read('data/info.json'))['Top50@Y']
        if len(stocks)!=50 or not all(isinstance(x,str) for x in stocks) or len(set(stocks))!=50:
            raise ValueError('Expected 50 unique string stock IDs')
        frames=[]
        required=['date',*STOCK_COLUMNS,*['institutional_investors_trading_summary:'+v for v in FLOW_COLUMNS.values()]]
        for stock in stocks:
            # Deliberately never read y, even if it is present in the official file.
            r=pd.read_csv(archive.open(f'data/market/{stock}.csv'),usecols=required)
            r['stock_id']=stock;frames.append(r)
        global_data=pd.read_csv(archive.open('data/global/global_data.csv'),usecols=['date','^VIX'])
    history=pd.concat(frames,ignore_index=True)
    history['date']=pd.to_datetime(history.date);global_data['date']=pd.to_datetime(global_data.date)
    if history.duplicated(['stock_id','date']).any() or global_data.date.duplicated().any():raise ValueError('Duplicate history dates')
    return history,global_data,stocks


def build_asof(history,global_data,date,stocks):
    target=pd.Timestamp(date)
    if target!=target.normalize():raise ValueError('Target must be a calendar date')
    # The cutoff precedes every feature calculation, including every rolling operation.
    past=history[history.date<target].copy()
    world=global_data[global_data.date<target].sort_values('date')
    if world.empty:raise ValueError('No strictly earlier VIX')
    rows=[]
    def divide(a,b):return float(a/b) if np.isfinite(b) and b!=0 else np.nan
    for stock in stocks:
        s=past[past.stock_id==stock].sort_values('date')
        if len(s)<20:raise ValueError(f'Not enough history for {stock}')
        last=s.iloc[-1]
        row={FEATURE_MAPPING[c]:last[c] for c in STOCK_COLUMNS}
        row['vix']=world.iloc[-1]['^VIX']
        volume=s['price:成交股數']
        for name,column in FLOW_COLUMNS.items():
            flow=s['institutional_investors_trading_summary:'+column]
            row[name+'_flow_ratio']=divide(flow.iloc[-1],volume.iloc[-1])
            row[name+'_flow_ratio_5']=divide(flow.rolling(5,min_periods=5).sum().iloc[-1],volume.rolling(5,min_periods=5).sum().iloc[-1])
        row['ownership_change_5']=s['foreign_investors_shareholding:全體外資及陸資持股比率'].diff(5).iloc[-1]
        returns=(s['etl:adj_close']-s['etl:adj_open'])/s['etl:adj_open']
        for lag in range(1,21):row[f'raw_intraday_lag_{lag}']=returns.iloc[-lag]
        keys=list(row)
        values=np.asarray([row[x] for x in keys],dtype='float64').round(10)
        if not np.isfinite(values).all():raise ValueError(f'Nonfinite as-of features for {stock}')
        row=dict(zip(keys,values))
        row.update(stock_id=stock,date=target,stock_source_date=last.date,vix_source_date=world.iloc[-1].date)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_asof(history,global_data,date,stocks,model_path,encoder_path,features):
    frame=build_asof(history,global_data,date,stocks)
    if any(x.startswith('frozen_hidden_') for x in features):
        encoder=joblib.load(encoder_path)
        lags=[f'raw_intraday_lag_{i}' for i in range(1,21)]
        hidden,_=extract(encoder,frame,lags)
        for i in range(hidden.shape[1]):frame[f'frozen_hidden_{i:02d}']=hidden[:,i]
    model=lgb.Booster(model_file=str(model_path))
    if model.feature_name()!=features:raise ValueError('Model feature order mismatch')
    pred=model.predict(frame[features])
    result=frame[['date','stock_id','stock_source_date','vix_source_date']].copy()
    result['pred_return']=pred;result['pred_label']=(pred>0).astype(int)
    return result
