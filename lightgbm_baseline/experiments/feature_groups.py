from zipfile import ZipFile
import json
import numpy as np
import pandas as pd


def augment(data, zip_path, group):
    if group == 'FEAT-00':
        return data, [], {}
    frames=[]
    definitions={}
    with ZipFile(zip_path) as z:
        stocks=json.loads(z.read('data/info.json'))['Top50@Y']
        for stock in stocks:
            r=pd.read_csv(z.open(f'data/market/{stock}.csv'))
            r['date']=pd.to_datetime(r.date)
            r=r[r.date<'2026-01-01'].sort_values('date').reset_index(drop=True)
            o,h,l,c=[r[f'etl:adj_{x}'] for x in ['open','high','low','close']]
            v,a,n=[r[x] for x in ['price:成交股數','price:成交金額','price:成交筆數']]
            def ratio(x,y): return x/y.replace(0,np.nan)
            x=pd.DataFrame(index=r.index)
            groups=group.split('+')
            if 'FEAT-01' in groups:
                x['intraday']=ratio(c-o,o)
                x['close_return']=c.pct_change(fill_method=None)
                x['overnight']=ratio(o-c.shift(),c.shift())
                x['range']=ratio(h-l,o)
                for w in [5,20]:
                    x[f'momentum_{w}']=ratio(c,c.shift(w))-1
                    x[f'volatility_{w}']=c.pct_change(fill_method=None).rolling(w,min_periods=w).std(ddof=1)
                definitions['FEAT-01']='At source day s: (C-O)/O, C/C[-1]-1, O/C[-1]-1, (H-L)/O, C/C[-5,20]-1, std(ddof=1) of 5/20 close returns; all shifted one observed row.'
            if 'FEAT-02' in groups:
                for w in [5,20]: x[f'volume_ratio_{w}']=ratio(v,v.rolling(w,min_periods=w).mean())
                x['amount_ratio_20']=ratio(a,a.rolling(20,min_periods=20).mean())
                x['average_trade_size']=ratio(v,n)
                x['count_ratio_20']=ratio(n,n.rolling(20,min_periods=20).mean())
                definitions['FEAT-02']='At source day s: volume/mean(volume,5,20), amount/mean(amount,20), volume/count (shares per trade), count/mean(count,20); all shifted one row; rolling includes source day; zero denominator missing.'
            if 'FEAT-03' in groups:
                columns=['外陸資買賣超股數(不含外資自營商)','投信買賣超股數','自營商買賣超股數(自行買賣)','自營商買賣超股數(避險)']
                for name,col in zip(['foreign','trust','dealer','hedging'],columns):
                    flow=r['institutional_investors_trading_summary:'+col]
                    x[f'{name}_flow_ratio']=ratio(flow,v)
                    x[f'{name}_flow_ratio_5']=ratio(flow.rolling(5,min_periods=5).sum(),v.rolling(5,min_periods=5).sum())
                x['ownership_change_5']=r['foreign_investors_shareholding:全體外資及陸資持股比率'].diff(5)
                definitions['FEAT-03']='Official net-flow columns in shares / official volume in shares; rolling sum(flow,5)/sum(volume,5), ownership ratio diff(5) in percentage points; all shifted one row. Availability is shift(1) approximation, not timestamp verification.'
            x=x.shift(1).replace([np.inf,-np.inf],np.nan)
            x['date'],x['stock_id']=r.date,stock
            frames.append(x)
    extra=pd.concat(frames,ignore_index=True)
    names=[c for c in extra if c not in ['date','stock_id']]
    if not names: raise ValueError('No supported feature group')
    return data.merge(extra,on=['date','stock_id'],validate='one_to_one'),names,definitions
