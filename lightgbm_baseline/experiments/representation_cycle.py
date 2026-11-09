"""C6: one frozen encoder, comparing scalar output with stable hidden coordinates."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from lightgbm import Booster
from .research_cycles import ROOT,LAB,preregister,train_run,paired,save
from .compare_experiments import table


def extract(model,frame,lags):
    with threadpool_limits(limits=1):
        x=model[0].transform(frame[lags])
        hidden=np.tanh(x@model[-1].coefs_[0]+model[-1].intercepts_[0])
        scalar=model.predict(frame[lags])*.01
    return hidden,scalar


def run():
    out=preregister('C6',dict(
        hypothesis='A fixed 16-dimensional hidden representation carries useful patterns discarded by a scalar upstream forecast.',
        reason='C5 nonlinear scalar had positive increments in both years, but very small 2023 gain and uncertainty across zero. Test representation without another architecture search.',
        encoder='Reuse C5 upstream_inner_2021-01-01.joblib; fitted only through 2020-12-31. Freeze encoder and scaler for all 2021-2024 samples.',
        alignment_reason='Separately fine-tuned encoders can permute hidden coordinates across OOF blocks. A single frozen encoder avoids meaningless concatenation of changing coordinate systems.',
        tests=['C6-FROZEN-SCALAR versus C5-RAW20','C6-FROZEN-HIDDEN versus C6-FROZEN-SCALAR and C5-RAW20'],
        controls='Same 2021+ training rows, 2023/2024 validation rows, canonical input and LightGBM parameters as C5.',
        keep_rule='Retain hidden representation only if its annual increments over both scalar and raw-history control are positive. Otherwise do not expand embedding dimension.',
        limitations='The reused MLP reached its fixed 40-iteration budget with convergence warnings. This tests its current features, not an optimally trained neural representation. No 2025/2026.'))
    data=joblib.load(LAB/'C5/upstream_features_exact.joblib')
    cfg=json.loads((ROOT/'results/C5-RAW20/config.json').read_text())
    features=cfg['features'];params={k:v for k,v in cfg['params'].items() if k!='objective'}
    encoder_path=LAB/'C5/upstream_inner_2021-01-01.joblib'
    encoder=joblib.load(encoder_path)
    assert data.date.min()>pd.Timestamp('2020-12-31')
    lags=[f'raw_intraday_lag_{i}' for i in range(1,21)]
    hidden,scalar=extract(encoder,data,lags)
    names=[f'frozen_hidden_{i:02d}' for i in range(hidden.shape[1])]
    data[names]=hidden
    data['frozen_scalar']=scalar
    joblib.dump(data,out/'upstream_features_exact.joblib')
    restored=joblib.load(out/'upstream_features_exact.joblib')
    pd.testing.assert_frame_equal(data,restored,check_exact=True)
    save(out/'encoder_manifest.json',dict(source=str(encoder_path),sha256=hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        fit_cutoff='2020-12-31',n_hidden=len(names),train_min=str(data.date.min().date()),
        feature_formulas='tanh(StandardScaler_train2020(X) @ W_hidden + b_hidden); scalar uses the same fixed output head * .01'))
    train_run('C6-FROZEN-SCALAR',data,features+['frozen_scalar'],params,parent='C5-RAW20')
    train_run('C6-FROZEN-HIDDEN',data,features+names,params,parent='C6-FROZEN-SCALAR')
    results=[paired(a,b) for a,b in [('C6-FROZEN-SCALAR','C5-RAW20'),('C6-FROZEN-HIDDEN','C5-RAW20'),('C6-FROZEN-HIDDEN','C6-FROZEN-SCALAR')]]
    save(out/'results.json',results)
    # Check actual delivered model artifacts, not only construction-time shapes.
    summary=[]
    for identifier,cols in [('C6-FROZEN-SCALAR',features+['frozen_scalar']),('C6-FROZEN-HIDDEN',features+names)]:
        p=pd.read_csv(ROOT/'results'/identifier/'predictions.csv',parse_dates=['date'],dtype={'stock_id':str},float_precision='round_trip')
        for fold,year in enumerate([2023,2024],1):
            v=restored[restored.date.dt.year==year]
            q=p[p.fold==fold]
            assert list(zip(v.date,v.stock_id))==list(zip(q.date,q.stock_id))
            model=Booster(model_file=str(ROOT/'models'/identifier/f'fold_{fold}.txt'))
            np.testing.assert_allclose(model.predict(v[cols]),q.pred_return,atol=1e-12,rtol=0)
        summary.append(dict(experiment_id=identifier,ar_2023_bp=p[p.fold==1].signed_return.mean()*10000,
                            ar_2024_bp=p[p.fold==2].signed_return.mean()*10000,pooled_ar_bp=p.signed_return.mean()*10000))
    scores=pd.DataFrame(summary)
    scores.to_csv(out/'comparison.csv',index=False)
    text='# C6 固定上游的表示實驗\n\n假說：MLP中間表示比其單一報酬預測含有更多可用資訊。理由：C5 scalar有弱的兩年增益，先測輸出壓縮瓶頸，不增加新模型家族。\n\n'
    text+='上游只用2020資料訓練，對2021以後全部樣本凍結；所有對照沿用C5相同訓練列與原始歷史特徵。這避免不同OOF模型的hidden神經元座標不一致。原MLP訓練達40次上限，有收斂警告，因此不是對最適MLP的結論。\n\n'+table(scores.round(6))+'\n'
    for r in results:
        text+=f"- {r['candidate']} 對 {r['control']}：2023 Δ={r['annual_delta']['1']*10000:.6f} bp，2024 Δ={r['annual_delta']['2']*10000:.6f} bp，Pooled Δ={r['delta_ar']*10000:.6f} bp；探索區間 {np.array(r['bootstrap_95_percentile'])*10000} bp。\n"
    keep=all(v>0 for r in results[1:] for v in r['annual_delta'].values())
    text+='\n決策：'+('隱表示通過兩年點估計規則，但需保留不確定性限制。' if keep else '隱表示未通過對兩個控制組的兩年增量規則；停止擴大此embedding分支，不以Pooled最高值選維度。')+'\n\n'
    text+='模型與快取重載4個Fold通過；2025/2026未使用。完整假說與設定見 cycles/C6/preregistration.json，結果見 results.json。後續應先完成少數候選凍結與數值／時間稽核；不能靠反覆看開發集提高分數就宣稱有官方榜單競爭力。\n'
    (ROOT/'reports/C6_representation_findings.md').write_text(text,encoding='utf-8')
    print(scores.to_string(index=False));print(json.dumps(results,indent=2))


if __name__=='__main__':run()
