import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).parent


def table(frame):
    columns=list(frame.columns)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def compare(ids=None):
    rows=[]
    for directory in sorted((ROOT/'results').iterdir()):
        if ids and directory.name not in ids: continue
        m=pd.read_csv(directory/'metrics.csv')
        p=pd.read_csv(directory/'predictions.csv',dtype={'stock_id':str},parse_dates=['date'])
        assert not p.duplicated(['stock_id','date']).any()
        assert set(p.fold)=={1,2}
        config=json.loads((directory/'config.json').read_text())
        row=dict(experiment_id=directory.name, feature_count=len(config.get('feature_names') or config['features']),
                 ar_2023=m.iloc[0].ar, ar_2024=m.iloc[1].ar,
                 pooled_ar=p.signed_return.mean(), fold_mean_ar=m.ar.mean(),
                 daily_equal_ar=p.groupby('date').signed_return.mean().mean(),
                 pooled_accuracy=(p.actual_y==p.pred_label).mean(),
                 delta_ar=np.average(m.delta_ar,weights=m.n_val),
                 direct_delta_ar=np.average(m.get('direct_delta_ar',m.delta_ar).fillna(m.delta_ar),weights=m.n_val),
                 min_annual_delta=m.delta_ar.min(),n_val=len(p))
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'comparison.csv',index=False)
    return result


def reports():
    out=ROOT/'reports'
    out.mkdir(exist_ok=True)
    baseline=ROOT.parent/'results_official_v1_20261031_200052'
    bm=pd.read_csv(baseline/'baseline_metrics.csv')
    bp=pd.read_csv(baseline/'baseline_predictions.csv',dtype={'stock_id':str},parse_dates=['date'])
    summary=json.loads((baseline/'baseline_summary.json').read_text())
    aggregations=pd.DataFrame([dict(aggregation=k,decimal=v,percent=v*100,basis_points=v*10000) for k,v in summary.items() if k!='percent'])
    distribution=bp.groupby('fold').agg(predicted_1=('pred_label','mean'),actual_1=('actual_y','mean')).reset_index()
    controls=pd.read_csv(baseline/'rule_metrics.csv')
    result=compare()
    registry=pd.read_csv(ROOT/'experiment_registry.csv')
    for identifier in ['OBJ-00']:
        p=pd.read_csv(ROOT/'results'/identifier/'predictions.csv',dtype={'stock_id':str},parse_dates=['date'])
        check=bp[bp.fold.isin([1,2])].sort_values(['date','stock_id'])
        p=p.sort_values(['date','stock_id'])
        assert list(zip(p.date,p.stock_id))==list(zip(check.date,check.stock_id))
        np.testing.assert_allclose(p.pred_probability,check.pred_probability,atol=1e-12,rtol=0)
    (out/'E0_baseline_review.md').write_text('# E0 Baseline Audit\n\n稽核對象：`../../results_official_v1_20261031_200052/`，不是其他實驗分數。\n\n'+table(bm)+'\n'+table(aggregations)+'\n'+table(distribution)+'\n'+table(controls)+'\n11 個官方特徵、group shift(1)、嚴格早於預測日的 VIX、固定年度 split 均直接重用既有程式。OBJ-00 重跑 2023/2024 的全部機率與原始 Baseline 在 1e-12 容差內一致。50 股清單由官方 info.json 固定；各年度日期自然不同，因此是同一個事先指定的股票集合與各年度實際樣本集合，不是三年相同日期。每個新實驗均按股票日期對齊原始結果；這批所有實驗的驗證列數保持 11,950 / 12,099。\n\n股價與 VIX 來源違規、重複鍵、標籤不一致均為 0。shift(1) 不等於發布時間驗證。原始 Baseline 已看過 2025，Phase 2 所有候選只跑 2023、2024。\n',encoding='utf-8')
    (out/'E1_objective_findings.md').write_text('# E1 Objective Findings\n\n比較範圍只含 2023、2024。AR 為 Decimal；乘 100 為百分比，乘 10000 為 basis points。\n\n'+table(result[~result.experiment_id.str.contains('__')])+'\n'+table(registry[~registry.experiment_id.str.contains('__')][['experiment_id','fold','ar','delta_ar','accuracy','log_loss','pred_positive_ratio','actual_positive_ratio','weight_cap']])+'\nOBJ-03 是唯一在兩個年度均提高 AR 的新 Objective，因此選它進 E2。OBJ-01/02 只在 2023 改善，2024 反而退步，不能憑 Pooled 高分宣稱最佳。權重與 99% cap 只由每 Fold Training 建立；權重均值正規化。Regression 以預測報酬 > 0 判 1，機率與 Log Loss 為空值，不造機率。\n\n來源：../experiment_registry.csv、../results/OBJ-*/metrics.csv 與 predictions.csv。\n',encoding='utf-8')
    (out/'E2_feature_findings.md').write_text('# E2 Feature Findings\n\n'+table(result[result.experiment_id.str.contains('__')])+'\n'+table(registry[registry.experiment_id.str.contains('__')][['experiment_id','fold','n_train','n_val','ar','delta_ar','direct_delta_ar','accuracy']])+'\n## 判讀\n\n法人群組 FEAT-03 在 OBJ-03 與 OBJ-00 的兩年均有增益，證據比只對單一 objective 有效更一致。價格 FEAT-01 也在兩年、兩個 objectives 改善；成交量群組在 OBJ-00 的 2023 退步，效果依賴 objective。FEAT-06 使用法人＋價格，選取依據是 OBJ-03 下兩年都正增益，再按最弱年度 direct delta 排序；不自動加入全部群組。實驗 ID 中 FEAT-03+FEAT-01 即 FEAT-06。\n\n公式與完整 Feature List 在各 results/<ID>/config.json；每個群組先在股票內建立完整歷史指標，再整體 shift(1)。rolling 包含來源日、完整窗口、波動率 ddof=1、零分母缺值；平均交易規模定義為股數/筆數。法人欄為官方買賣超股數，除以官方成交股數，持股變化為百分點。這是官方欄位名稱的單位假設，未另做供應商單位或發布時間驗證。各 run 的 data_quality_report.json 記錄額外缺值及排除，gain_fold_*.csv 保存 Gain。\n\n窗口暖機導致 Training 列數不同；驗證日期股票集合本次完全相同。增益因此是特徵工程與其必要暖機篩選的整體效果，尚不能完全歸因於新增數值。後續最小檢查應以相同 Training 列重訓 FEAT-00 作嚴格暖機控制，避免把資料列改變誤認為純特徵增益。Gain 僅輔助，未據此選群組。\n\n## 後續最小工作單\n\n1. 以相同訓練列建立暖機控制，優先核對 OBJ-03+法人、OBJ-03+法人價格、OBJ-00+法人。\n2. 查核法人資料發布延遲與單位，尤其百分比與股數。\n3. 事先鎖定至多兩個候選後，才做一次 2025 confirmation；本次未執行。\n4. E3 先不執行；待暖機與確認檢查支持後才考慮 16–24 次 seed=42 random search，learning_rate 與樹數共同取樣；early stopping 只能使用 outer training 內的時間序 inner validation。\n\nFEAT-04 技術、FEAT-05 全球、E4 windows/Top100、E5 日期 block bootstrap 均列待辦，不執行。沒有確認期或不確定性證據前，這些是開發候選而非已確認泛化改善。\n\n來源：../comparison.csv、../experiment_registry.csv、各 run configs / quality / gain。\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--ids',nargs='*')
    parser.add_argument('--reports',action='store_true')
    a=parser.parse_args()
    if a.reports: reports()
    else: print(compare(a.ids).to_string(index=False))
