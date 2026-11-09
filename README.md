# 股票預測 C26＋C21｜Phase 1 官方 AR 14.39 bp

本庫公開交接 2026-11-08 已提交的 C26＋C21 版本，官方 Phase 1 顯示 AR＝0.001439＝0.1439%＝14.39 bp。這是整份混合路由 CSV 的成績，不是 C26 單一模型分數；未核對全站名次。先前隊內版本為 7.02 bp，本版增加 7.37 bp。

## 安裝

建議 Windows / Python 3.12。先執行 `git clone https://github.com/Gule7309/ML_stock_forecast.git`，再 `cd ML_stock_forecast`；以下指令均在本 README 所在目錄執行。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m lightgbm_baseline.experiments.phase2_daily --help
```

`requirements.txt` 是原驗證環境的完整版本紀錄。官方資料另行從官方來源取得，放進 `inputs/`；本包不包含資料 ZIP、帳密、cookies 或其他研究筆記。

## 包內內容

- C26 原始推論程式與其 Python 匯入相依程式。
- 凍結的 20 個 horizon 模型、原設定與模型雜湊。
- 每日 Phase2 入口；只產生離線稽核／草稿，沒有上傳功能。
- `HANDOFF_SHA256.json`：交接時的檔案指紋，可追蹤版本；不是防複製或數位簽章。

每日流程只用 `h01.txt`，43 個特徵，模型訓練截至 2025 年底。其他 horizon 模型保留作版本交接，本包不包含完整研究快取與重新訓練流程。不要直接執行研究模組的實驗入口。

## 重現最高分 Phase 1 版本

已提交檔位於 [submissions/phase1_lgbm_C26_C21_20261108.csv](submissions/phase1_lgbm_C26_C21_20261108.csv)，SHA256 為 `8f2ad9126dc1f02e4ed4db5e6027afc90816722ba7cb0540b5e57113819f7bea`。

```powershell
.\.venv\Scripts\python.exe predict_phase1.py --data-zip inputs/fin_pred_data.zip --template inputs/sample_submission.csv --output phase1_replayed.csv
```

使用當時官方原始 ZIP 與樣板；程式會檢查兩者與模型的 SHA，並驗證 4200 個值逐一等於已計分 CSV，既有輸出拒絕覆寫。此入口不重新訓練、不讀 2026 標籤，也不自動提交。

路由：6–8 月 64 個預測日各以當日前歷史使用 C26 h1（43 特徵）；9/1 使用同一 C26 h1；9 月其餘 19 日固定 9/1 起點，使用 C21 h2–h20（40 特徵）。h2–h20 不加入 C26 的三個排名特徵。

原 C26 的 2023／2024 開發 AR 為 9.0006／9.0096 bp，合併 9.0051 bp；這些開發結果不能和官方 14.39 bp 混用。兩組趨勢／跳空新增特徵沒有入選本次提交。

## 每日使用

準備原始 `fin_pred_data.zip`、當日官方 `data.zip`、當日 50 股樣板 `template.csv`，以及依官方更新狀態建立的 `availability.json`。

availability 格式如下；日期與時間必須換成實際觀察值，不能把範例當成資料完整的證據。`dates` 必須涵蓋本次會用到的 live 日期；只有官方確認完整的日期才能填 `complete`，確認休市且資料不完整才填 `closed_incomplete`。

```json
{"observed_at":"2026-11-09T08:00:00+08:00","dates":{"2026-11-06":{"other_features":"complete"}}}
```

以下日期僅示範參數格式。先將樣板與來源日換成當日官方資料，每次使用新的 `TEAM...` run ID。

```powershell
.\.venv\Scripts\python.exe -m lightgbm_baseline.experiments.phase2_daily --stage audit --data-zip inputs/fin_pred_data.zip --live-zip inputs/data.zip --availability inputs/availability.json --template inputs/template.csv --expected-source-date 2026-11-06 --run-id TEAM_AUDIT_001
```

稽核通過後，以全新 run ID 和 `--stage draft` 重跑。輸出位於 `lightgbm_baseline/experiments/next_stage/<run-id>/`，`phase2_draft.csv` 才是每日提交草稿。核對 `status.json`、資料日期與 50 股順序後，由隊內指定成員依當日競賽規則提交。

程式於預測日台北 09:00 後拒絕產生當日草稿；來源過舊、缺失、重複鍵、重疊資料衝突或未確認完整也會中止。不要填零、關掉檢查或改用舊檔繞過錯誤。法人流量的真值 −1 不應一律當缺值。

## 分數與限制

回歸輸出大於 0 判為 1，其餘判為 0；每日預測不讀 2026 標籤。官方整體分數不能歸因為單一 C26 元件分數，離線 AR 也不能當官方分數。法人逐欄發布時點與復權歷史版本仍有原流程的限制。

## 隊內使用限制

本庫依作者決定公開供檢視；公開可讀不代表授權其他隊伍使用。只有作者明確授權的本隊成員可在本隊專案範圍執行、修改及提交。未經書面許可，不得轉用於其他隊伍、重發或宣稱為自己的作品，詳見 `LICENSE`。第三方套件與官方資料另依各自條款。

公開庫的程式、模型及 CSV 可被下載；LICENSE 與版本指紋只記錄使用限制及版本，無法技術性防止複製。
