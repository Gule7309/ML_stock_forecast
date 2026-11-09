FEATURE_MAPPING = {
    "etl:adj_open": "adj_open",
    "etl:adj_high": "adj_high",
    "etl:adj_low": "adj_low",
    "etl:adj_close": "adj_close",
    "price:成交股數": "volume",
    "price:成交金額": "turnover",
    "price:成交筆數": "transactions",
    "foreign_investors_shareholding:全體外資及陸資持股比率": "foreign_holding_ratio",
    "RSI": "rsi",
    "SMA": "sma",
    "^VIX": "vix",
}
STOCK_COLUMNS = list(FEATURE_MAPPING)[:-1]
FEATURES = list(FEATURE_MAPPING.values())
FOLDS = [dict(fold=i, train_start="2020-01-01", train_end=f"{year-1}-12-31",
              val_start=f"{year}-01-01", val_end=f"{year}-12-31")
         for i, year in enumerate((2023, 2024, 2025), 1)]
PARAMS = dict(boosting_type="gbdt", objective="binary", n_estimators=200,
              learning_rate=0.05, num_leaves=15, max_depth=-1,
              min_child_samples=100, reg_lambda=1.0, reg_alpha=0.0,
              subsample=1.0, colsample_bytree=1.0, random_state=42)
