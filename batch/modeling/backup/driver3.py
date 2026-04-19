########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test_driver,
    )
from batch.modeling.visualize import (
    plot_driver_soft_label,
    )
from batch.modeling.featuring import (
    get_columns_by_frequency,
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################
driver_index = {
    # 教師ラベル
    "^GSPC": 1,
    "TLT": 1,
    "BAMLH0A0HYM2": 1,
    "BAA10Y":1,

    # アンカー
    "RRPONTSYD":1,
    "WALCL":1,
    "WDTGAL":1,
    "WCURCIR":1,
    "NFCI":5,
    "CPF3M":1,
    "DTB3":1,
    "DX-Y.NYB":1,
    "HG=F":1,
    "GC=F":1,
    "DFII10":1,
    "NDFACBM027SBOG":1,
    "T10YIE":1,

    # 火薬
    "VIXCLS":1,
    "^MOVE":1,
    "VVIX":1,
    "EFFR":1,
    "SOFR":1,
    "DGS10":1,
    "DGS2":1,
    "DGS3MO":1,
    "CL=F":1,
    "STLFSI4":1
    }

def get_driver_beta(df_index, df_sp500):
    # --- データの取得：マスターデータからLiq_eff_modelに必要なデータを取り出す ---
    keys_list = list(driver_index.keys())
    df = df_index[keys_list]
    #check_nan_time(df, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df.tail(20))
    #print(df["BAMLH0A0HYM2"].dropna().head(10))
    #print(df["BAA10Y"].dropna().head(10))
    
    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df[["^GSPC","TLT","BAMLH0A0HYM2","BAA10Y"]])
    #check_nan_time(df_label, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_label.tail(50))


    # --- データ集計：日時、週次、月次を、すべて日次にする ---
    df_agg_daily = _aggregation_daily(df)
    #check_nan_time(df_agg_daily, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_agg_daily.tail(50))

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_agg_daily)
    
    #check_nan_time(df_features, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_features.tail(20))

    features_refined = {
        # アンカー
        "Liq_eff":0,
        #"Real_Yield_Level":0,
        #"Real_Yield_gap_ma500":0,
        #"T10YIE":0,
        #"T10YIE_gap_ma500":0,
        #"DXY_Level":0,
        "DXY_gap_ma500":0,
        "Cu_Au_Ratio":0,
        #"Cu_Au_Ratio_gap_ma500":0,
        #"cp_spread":0,
        #"cp_spread_gap_ma500":0,
        # Era
        "Era":0,
        # 火薬
        #"VIX_Accel":0,
        "MOVE_to_VIX_Ratio_z252":0,#
        #"VVIX_z252":0,#
        #"HY_diff5_z252":0,
        "OAS_to_VIX_Ratio_z252":0,#
        #"cp_spread_z252":0,
        #"rate_shock_z252",
        #"DFII10_diff5_z252":0,
        "Term_Premium_Momentum_z252":0,#
        "Curve_Steepening_Accel_z252":0,#
        #"Stock_Bond_Corr_z252":0,#
        #"Stock_Bond_Corr_raw",
        "Copper_Gold_Momentum_z252":0,#
        #"Equity_Gold_Ratio_z252":0,
        #"tlt_hy_ratio_z252":0,
        "stlfsi4":0
#
    }
    df_features = df_features[list(features_refined.keys())]
    monotone_constraints = list(features_refined.values())
    #print(monotone_constraints)

    # --- 学習モデル生成 ---
    # サンプルフェイト
    df_label['sample_weight'] = 1.0

    mask_credit = (df_label['driver'] == 1)
    df_label.loc[mask_credit, 'sample_weight'] = df_label['credit_score']
    mask_bond = (df_label['driver'] == 2)
    df_label.loc[mask_bond, 'sample_weight'] = df_label['bond_score']
    mask_bond = (df_label['driver'] == 3)
    df_label.loc[mask_bond, 'sample_weight'] = 0.8

    df_driver = df_features.join(df_label[["driver", "next_20d_diff_hy"]])
    start = df_driver.apply(pd.Series.first_valid_index).max()
    end = df_driver.apply(pd.Series.last_valid_index).min()
    df_driver = df_driver.loc[start:end]
    df_driver = df_driver.loc["2009-01-01":]
    #check_nan_time(df_driver, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_driver.tail(20))

    """driver_clf, df_driver_trajectory = learning_lgbm_final(
        df_driver, "driver", model_name="Driver", label_name_list=["1:Credit", "2:Bond", "3:Mix"],
        n_estimators=2800,learning_rate=0.001,num_leaves=50, min_data_in_leaf=100,
        reg_alpha=0.3, reg_lambda=0.3,)"""

    print(f"特徴量とmonotone_constraints設定: {features_refined}")
    df_oof_all, df_shap, df_oof_ev = learning_lgbm_test_driver(
        df_driver, "driver", labels=["1:Credit", "2:Bond", "3:Mix"],
        n_splits=5, gap =30,
        n_estimators=8000,learning_rate=0.005,
        num_leaves=35, min_data_in_leaf=35,max_depth=7,
        reg_alpha=0.5, reg_lambda=0.5,
        extra_trees="False",
        class_weight="balanced",
        monotone_constraints = None,
        importance_type='gain',
        sample_weight=None,#df_label["sample_weight"],
        learning_curve=True,
        )
    #shap_stats(df_driver, df_features.columns, df_shap)

     # --- ファイル保存 ---

    """df_oof_all.to_parquet("driver_oof.parquet", engine="pyarrow")
    df_shap["1:Credit"].to_parquet("driver_shap_credit.parquet", engine="pyarrow")
    df_shap["2:Bond"].to_parquet("driver_shap_bond.parquet", engine="pyarrow")
    df_shap["3:Mix"].to_parquet("driver_shap_mix.parquet", engine="pyarrow")
    df_oof_ev.to_parquet("driver_oof_ev.parquet", engine="pyarrow")
    df_daily.to_parquet("driver_daily.parquet", engine="pyarrow")
    df_driver.to_parquet("driver_driver.parquet", engine="pyarrow")
    df_features.to_parquet("driver_features.parquet", engine="pyarrow")
    df_label.to_parquet("driver_label.parquet", engine="pyarrow")"""

def _aggregation_daily(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")]
    df_weekly = df[get_columns_by_frequency(df, target="weekly")]
    df_monthly = df[get_columns_by_frequency(df, target="monthly")]
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")]

    print(f"--- 特徴量や目的変数で使う指標の頻度 ---")
    print(f"日次: {df_daily.columns.tolist()}")
    print(f"週次: {df_weekly.columns.tolist()}")
    print(f"月次: {df_monthly.columns.tolist()}")
    print(f"四半期: {df_quarterly.columns.tolist()}\n")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.dropna(how="all").tail(10))
    
    master_index = df["^GSPC"].dropna().index

    # 日次>日次
    lagged_series_list = []
    for col in df_daily.columns:
        # オリジナル
        s = df_daily[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 2)
        s.index = s.index + pd.Timedelta(days=lag_days)
        # 日次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_daily_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))

    # 週次>週次
    lagged_series_list = []
    for col in df_weekly.columns:
        # オリジナル
        s = df_weekly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 7)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 週次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_weekly_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))

    # 月次>週次
    df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)
    lagged_series_list = []
    for col in df_monthly.columns:
        # オリジナル
        s = df_monthly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 31)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 月次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_monthly_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_monthly_w_lagged,"1990-01-01")
    #print(df_monthly_w_lagged.tail(20))

    # 結合
    df_combine = pd.concat([df_daily_d_lagged, df_weekly_d_lagged, df_monthly_d_lagged], axis=1)
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

def _aggregation_weekly(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")]
    df_weekly = df[get_columns_by_frequency(df, target="weekly")]
    df_monthly = df[get_columns_by_frequency(df, target="monthly")]
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")]

    print(f"--- 特徴量や目的変数で使う指標の頻度 ---")
    print(f"日次: {df_daily.columns.tolist()}")
    print(f"週次: {df_weekly.columns.tolist()}")
    print(f"月次: {df_monthly.columns.tolist()}")
    print(f"四半期: {df_quarterly.columns.tolist()}\n")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.dropna(how="all").tail(10))

    # 日次>週次
    lagged_series_list = []
    for col in df_daily.columns:
        # オリジナル
        s = df_daily[col].dropna().copy()
        # ラグ
        lag_days = driver_index.get(col, 2)
        s.index = s.index + pd.Timedelta(days=lag_days)
        # 日次>週次
        s_w = s.resample("W-FRI").mean()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_daily_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))

    # 週次>週次
    lagged_series_list = []
    for col in df_weekly.columns:
        # オリジナル
        s = df_weekly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 7)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 週次>週次
        s_w = s.resample("W-FRI").ffill()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_weekly_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_weekly_w_lagged,"1990-01-01")
    #print(df_weekly_w_lagged.tail(20))

    # 月次>週次
    df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)
    lagged_series_list = []
    for col in df_monthly.columns:
        # オリジナル
        s = df_monthly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 31)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 月次>週次
        s_w = s.resample("W-FRI").ffill()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_monthly_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_monthly_w_lagged,"1990-01-01")
    #print(df_monthly_w_lagged.tail(20))

    # 結合
    df_combine = pd.concat([df_daily_w_lagged, df_weekly_w_lagged, df_monthly_w_lagged], axis=1)
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

########################################################
# 特徴量抽出
########################################################


def _featuring_all(df_daily):

    feats = pd.DataFrame(index=df_daily.index)
    master_index = df_daily["^GSPC"].dropna().index

    # --- アンカー ---
    feats = _featuring_anchors(df_daily, feats, master_index)

    # --- Era ---
    conditions = [
        (feats.index < '2010-10-01'),
        (feats.index >= '2010-10-01') & (feats.index < '2013-06-01'),
        (feats.index >= '2013-06-01') & (feats.index < '2019-09-01'),
        (feats.index >= '2019-09-01') & (feats.index < '2020-04-01'),
        (feats.index >= '2020-04-01') & (feats.index < '2021-12-01'),
        (feats.index >= '2021-12-01') & (feats.index < '2023-10-01'),
        (feats.index >= '2023-10-01')
    ]
    choices = [0, 1, 2, 3, 4, 5, 6]

    # 2. 条件に合致しない場合は最新のEra4とする
    feats['Era'] = np.select(conditions, choices, default=4)

    # 3. 【超重要】数値型からカテゴリ型へ明示的に変換
    feats['Era'] = feats['Era'].astype('category')

    # ---  恐怖の先行指標 - 初期震動の検知 ---
    feats = _vol_feats(df_daily, feats, master_index)

    # --- システムの目詰まり（Credit & Liquidity）- Creditレジュームを仕留める ---
    feats = _credit_liq_feats(df_daily, feats, master_index)

    # --- マクロの重力（Rates & Inflation）- Bondレジュームを仕留める ---
    feats = _macro_gravity_feats(df_daily, feats, master_index)

    # --- 資金のうねり（Momentum & Flow）- Creditレジュームを仕留める ---
    feats = _momentum_flow_feats(df_daily, feats, master_index)
    


    # 開始日、終了日をを決める
    start = "2005-03-16"
    end = feats.apply(pd.Series.last_valid_index).min()
    feats = feats.loc[start:end]
    
    #check_nan_time(feats, date="2005-01-01")

    return feats

def _featuring_anchors(df_daily, feats, master_index):
    # Net Liquidity
    feats['RRP_filled'] = df_daily['RRPONTSYD'].fillna(0) * 1000
    feats["Net_Liquidity"] = (df_daily['WALCL'] - (df_daily['WDTGAL'] +  feats['RRP_filled'] + df_daily["WCURCIR"]))
    feats["Net_Liquidity_z252"] = _featuring_z_score(feats["Net_Liquidity"], window=252)
    #feats["Net_Liquidity_z500"] = _featuring_z_score(feats["Net_Liquidity"], window=500)

    # NFIC
    feats["NFCI_z252"] = _featuring_z_score(df_daily["NFCI"], window=252)
    #feats["NFCI_z500"] = _featuring_z_score(df_daily["NFCI"], window=500)

    # Liq_eff
    feats["Liq_eff"] = (feats["Net_Liquidity_z252"] - feats["NFCI_z252"]).reindex(master_index, method="ffill")
    #feats["Liq_eff"] = (feats["Net_Liquidity_z500"] - feats["NFCI_z500"]).reindex(master_index, method="ffill")

    # 実質金利
    feats["Real_Yield_Level"] = df_daily["DFII10"].ewm(span=5, adjust=False).mean().reindex(master_index, method="ffill")
    ma500 = df_daily["DFII10"].ewm(span=500, adjust=False).mean().reindex(master_index, method="ffill")
    feats["Real_Yield_gap_ma500"] = df_daily["DFII10"] -ma500

    # ドル指標
    feats['DXY_Level'] = df_daily["DX-Y.NYB"].ewm(span=5, adjust=False).mean().reindex(master_index, method='ffill')
    feats['DXY_Level_z252'] = _featuring_z_score(feats['DXY_Level'], window=252).reindex(master_index, method='ffill')
    ma500 = df_daily["DX-Y.NYB"].ewm(span=500, adjust=False).mean().reindex(master_index, method="ffill")
    feats["DXY_gap_ma500"] = df_daily["DX-Y.NYB"] -ma500

    # Copper/Gold Ratio (リスクオンセンチメント)
    feats["Cu_Au_Ratio"] = df_daily["HG=F"] / df_daily["GC=F"]
    feats["Cu_Au_Ratio_z252"] = _featuring_z_score(feats["Cu_Au_Ratio"], 252).reindex(master_index, method='ffill')
    ma500 = feats["Cu_Au_Ratio"].ewm(span=500, adjust=False).mean().reindex(master_index, method="ffill")
    feats["Cu_Au_Ratio_gap_ma500"] = feats["Cu_Au_Ratio"] -ma500
    
    # cp spread
    cpf3m = df_daily["CPF3M"].dropna()
    dtb3 = df_daily["DTB3"].dropna()
    feats["cp_spread"] = (cpf3m - dtb3).reindex(master_index, method="ffill")
    feats["cp_spread_z252"] = _featuring_z_score((cpf3m - dtb3).dropna(), window=252).reindex(master_index, method="ffill")
    ma500 = feats["cp_spread"] .ewm(span=500, adjust=False).mean().reindex(master_index, method="ffill")
    feats["cp_spread_gap_ma500"] = feats["cp_spread"]  -ma500
    
    feats["T10YIE"] = df_daily["T10YIE"].dropna()
    feats["T10YIE_z252"] = _featuring_z_score(feats["T10YIE"], window=252).reindex(master_index, method="ffill")
    ma500 = df_daily["T10YIE"].ewm(span=500, adjust=False).mean().reindex(master_index, method="ffill")
    feats["T10YIE_gap_ma500"] =  df_daily["T10YIE"]  -ma500

    return feats

def _vol_feats(df, feats, master_index):
    # 指標
    vix = df['VIXCLS'].dropna()
    vvix = df["VVIX"].dropna()
    move = df["^MOVE"].dropna()

    feats["VIX_Accel"] = vix.rolling(5).mean() / vix.rolling(21).mean() -1
    ratio = move / vix
    ratio = ratio.ffill()
    feats['MOVE_to_VIX_Ratio_z252'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")
    feats["VVIX_z252"] = _featuring_z_score(vvix, window=252).reindex(master_index, method="ffill")
    
    #print(feats[["VIX_Accel", "MOVE_to_VIX_Ratio_z252", "VVIX_z252"]].dropna(how="all"))

    return feats

def _credit_liq_feats(df, feats, master_index):
    # 指標
    #hy = df['BAMLH0A0HYM2'].dropna()
    hy = df["BAA10Y"].dropna()
    sofr = df["SOFR"].dropna()
    effr = df["EFFR"].dropna()

    vix = df["VIXCLS"].dropna()
    cpf3m = df["CPF3M"].dropna()
    dtb3 = df["DTB3"].dropna()

    hy_diff = hy.diff(5)
    feats['HY_diff5_z252'] = _featuring_z_score(hy_diff, window=252).reindex(master_index, method="ffill")
    ratio = hy / vix
    ratio = ratio.ffill()
    feats["OAS_to_VIX_Ratio_z252"] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")
    idx = sofr.index.union(effr.index)
    short_rate = sofr.reindex(idx).combine_first(effr.reindex(idx))
    rate_diff_5d = short_rate.diff(5)
    feats['rate_shock_z252'] = _featuring_z_score(rate_diff_5d, window=252).clip(lower=0).reindex(master_index, method="ffill")
    #pd.set_option("display.max.rows", None)
    #print(feats["rate_shock_z"].dropna().head(100))

    return feats

def _macro_gravity_feats(df, feats, master_index):
    # 指標
    dfii10 = df["DFII10"].dropna()
    dgs10 = df["DGS10"].dropna()
    dgs2 = df["DGS2"].dropna()
    dgs3mo = df["DGS3MO"].dropna()
    stlfsi4 = df["STLFSI4"].dropna()

    feats['DFII10_diff5_z252'] = _featuring_z_score(dfii10.diff(5), window=252).reindex(master_index, method="ffill")
    term_premium = dgs10 - dgs3mo
    feats["Term_Premium_Momentum_z252"] = _featuring_z_score(term_premium.diff(5), window=252).reindex(master_index, method="ffill")
    curve10y2y = dgs10 - dgs2
    feats["Curve_Steepening_Accel_z252"] = _featuring_z_score(curve10y2y.diff(5), window=252).reindex(master_index, method="ffill")
    feats["stlfsi4"] = stlfsi4.reindex(master_index, method="ffill")
    feats["stlfsi4_z252"] = _featuring_z_score(stlfsi4, window=252).reindex(master_index, method="ffill")

    return feats

def _momentum_flow_feats(df, feats, master_index):
    # 指標
    dxy = df["DX-Y.NYB"].dropna()
    gold = df["GC=F"].dropna()
    sp500 = df["^GSPC"].dropna()
    tlt = df["TLT"].dropna()
    #hy = df["BAMLH0A0HYM2"].dropna()
    hy = df["BAA10Y"].dropna()
    cu = df["HG=F"].dropna()

    returns_sp = sp500.pct_change()
    returns_tlt = tlt.pct_change()
    corr20d = returns_sp.rolling(20, min_periods=10).corr(returns_tlt)
    feats['Stock_Bond_Corr_z252'] = _featuring_z_score(corr20d, window=252).reindex(master_index, method="ffill")
    feats['Stock_Bond_Corr_raw'] = corr20d.reindex(master_index, method="ffill")
    ratio = cu / gold
    ratio = ratio.ffill()
    feats['Copper_Gold_Momentum_z252'] = _featuring_z_score(ratio.pct_change(20), window=252).reindex(master_index, method="ffill")
    equity_gold = sp500 / gold
    equity_gold = equity_gold.ffill()
    feats['Equity_Gold_Ratio_z252'] = _featuring_z_score(equity_gold, window=252).reindex(master_index, method="ffill")

    ratio = tlt / hy
    ratio = ratio.ffill()
    feats['tlt_hy_ratio_z252'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")

    return feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# 教師ラベル作成 - カンニングラベル
########################################################

def _make_label(df, smear_days=5):
    future_lag = 20

    # リバランスの実行基準となるマスターカレンダー
    master_index = df["^GSPC"].dropna().index
    df_label = pd.DataFrame(index=master_index)

    # --- Step 1: 現在の常識（閾値）の計算 ---
    sp500_clean = df['^GSPC'].dropna()
    sp500_vol_20d_current = sp500_clean.pct_change(future_lag).rolling(252, min_periods=60).std().reindex(master_index, method='ffill')

    tlt_clean = df["TLT"].dropna()
    tlt_vol_20d_current = tlt_clean.pct_change(future_lag).rolling(252, min_periods=60).std().reindex(master_index, method='ffill')

    #hy_clean = df["BAMLH0A0HYM2"].dropna()
    #hy_diff_20d_current_vol = hy_clean.diff(future_lag).rolling(252, min_periods=60).std().reindex(master_index, method='ffill')
    
    hy_clean = df["BAA10Y"].dropna()
    hy_diff_20d_current_vol = hy_clean.diff(future_lag).rolling(252, min_periods=60).std().reindex(master_index, method='ffill')

    # --- Step 2: 未来の事実（ターゲット）の計算 ---
    future_sp500_ret = sp500_clean.pct_change(future_lag).shift(-future_lag)
    df_label['next_20d_ret_sp500'] = future_sp500_ret.reindex(master_index, method="ffill")

    future_tlt_ret = tlt_clean.pct_change(future_lag).shift(-future_lag)
    df_label['next_20d_ret_tlt'] = future_tlt_ret.reindex(master_index, method="ffill")

    future_hy_diff = hy_clean.diff(future_lag).shift(-future_lag)
    df_label['next_20d_diff_hy'] = future_hy_diff.reindex(master_index, method="ffill")

    # --- Step 3: 生のフラグ（Raw Flags）を立てる ---
    # Credit: HYスプレッドの異常な拡大 ＋ 株安
    raw_credit = (
        (df_label['next_20d_diff_hy'] > (1.75 * hy_diff_20d_current_vol)) #& # 20日差分が現在の2シグマを超える
        #(df_label['next_20d_ret_sp500'] < 0) & # 必ず株安を伴う
        #((df_label['next_20d_diff_hy'] / hy_diff_20d_current_vol) > (df_label['next_20d_ret_sp500'].abs() / sp500_vol_20d_current * 0.5))
    )
    # Bond: TLTの異常な変動 ＋ 株安
    raw_bond = (
        (df_label['next_20d_ret_tlt'].abs() > (1.75 * tlt_vol_20d_current))# &
        #(df_label['next_20d_ret_sp500'] < 0) & # ★ポジティブな金利上昇（株高）をノイズとして除外
        #((df_label['next_20d_ret_tlt'].abs() / tlt_vol_20d_current) > (df_label['next_20d_ret_sp500'].abs() / sp500_vol_20d_current * 0.5))
    )

    # --- Step 2: 数学的Smearing（減衰スコアの計算） ---
    def calculate_decay_score(raw_series, window):
        scores = np.zeros(len(raw_series))
        event_indices = np.where(raw_series)[0]

        # ハーフライフ（半減期）をwindowの半分に設定
        tau = window / 1.5
        for idx in event_indices:
            for d in range(window + 1):
                if idx - d >= 0:
                    decay_val = np.exp(-d / tau)
                    scores[idx - d] = max(scores[idx - d], decay_val)
        return scores

    # スコア（確信度 0.0 ~ 1.0）を算出
    df_label['credit_score'] = calculate_decay_score(raw_credit, smear_days)
    df_label['bond_score'] = calculate_decay_score(raw_bond, smear_days)

    # --- Step 5: スコアに基づく動的ラベル付与 ---
    threshold = 0.4
    df_label["driver"] = 3 # Neutral

    is_bond_candidate = df_label['bond_score'] > threshold
    is_credit_candidate = df_label['credit_score'] > threshold

    # 基本はスコアが高い方を採用（勝者総取り）
    df_label.loc[is_bond_candidate, 'driver'] = 2
    df_label.loc[is_credit_candidate & (df_label['credit_score'] >= df_label['bond_score']), 'driver'] = 1
    df_label.loc[is_bond_candidate & (df_label['bond_score'] > df_label['credit_score']), 'driver'] = 2

    df_label = df_label.dropna()

    #_analysis_label(df_label, df)

    return df_label

def _analysis_label(df, df_daily):
    df_origin = df.copy()
    terms = [
        ("2010-10-01","2013-06-01"),("2013-06-01","2016-10-01"),("2016-10-01","2019-09-01"),
        ("2019-09-01","2020-04-01"),("2020-04-01","2021-12-01"),("2021-12-01","2023-10-01"),
        ("2023-10-01","2026-02-01"),("2010-10-01","2021-12-01"),("2021-12-01","2026-02-01"),
        ]
    for start,end in terms:
        print(f"\nDriver教師ラベルの期間 : {start}〜{end}")

        df_sub = df_origin.loc[start:end]
        #df_daily_sub = df_daily.loc[start:end]

        # 分析・可視化
        stats = df_sub['driver'].value_counts().to_frame(name='Count')
        stats['Percentage (%)'] = (df_sub['driver'].value_counts(normalize=True) * 100).round(2)
        print(stats)

        market_summary = df_sub.groupby('driver').agg({
            'next_20d_ret_sp500': [
                'count', 'mean', 'median', 'std', 
                lambda x: x.quantile(0.05), 'min', # 下位5%と最小値でリスクの深さを測る
                lambda x: (x > 0).mean() # 勝率
            ],
            'next_20d_ret_tlt': ['mean', 'std', 'min'],
            'next_20d_diff_hy': ['mean', 'std', 'max'] # HYは拡大(max)がリスク
        }).round(4)

        # カラム名を分かりやすく整理（任意）
        market_summary.columns = [
            'count', 'ret_mean', 'ret_median', 'ret_std', 
            'ret_q05', 'ret_min', 'win_rate',
            'tlt_mean', 'tlt_std', 'tlt_min',
            'hy_diff_mean', 'hy_diff_std', 'hy_diff_max'
        ]
        print(market_summary)

        # 継続日数の算出
        df_sub['change'] = df_sub['driver'] != df_sub['driver'].shift()
        df_sub['regime_id'] = df_sub['change'].cumsum()

        # 各期間の長さをカウント
        duration_stats = df_sub.groupby(['regime_id', 'driver']).size().reset_index(name='duration')
        avg_duration = duration_stats.groupby('driver')['duration'].mean().round(1)
        print(f"平均継続日数:\n{avg_duration}")

        # 遷移マトリクス（現在の状態 -> 次の状態）
        transition_matrix = pd.crosstab(
            df_sub['driver'], 
            df_sub['driver'].shift(-1), 
            normalize='index'
        ).round(2)

        print("遷移マトリクス（行：現在 -> 列：次）:")
        print(transition_matrix)
        #plot_driver_soft_label(df, df_daily, start_date=s_date, end_date=e_date)

########################################################
# 実装確認・デバッグ
########################################################
def check_nan_time(df, date:str="2006-01-01"):
    df_s = df.apply(pd.Series.first_valid_index)
    df_e = df.apply(pd.Series.last_valid_index)
    print("")
    print("--- Nanの数 ---")
    print(df.isna().sum())
    print("")
    print("--- " + date + "以降でデータが入っている最初の日付 ---")
    print(df_s[df_s > date])
    print("")
    print("--- データが終わっている日付 ---")
    print(df_e)

def _chk_ev_hist(df_oof_ev):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. 基本統計量の確認（平均、最小、最大、四分位数）
    print("=== risk_score の基本統計量 ===")
    print(df_oof_ev['risk_score'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df_oof_ev['risk_score'], bins=50, kde=True, color='royalblue')
    plt.title('Distribution of Risk Sum (Credit + 0.5*Bond Probability)')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.show()

def shap_stats(df_master, features_list, df_shap):
    for label, shap_df in df_shap.items():
        print(f"\n=== レジーム: {label} の符号検証 ===")
        # 検証データ期間の元の特徴量を取得
        original_X = df_master.loc[shap_df.index, features_list]

        logic_results = []
        for col in features_list:
            # 元の値とSHAP値の相関を計算
            correlation = original_X[col].corr(shap_df[col])

            # 方向性の判定
            direction = "正の相関 (+)" if correlation > 0 else "負の相関 (-)"
            logic_results.append({
                "特徴量": col,
                "方向性": direction,
                "相関係数": f"{correlation:.3f}"
            })

        print(pd.DataFrame(logic_results))
