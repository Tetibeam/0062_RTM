########################################################
# Liquidity予測モデル
########################################################
from batch.modeling.featuring import (
    get_columns_by_frequency,
    cap_by_sigma,
    standard_scalar_df,
    balanced_clip,
    cap_outliers
    )
from batch.modeling.visualize import (
    plot_index,_plot_graphs,
    _plot_lag_correlation,
    plot_gli_trajectory
    )
from batch.modeling.learning import (
    lag_analysis,
    learning_dfa,
    learning_lgbm_test_gli,
    learning_logistic_lasso_test
    )
import pandas as pd
import numpy as np

liq_index = {
    "^GSPC": 1,
    "TLT": 1,
    "BAMLH0A0HYM2": 1,
    "RRPONTSYD":1,
    "WALCL":1,
    "WDTGAL":1,
    "WCURCIR":1,
    "NFCI":5,
    "DTB3":1,
    "DX-Y.NYB":1,
    "HG=F":1,
    "GC=F":1,
    "DFII10":1,
    "NDFACBM027SBOG":1,
    "CPF3M":1
}

########################################################
# メインプロセス
########################################################
def get_liq_index_model_beta(df_index):
    # --- データの取得：マスターデータからLiq_eff_modelに必要なデータを取り出す ---
    keys_list = list(liq_index.keys())
    df = df_index[keys_list]

    # --- データ集計：日時、週次、月次、四半期を、すべて週次にする ---
    df_agg_weekly = _aggregation_weekly(df)

    # --- データ集計：日時、週次、月次、四半期を、すべて日次にする ---
    df_agg_daily = _aggregation_daily(df)

    # --- 特徴量を作る（週次） ---
    df_features_weekly =  _featuring(df_agg_weekly, 52)
    #check_nan_time(df_features_weekly, "1990-01-01")

    # --- 特徴量を作る（日次） ---
    df_features_daily =  _featuring(df_agg_daily, 252)
    #check_nan_time(df_features_daily, "1990-01-01")

    # --- 保存 ---
    save_model(df_features_weekly, df_features_daily)

########################################################
# データ集計
########################################################

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
        lag_days = liq_index.get(col, 2)
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
        lag_days = liq_index.get(col, 7)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 週次>週次
        s_w = s.resample("W-FRI").mean()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_weekly_w_lagged = pd.concat(lagged_series_list, axis=1)
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
        lag_days = liq_index.get(col, 31)
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
        lag_days = liq_index.get(col, 2)
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
        lag_days = liq_index.get(col, 7)
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
        lag_days = liq_index.get(col, 31)
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

########################################################
# 特徴量
########################################################

def _featuring(df, window):
    df_feats = df.dropna(how="all")

    # Net Liquidity
    df_feats['RRP_filled'] = df_feats['RRPONTSYD'].fillna(0) * 1000
    df_feats["Net_Liquidity"] = (df_feats['WALCL'] - (df_feats['WDTGAL'] +  df_feats['RRP_filled'] + df_feats["WCURCIR"]))
    df_feats[f"Net_Liquidity_z{window}"] = _featuring_z_score(df_feats["Net_Liquidity"], window=window)

    # NFIC
    df_feats[f"NFCI_z{window}"] = _featuring_z_score(df_feats["NFCI"], window=window)

    # Liq_eff
    df_feats["Liq_eff"] = df_feats[f"Net_Liquidity_z{window}"] - df_feats[f"NFCI_z{window}"]

    # Copper/Gold Ratio (リスクオンセンチメント)
    df_feats[f"Cu_Au_Ratio_z{window}"] = _featuring_z_score(df_feats["HG=F"] / df_feats["GC=F"], window)

    df_feats[f"NDFACBM027SBOG_z{window}"] = _featuring_z_score(df_feats["NDFACBM027SBOG"], window)

    # --- [Layer 1b] 重力 (Gravity) の算出 ---

    # 5. Real Rate (10年実質金利) -> 2022年以降の重力の主犯
    df_feats[f"DFII10_z{window}"] = _featuring_z_score(df_feats["DFII10"], window=window)

    # TB3MS_DFF
    df_feats["CPF3M_DTB3_Spread"] = df_feats["CPF3M"] - df_feats["DTB3"]
    df_feats[f"CPF3M_DTB3_Spread_z{window}"] = _featuring_z_score(df_feats["CPF3M_DTB3_Spread"], window=window)

    # DXY
    df_feats[f"DXY_z{window}"] = _featuring_z_score(df_feats["DX-Y.NYB"], window=window)

    df_feats = df_feats.dropna(how="all")
    #check_nan_time(df_feats, "1990-01-01")
    return df_feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# デバッグ・保存
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

def save_model(df_features_weekly, df_features_daily):

    df_features_weekly.to_parquet("liq_engine_feats_w.parquet", engine="pyarrow")
    df_features_daily.to_parquet("liq_engine_feats_d.parquet", engine="pyarrow")

