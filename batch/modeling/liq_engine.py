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
    "DFF":1,
    "DX-Y.NYB":1,
    "HG=F":1,
    "GC=F":1,
    "DFII10":1,
    "M2SL":31,
    "TOTBKCR":7,
    "INDPRO":31,
    "THREEFYTP10":1,
    "DFII10":1,
}

########################################################
# メインプロセス
########################################################
def get_liq_index_model_beta(df_index):
    # --- データの取得：マスターデータからLiq_eff_modelに必要なデータを取り出す ---
    keys_list = list(liq_index.keys())
    df = df_index[keys_list]

    # --- データ集計：日時、週次、月次、四半期を、すべて日次にする ---
    df_agg = _aggregation(df)

    # --- 教師ラベルの生成 ---
    # 教師ラベルは発表ラグは関係ないので生値でつくる
    df_label = _make_label(df, target_col="^GSPC", weeks_ahead=8, num_classes=3)

    # --- 特徴量を作る ---
    df_features =  _featuring(df_agg)

    # --- 特徴量と目的変数のラグ相関分析 ---
    #_lag_corr_check(df_features, df_target_var)

    # --- 特徴量の選択 ---
    df_features = df_features[[
        "Liq_eff",
        "Liq_eff_diff20",
        #"Net_Liquidity_z252",
        #"NFCI_z252",
        #"DTB3_DFF_Spread",
        "DXY_roc65",
        #"Marshallian_K",
        "Credit_Growth_z252",
        "Cu_Au_Ratio_z252",
        #"Real_Rate_Gravity",
        #"Term_Premium",
        "HY_Spread_Momentum",
        ]]

    print(f"特徴量: {df_features.tail()}")
    #pd.set_option("display.max_rows", None)
    #print(df_features)
    #check_nan_time(df_features, "1900-01-01")

    # --- 学習用マスターデータの作成
    df_master = df_label[["target_label"]].join(df_features, how='left')
    df_master = df_master.dropna(subset=["target_label"])
    start = df_master.apply(pd.Series.first_valid_index).max()
    #start = "2010-01-01"
    df_master = df_master.loc[start:]
    check_nan_time(df_master, "1900-01-01")

    # --- LGBM学習 ---
    """df_oof_all, df_shap, df_oof_ev = learning_lgbm_test_gli(
        df_master, target_col="target_label",labels=["1:Bear", "2:Neutral", "3:Bull"],
        n_splits=3, gap=50,
        n_estimators=5000, learning_rate=0.001, num_leaves=21, min_data_in_leaf=65,
        reg_alpha=1, reg_lambda=1, max_depth=4,
        class_weight="balanced",extra_trees="True",
        importance_type="gain",stopping_rounds=100,
        feature_fraction=1.0,#bagging_fraction=0.5,bagging_freq=1,path_smooth=1.0,min_gain_to_split=0.1,
        #monotone_constraints=(1, 1, -1, 0) ,monotone_constraints_method="advanced",
        learning_curve=True,
    )"""

    # --- シャップ統計 ---
    #shap_stats(df_master, df_features.columns, df_shap)

    # --- リターン統計 ---
    #return_stats(df_agg, df_oof_ev, weeks_ahead=8)

    # --- Bearの確率ごとのリターン統計 ---
    #return_prob_stats(df_agg, df_oof_all, "1:Bear", weeks_ahead=8)
    # --- BULLの確率ごとのリターン統計 ---

    # --- 保存 ---
    #save_model(df_oof_all, df_shap, df_oof_ev, df_agg, df_master, df_features, df_label)

    # --- ロジスティック回帰 ---
    """mean_coefs, all_y_probs, all_y_test = learning_logistic_lasso_test(
        df_master, target_col="Liq_eff_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=3, gap=13,solver='saga',max_iter=5000,
        C=0.1, penalty="l1",class_weight="balanced",
    )"""

########################################################
# データ集計
########################################################

def _aggregation(df):

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

    # 日次営業日をマスターにする
    master_index = df["^GSPC"].dropna().index
    def adj_lag(df, index_list):
        lagged_series_list = []
        for col in index_list:
            s = df[col].dropna().copy()  # オリジナルのインデックス
            #print(s.tail(20))
            lag_days = liq_index.get(col, 7)
            s.index = s.index + pd.Timedelta(days=lag_days) # ラグ
            #print(s.tail(20))
            s = s.resample("D").ffill() # 日次
            #print(s.tail(20))
            s_d = s.reindex(master_index).ffill() #日次マスター
            lagged_series_list.append(s_d)
        df_lagged = pd.concat(lagged_series_list, axis=1)
        return df_lagged.dropna(how="all")

    # 日次、週次
    df_daily_d_lagged = adj_lag(df_daily, df_daily.columns)
    df_weekly_d_lagged = adj_lag(df_weekly, df_weekly.columns)
    # 月次
    df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)
    df_monthly_lagged = adj_lag(df_monthly, df_monthly.columns)

    #pd.set_option("display.max_rows", None)
    #check_nan_time(df_weekly_d_lagged,"1990-01-01")
    #print(df_weekly_d_lagged.tail(20))

    # 結合
    df_combine = pd.concat([df_daily_d_lagged, df_weekly_d_lagged, df_monthly_lagged], axis=1)
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

########################################################
# 特徴量
########################################################
def _featuring(df):
    df_feats = df.dropna(how="all")

    # Net Liquidity
    df_feats['RRP_filled'] = df_feats['RRPONTSYD'].fillna(0) * 1000
    df_feats["Net_Liquidity"] = (df_feats['WALCL'] - (df_feats['WDTGAL'] +  df_feats['RRP_filled'] + df_feats["WCURCIR"]))
    df_feats["Net_Liquidity_z252"] = _featuring_z_score(df_feats["Net_Liquidity"], window=252)

    # NFIC
    df_feats["NFCI_z252"] = _featuring_z_score(df_feats["NFCI"], window=252)

    # Liq_eff
    df_feats["Liq_eff"] = df_feats["Net_Liquidity_z252"] - df_feats["NFCI_z252"]
    df_feats["Liq_eff_ma5"] = df_feats["Liq_eff"].rolling(window=5).mean()
    df_feats["Liq_eff_diff20"] = df_feats["Liq_eff"].diff(20)

    # --- [Layer 1a] 浮力 (Buoyancy) の算出 ---

    # Marshallian K (余剰資金の勢い)
    m2_growth = df_feats["M2SL"].pct_change(252)
    ip_growth = df_feats["INDPRO"].pct_change(252)
    df_feats["Marshallian_K"] = m2_growth - ip_growth

    # Bank Credit Growth (信用の拡大)
    df_feats["Credit_Growth_z252"] = _featuring_z_score(df_feats["TOTBKCR"].pct_change(252), 252)

    # Copper/Gold Ratio (リスクオンセンチメント)
    df_feats["Cu_Au_Ratio_z252"] = _featuring_z_score(df_feats["HG=F"] / df_feats["GC=F"], 252)

    # --- [Layer 1b] 重力 (Gravity) の算出 ---

    # 5. Real Rate (10年実質金利) -> 2022年以降の重力の主犯
    df_feats["Real_Rate_Gravity"] = df_feats["DFII10"]

    # 6. Term Premium (タームプレミアム)
    df_feats["Term_Premium"] = df_feats["THREEFYTP10"]

    # 8. Credit Stress (ハイイールド債スプレッド)
    df_feats["HY_Spread_Momentum"] = df_feats["BAMLH0A0HYM2"].diff(20)

    # TB3MS_DFF
    df_feats["DTB3_DFF_Spread"] = df_feats["DTB3"] - df_feats["DFF"]
    # DXY
    df_feats["DXY_roc65"] = df_feats["DX-Y.NYB"].pct_change(65)

    df_feats = df_feats.dropna(how="all")
    #check_nan_time(df_feats, "1990-01-01")
    return df_feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# ラグ分析
########################################################
def _lag_corr_check(features, target):

    # GLI をdiffにする
    #target = target.loc["2010-01-01":]

    # GLI のインデックスに合わせる
    df_all = pd.concat([features.reindex(target.index), target], axis=1).dropna()

    # 特徴量とGLIのラグ相関分析
    df_lag = lag_analysis(df_all, target_col="Liq_eff", max_lag=52)

    # 結果の確認・デバッグ
    _plot_lag_correlation(df_lag)

########################################################
# 教師ラベル
########################################################

def _make_label(df, target_col="^GSPC", weeks_ahead=8, low_th=0.25, high_th=0.75):
    # 営業日換算のシフト日数を計算（1週間 = 5営業日）
    shift_days = weeks_ahead * 5

    df_out = df[[target_col]].dropna().copy()

    # 1. 指定期間先（未来）のフォワード・リターンを計算
    df_out['future_return'] = df['Close'].pct_change(shift_days).shift(-shift_days)
    valid_idx = df_out["fwd_ret"].dropna().index

    # 境界値の算出
    lower_bound = df['future_return'].quantile(low_th)
    upper_bound = df['future_return'].quantile(high_th)

    # ラベル付け: Bear(1.0), Bull(3.0), それ以外は NaN
    df['target_label'] = np.nan
    df.loc[df['future_return'] <= lower_bound, 'target_label'] = 1.0
    df.loc[df['future_return'] >= upper_bound, 'target_label'] = 3.0

    print(f"Bear (<= {lower_bound:.2%}) / Bull (>= {upper_bound:.2%})")
    print(f"学習対象データ数: {df['target_label'].notna().sum()} / 全体: {len(df)}")
    
    return df

    return df_out

########################################################
# 統計
########################################################
def _analysis_label(df):
    # 分析・可視化
    terms = [
        ("2012-01-01", "2026-01-01"),
        ("2012-01-01", "2015-01-01"),
        ("2015-01-01", "2018-01-01"),
        ("2018-01-01", "2021-01-01"),
        ("2021-01-01", "2024-01-01"),
        ("2024-01-01", "2026-01-01"),
        ]
    for start, end in terms:
        print(f"期間: {start} 〜 {end}")
        df_term = df.loc[start:end]
        stats = df_term['Liq_eff_label'].value_counts().to_frame(name='Count')
        stats['Percentage (%)'] = (df_term['Liq_eff_label'].value_counts(normalize=True) * 100).round(2)
        print(stats)

        market_summary = df_term.groupby('Liq_eff_label').agg({
            'next_xm_ret_sp500': ['mean', 'std', 'min', 'max', "count", lambda x: (x > 0).mean()],
            'next_xm_ret_tlt': ['mean', 'std', 'min', 'max'],
            'next_xm_diff_hy': ['mean', 'std', 'min', 'max']
        }).round(4)
        market_summary.columns = [
            "sp500_mean", "sp500_std", "sp500_min", "sp500_max", "counts", "勝率",
            "tlt_mean", "tlt_std", "tlt_min", "tlt_max",
            "hy_mean", "hy_std", "hy_min", "hy_max"]
        print(market_summary)

        #print(df[df["Liq_eff_label"]==1.0].index)

    """# 継続日数の算出
    df['change'] = df['Liq_eff_label'] != df['Liq_eff_label'].shift()
    df['regime_id'] = df['change'].cumsum()

    # 各期間の長さをカウント
    duration_stats = df.groupby(['regime_id', 'Liq_eff_label']).size().reset_index(name='duration')
    avg_duration = duration_stats.groupby('Liq_eff_label')['duration'].mean().round(1)
    print(f"平均継続日数:\n{avg_duration}")

    # 遷移マトリクス（現在の状態 -> 次の状態）
    transition_matrix = pd.crosstab(
        df['Liq_eff_label'],
        df['Liq_eff_label'].shift(-1),
        normalize='index'
    ).round(2)

    print("遷移マトリクス（行：現在 -> 列：次）:")
    print(transition_matrix)"""

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

def return_stats(df, df_oof_ev, weeks_ahead):
    # 営業日換算のシフト日数を計算（1週間 = 5営業日）
    shift_days = weeks_ahead * 5

    assets = df[["^GSPC"]].dropna() # 営業日カウントになる
    assets['next_xm_ret_sp500'] = assets["^GSPC"].pct_change(shift_days).shift(-shift_days)

    combined = pd.concat([df_oof_ev, assets[
        'next_xm_ret_sp500']], axis=1).dropna()

    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    print("=== ラベル数の確認 ===")
    print(df_oof_ev.value_counts())

    print("=== リターン統計  ===")
    terms = [
        ("2012-01-01", "2026-01-01"),
        ("2012-01-01", "2015-01-01"),
        ("2015-01-01", "2018-01-01"),
        ("2018-01-01", "2021-01-01"),
        ("2021-01-01", "2024-01-01"),
        ("2024-01-01", "2026-01-01"),
        ]
    for start,end in terms:
        print(f"\n--- 期間: {start} 〜 {end} ---")
        combined_tmp = combined.loc[start:end]
        stats = combined_tmp.groupby("predict_label").agg({
            'next_xm_ret_sp500': ['mean', 'std', 'min', 'max', "count", lambda x: (x > 0).mean()],
            #'next_xm_ret_tlt': ['mean', 'std', 'min', 'max'],
            #'next_xm_diff_hy': ['mean', 'std', 'min', 'max']
            })
        stats.columns = [
            "sp500_mean", "sp500_std", "sp500_min", "sp500_max", "counts", "勝率",
            #"tlt_mean", "tlt_std", "tlt_min", "tlt_max",
            #"hy_mean", "hy_std", "hy_min", "hy_max"
            ]
        print(stats)

def return_prob_stats(df, df_oof_all, label, weeks_ahead):

    # 営業日換算のシフト日数を計算（1週間 = 5営業日）
    shift_days = weeks_ahead * 5

    assets = df[["^GSPC"]].dropna() # 営業日カウントになる
    assets['next_xm_ret_sp500'] = assets["^GSPC"].pct_change(shift_days).shift(-shift_days)
    
    print("=== 基本統計量 ===")
    print(df_oof_all[label].describe())
    bins = [0, 0.28, 0.35, 1.1]
    df_oof_all['ev_rank'] = pd.cut(
        df_oof_all[label],
        bins=bins,
        labels=['Low', 'Middle', 'High'],
        include_lowest=True
    )
    combined = pd.concat([df_oof_all, assets['next_xm_ret_sp500']], axis=1).dropna()
    stats = combined.groupby("ev_rank").agg({
            'next_xm_ret_sp500': ['mean', 'std', 'min', 'max', "count", lambda x: (x > 0).mean()],
    })
    stats.columns = [
            "sp500_mean", "sp500_std", "sp500_min", "sp500_max", "counts", "勝率",
            #"tlt_mean", "tlt_std", "tlt_min", "tlt_max",
            #"hy_mean", "hy_std", "hy_min", "hy_max"
            ]
    print(stats)
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

def save_model(df_oof_all,df_shap,df_oof_ev,df,df_master,df_features,df_label):
    df_oof_all.to_parquet("gli_oof.parquet", engine="pyarrow")
    df_shap["1:STALL"].to_parquet("gli_shap_stall.parquet", engine="pyarrow")
    df_shap["2:CRUISE"].to_parquet("gli_shap_cruise.parquet", engine="pyarrow")
    df_shap["3:LIFT"].to_parquet("gli_shap_lift.parquet", engine="pyarrow")
    df_oof_ev.to_parquet("gli_oof_ev.parquet", engine="pyarrow")
    df.to_parquet("gli_raw.parquet", engine="pyarrow")
    df_master.to_parquet("gli_master.parquet", engine="pyarrow")
    df_features.to_parquet("gli_features.parquet", engine="pyarrow")
    df_label.to_parquet("gli_label.parquet", engine="pyarrow")
