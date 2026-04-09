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

liq_index = [
    "BUSLOANS",
    "PNFIC1",
    "SOFR",
    "TB3MS",
    "BAMLC0A4CBBB",
    "BAMLC0A3CA",
    "DX-Y.NYB",
    "RRPONTSYD",
    "WALCL",
    "RESBALNS",
    "TOTRESNS",
    "WDTGAL",
    "PAYEMS",
    "PCE",
    "NDFACBM027SBOG",
    "TEDRATE",
    "DTB3",
    "VXTLT",
    "WCURCIR",
    "DFF",
    "NFINCP",
    "PCEPI",
    "DFII10",
    "^GSPC",
    "ACWI",
    "TLT",
    "BAMLH0A0HYM2",
    "B069RC1",
    "DSPI",
    "EEM",
    "UUP",
    "gli",
    "NFCI",
    "VIXCLS",
    "^MOVE",
    "BAMLH0A3HYC",
    "NFCIRISK",
    "STLFSI4"
    ]

########################################################
# メインプロセス
########################################################
def get_liq_index_model_beta(df_index):
    # --- データの取得：マスターデータからLiq_eff_modelに必要なデータを取り出す ---
    df = df_index[liq_index]

    # --- 目的変数の生成：Liq_eff = NDFACBM027SBOG_z52 - NFCI_z52 ---
    df_target_var = _make_target_variable(df)

    # --- データ集計：日時、週次、月次、四半期を、すべて週次にする ---
    df_agg = _aggregation(df)

    # --- 教師ラベルの生成 ---
    df_label = _make_label(df_target_var["Liq_eff"], df_agg)

    # --- 特徴量を作る ---
    df_features =  _featuring(df_agg)

    # --- 特徴量と目的変数のラグ相関分析 ---
    #_lag_corr_check(df_features, df_target_var)

    # --- 特徴量の選択 ---
    df_features = df_features[[
        #"Abs_Rate_z52",
        'Net_Liquidity_z52',
        "Dollar_Squeeze_Index",
        'Burden_Ratio_z52',
        "PAYEMS_qoq_Abs_Rate_z52",
        #"Liquidity_Divergence",
        "MOVE_z52",
        #"VXTLT_z52",
        #"NDFACB_z52",
        "VIX_z52",
        #"VVIX_z52",
        #"NFCI_diff4_z52",
        "CCC_Spread_diff4",
        #"NFCIRISK_diff13_z52",
        #"HY_diff8_z52",
        #"HY_z52",
        #"Bank_Dependency_z52",
        #"TED_Z52",
        "DXY_z52",
        #"DXY_diff13_z52",
        "STLFSI4_z52"
        

    ]]
    print(f"特徴量のリスト: {df_features.columns}")

    # --- 学習用マスターデータの作成
    df_master = df_label.join(df_features, how='left')
    #df_master = df_master.loc["2010-01-01":].ffill()

    # --- LGBM学習 ---
    df_oof_all, df_shap, df_oof_ev = learning_lgbm_test_gli(
        df_master, target_col="Liq_eff_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=2, gap=10,
        n_estimators=12000, learning_rate=0.001, num_leaves=31, min_data_in_leaf=35,
        reg_alpha=0.5, reg_lambda=0.5, max_depth=5,
        class_weight="balanced",extra_trees="True",
        importance_type="gain",stopping_rounds=100,
        feature_fraction=0.5,#bagging_fraction=0.5,bagging_freq=1,path_smooth=1.0,min_gain_to_split=0.1,
        learning_curve=True,
    )

    # --- シャップ統計 ---
    #shap_stats(df_master, df_features.columns, df_shap)

    # --- リターン統計 ---
    return_stats(df_agg, df_oof_ev, 8)

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
    #print(df_daily.columns,df_weekly.columns,df_monthly.columns,df_quarterly.columns)
    #check_nan_time(df_quarterly,"1990-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.tail(10))

    df_daily_w = df_daily.resample('W-FRI').mean()
    df_daily_w_s = df_daily_w.rolling(window=4).mean().dropna(how="all")

    df_weekly_w = df_weekly.resample('W-FRI').mean()
    df_weekly_w_s = df_weekly_w.rolling(window=4).mean().dropna(how="all")

    df_monthly_w = df_monthly.dropna(how="all").resample('W-FRI').interpolate(method='linear')
    df_monthly_w_lagged = df_monthly_w.shift(4).dropna(how="all")

    q_m = df_quarterly.dropna(how="all")
    q_m.index = q_m.index + pd.offsets.MonthEnd(-1)
    q_m_lagged = q_m.shift(1)
    df_quarterly_w = q_m_lagged.resample('W-FRI').interpolate(method='linear')
    df_quarterly_w = df_quarterly_w.dropna(how="all")

    #pd.set_option("display.max_rows", None)
    #print(df_daily_w_s.tail(10))
    #check_nan_time(df_monthly_w_lagged)

    # 結合
    df_combine = pd.concat([df_daily_w_s, df_weekly_w_s, df_monthly_w_lagged, df_quarterly_w], axis=1)

    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_combine.dropna(how="all").tail(20))
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

def _make_target_variable(df):
    var = df[["NFCI", "NDFACBM027SBOG"]].copy().dropna(how="all")
    # 週次>週次（週末）
    var["NFCI"] = var["NFCI"].resample('W-FRI').mean().dropna()
    # 月次>月次（週末）
    var["NDFACBM027SBOG"] = var["NDFACBM027SBOG"].resample('W-FRI').interpolate(method='linear').dropna()

    var["NFCI_z52"] = _featuring_z_score(var["NFCI"], window=52)
    var["NDFACBM027SBOG_z52"] = _featuring_z_score(var["NDFACBM027SBOG"], 52)
    var["Liq_eff"] = var["NDFACBM027SBOG_z52"] - var["NFCI_z52"]
    #print(var[["Liq_eff"]].dropna())

    return var[["Liq_eff"]].dropna()

########################################################
# 特徴量
########################################################
def _featuring(df):
    df_feats = df.dropna(how="all")

    # --- 中央銀行による流動性供給（ベースマネーの増減）---

    df_feats['Unified_Reserves'] = df_feats['RESBALNS'].fillna(df_feats['TOTRESNS']) * 1000
    df_feats['RRP_filled'] = df_feats['RRPONTSYD'].fillna(0) * 1000
    # Net Liquidity
    df_feats["Net_Liquidity"] = (df_feats['WALCL'] - (df_feats['WDTGAL'] +  df_feats['RRP_filled'] + df_feats["WCURCIR"]))
    df_feats['Net_Liquidity_qoq'] = df_feats['Net_Liquidity'].pct_change(13)
    df_feats['Net_Liquidity_roc4'] = df_feats['Net_Liquidity'].pct_change(4)
    df_feats['Net_Liquidity_z52'] = _featuring_z_score(df_feats['Net_Liquidity'], 52)
    # 吸収率 (TGA+RRPが資産に占める割合)
    df_feats["Abs_Rate"] = ((df_feats['WDTGAL'] + df_feats['RRP_filled']) / df_feats['WALCL']).rename("Abs_Rate")
    df_feats["Abs_Rate_z52"] = _featuring_z_score(df_feats["Abs_Rate"], 52)
    # 準備預金の占有率
    df_feats['Res_Ratio'] = df_feats['Unified_Reserves'] / df_feats['WALCL']
     # 現金の漏出スピード
    df_feats['WCUR_Ratio'] = df_feats['WCURCIR'] / df_feats['WALCL']
    df_feats['WCUR_qoq'] = df_feats['WCURCIR'].pct_change(13)
    
    df_feats["WDTGAL_z52"] = _featuring_z_score(df_feats['WDTGAL'], 52)

    # --- グローバル・ドル調達圧力（国際流動性） ---

    df_feats['UUP_qoq'] = df_feats['UUP'].pct_change(13)
    df_feats['UUP_diff13'] = df_feats['UUP'].diff(13)
    df_feats['UUP_z52'] = _featuring_z_score(df_feats['UUP'], 52)
    # グローバルなドル圧迫
    df_feats['DXY_qoq'] = df_feats['DX-Y.NYB'].pct_change(13)
    df_feats['DXY_diff13_z52'] = _featuring_z_score(df_feats['DXY_qoq'].diff(13),52)
    df_feats['DXY_z52'] =  _featuring_z_score(df_feats['DX-Y.NYB'], 52)
    df_feats["Dollar_Squeeze_Index"] = df_feats["DXY_qoq"] - df_feats["Net_Liquidity_z52"]

    df_feats["NDFACB_z52_diff4"] = _featuring_z_score(df_feats['NDFACBM027SBOG'], 52).diff(4)
    df_feats["NDFACB_z52"] = _featuring_z_score(df_feats['NDFACBM027SBOG'], 52)

    # --- 民間部門の信用創造とレバレッジ（銀行の資金仲介) ---

    df_feats['Loan_qoq'] = df_feats['BUSLOANS'].pct_change(13)
    df_feats['NFINCP_qoq'] = df_feats['NFINCP'].pct_change(13)
    # 銀行依存度指標 (Loan / CP 比率)
    df_feats['Bank_Dependency'] = df_feats['BUSLOANS'] / df_feats['NFINCP']
    df_feats['Bank_Dependency_z52'] = _featuring_z_score(df_feats['Bank_Dependency'], 52)
    
    df_feats["STLFSI4_z52"] = _featuring_z_score(df_feats["STLFSI4"], 52)

    # --- 資本コストと市場のリスク許容度（リスクプレミアム） ---
    df_feats["SOFR"] = df_feats["SOFR"].fillna(df_feats["DFF"])

    # 短期指標のスプレッド
    df_feats['SOFR_TB3MS_Spread'] = df_feats['SOFR'] - df_feats['TB3MS']
    df_feats["TED_Z52"] = _featuring_z_score(df_feats["TEDRATE"], 52)

    # 信用スプレッド
    df_feats['spd_BBB_A'] = df_feats['BAMLC0A4CBBB'] - df_feats['BAMLC0A3CA']
    df_feats["HY_z52"] = _featuring_z_score(df_feats['BAMLH0A0HYM2'], 52)
    df_feats['HY_diff13'] = df_feats['BAMLH0A0HYM2'].diff(13)
    df_feats['HY_diff8_z52'] = _featuring_z_score(df_feats['BAMLH0A0HYM2'].diff(8), 52)
    df_feats["CCC_Spread_diff4"] = df_feats['BAMLH0A3HYC'].diff(4)

    # 債券市場のボラ
    df_feats['MOVE_z52'] = _featuring_z_score(df_feats['^MOVE'], 52)
    df_feats['VXTLT_z52'] = _featuring_z_score(df_feats['VXTLT'], 52)
    
    df_feats["VIX_z52"] = _featuring_z_score(df_feats['VIXCLS'], 52)
    ret = np.log(df_feats['VIXCLS']).diff()
    df_feats["VVIX_z52"] = _featuring_z_score(ret.rolling(21).std() * np.sqrt(252), 52)

    # --- 実体経済のファンダメンタルズと制約条件 ---
    # 実質金利のレベル感
    df_feats['DFII10'] = df_feats['DFII10']
    df_feats['DFII10_diff4'] = df_feats['DFII10'].diff(4)

    # インフレの勢い、雇用の勢い
    df_feats['PCEPI_yoy'] = df_feats['PCEPI'].pct_change(52) # 前年比
    
    df_feats['PAYEMS_qoq'] = df_feats['PAYEMS'].pct_change(13)     # 直近の加速
    df_feats['PAYEMS_qoq_sm13'] = df_feats['PAYEMS'].rolling(window=13).mean() 

    # 利払い負担率
    df_feats['Burden_Ratio'] = (df_feats['B069RC1'] / df_feats['DSPI']) * 100
    df_feats["Burden_Ratio_z52"] = _featuring_z_score(df_feats['Burden_Ratio'], 52)
    df_feats['Burden_diff13'] = df_feats['Burden_Ratio'].diff(13)
    df_feats['Burden_qoq'] = df_feats['Burden_Ratio'].pct_change(13)

    df_feats["NFCIRISK_diff13_z52"] = _featuring_z_score(df_feats['NFCIRISK'].diff(4), 52)
    df_feats["NFCI_z52"] = _featuring_z_score(df_feats['NFCI'], 52)
    df_feats["NFCI_diff4_z52"] = _featuring_z_score(df_feats["NFCI"].diff(4), window=52)
    df_feats["Liquidity_Divergence"] = df_feats["NDFACB_z52"] - df_feats["NFCI_z52"]

    # --- 交錯 ---
    df_feats["PAYEMS_qoq_Abs_Rate_z52"] = df_feats["PAYEMS_qoq"] * df_feats["Abs_Rate_z52"]
    df_feats["PAYEMS_qoq_DFII10"] = df_feats["PAYEMS_qoq"] * df_feats["DFII10"]
    df_feats["Liq_Interaction"] = df_feats["NDFACB_z52"] * df_feats["NFCI_z52"]

    return df_feats.dropna(how="all")

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# ラグ分析
########################################################
def _lag_corr_check(df_a, df_b, df_c, df_d, target):

    # GLI をdiffにする
    target_diff = target.diff(13).dropna().rename("NDFACBM027SBOG_diff")
    #target_diff = target_diff.loc["2008-01-01":]
    target_diff = target_diff.loc["2010-01-01":]

    # GLI のインデックスに合わせる
    df_a_all = pd.concat([df_a.reindex(target_diff.index), target_diff, target], axis=1).dropna()
    df_b_all = pd.concat([df_b.reindex(target_diff.index), target_diff, target], axis=1).dropna()
    df_c_all = pd.concat([df_c.reindex(target_diff.index), target_diff, target], axis=1).dropna()
    df_d_all = pd.concat([df_d.reindex(target_diff.index), target_diff, target], axis=1).dropna()

    """for feat  in [
        "level_SOFR","diff13_SOFR","diff52_SOFR",
        "z52_diff13_SOFR","z52_diff52_SOFR", "z52_SOFR",
        "z104_SOFR", "mom4_SOFR"
        ]:
        _plot_graphs(df_c_all["NDFACBM027SBOG_diff"], df_c_all[feat])"""


    # 特徴量とGLIのラグ相関分析
    df_lag_a = lag_analysis(df_a_all, target_col="NDFACBM027SBOG_diff", max_lag=156)
    df_lag_b = lag_analysis(df_b_all, target_col="NDFACBM027SBOG_diff", max_lag=156)
    df_lag_c = lag_analysis(df_c_all, target_col="NDFACBM027SBOG_diff", max_lag=156)
    df_lag_d = lag_analysis(df_d_all, target_col="NDFACBM027SBOG_diff", max_lag=156)

    # 結果の確認・デバッグ
    #_plot_lag_correlation(df_lag_a)
    #_plot_lag_correlation(df_lag_b)
    #_plot_lag_correlation(df_lag_c)
    _plot_lag_correlation(df_lag_d)

########################################################
# 教師ラベル
########################################################

def _make_label(target, df_index):
    #LAG=8
    quantile_low=0.125
    quantile_high=0.875
    winsow=156
    #plot_index(target.to_frame())

    for LAG in [8]:
        print(f"---------------- LAG:{LAG} ----------------")
        # Net Liquidity
        df_net_l = df_index[["RRPONTSYD", "WALCL", "WDTGAL", "WCURCIR"]].dropna(how="all")
        df_net_l['RRP_filled'] = df_net_l['RRPONTSYD'].fillna(0) * 1000
        df_net_l["Net_Liquidity"] = (df_net_l['WALCL'] - (df_net_l['WDTGAL'] +  df_net_l['RRP_filled'] + df_net_l["WCURCIR"]))
        df_net_l['net_liq_mom'] = df_net_l["Net_Liquidity"].pct_change(4).shift(-LAG)
        #print(df_net_l)

        # VIX
        df_vix = df_index[["VIXCLS"]].shift(-LAG).dropna()
        df_vix['vix_ma'] = df_vix['VIXCLS'].rolling(52).mean()
        df_vix['is_fear'] = df_vix['VIXCLS'] > df_vix['vix_ma']  # 心理的に「怖い」状態か
        df_vix = df_vix.dropna()
        #print(df_vix)

        # MOVE
        df_move = df_index[['^MOVE']].shift(-LAG).dropna() # Ticker名は環境に合わせて調整
        df_move['move_ma'] = df_move['^MOVE'].rolling(52).mean()
        df_move['is_bond_stress'] = df_move['^MOVE'] > df_move['move_ma']

        # Liq_eff

        df = pd.DataFrame(index=target.index)
        df["future_liq_eff"] = target.shift(-LAG).dropna()
        df = df.join(df_net_l["net_liq_mom"], how="left").join(df_vix["is_fear"], how="left").join(df_move["is_bond_stress"], how="left")
        df = df.dropna()
        #df = df.loc["2010-01-01":]
        #print(df)

        # ダイナミック閾値
        df['dynamic_q_low'] = df['future_liq_eff'].rolling(window=winsow, min_periods=52).quantile(quantile_low)
        df['dynamic_q_high'] = df['future_liq_eff'].rolling(window=winsow, min_periods=52).quantile(quantile_high)

        # ラベル
        df["Liq_eff_label"] = 2.0
        stall_condition = (
            (df["future_liq_eff"] <= df["dynamic_q_low"])# &
            #((df['is_fear']) | (df['is_bond_stress']) | (df['net_liq_mom'] < 0))
            #(df['is_fear'])
            #(df['is_bond_stress']) 
            #(df['net_liq_mom'] < 0)
        )
        df.loc[stall_condition, "Liq_eff_label"] = 1.0
        lift_condition = (
            (df["future_liq_eff"] >= df["dynamic_q_high"]) #&
            #((~df['is_fear']) | (~df['is_bond_stress']) | (df['net_liq_mom'] > 0))
            #(~df['is_fear'])
            #(~df['is_bond_stress'])
            #(df['net_liq_mom'] > 0)
        )
        df.loc[lift_condition, "Liq_eff_label"] = 3.0

        #print(df["Liq_eff_label"])

        df_index['next_2m_ret_sp500'] = df_index["^GSPC"].dropna().pct_change(LAG).shift(-LAG).dropna()
        df_index['next_2m_ret_tlt'] = df_index["TLT"].dropna().pct_change(LAG).shift(-LAG).dropna()
        df_index['next_2m_diff_hy'] = df_index["BAMLH0A0HYM2"].dropna().diff(LAG).shift(-LAG).dropna()

        """_analysis_label(
            pd.concat([df["Liq_eff_label"], df_index['next_2m_ret_sp500'], df_index['next_2m_ret_tlt'], df_index['next_2m_diff_hy']], axis=1).dropna()
        )"""

    return df[["Liq_eff_label"]].dropna()

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
            'next_2m_ret_sp500': ['mean', 'std', 'min', 'max', "count", lambda x: (x > 0).mean()],
            'next_2m_ret_tlt': ['mean', 'std', 'min', 'max'],
            'next_2m_diff_hy': ['mean', 'std', 'min', 'max']
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

def return_stats(df, df_oof_ev, LAG):
    assets = df[["^GSPC", "BAMLH0A0HYM2", "TLT"]].dropna(how="all")
    assets['next_2m_ret_sp500'] = assets["^GSPC"].pct_change(8).shift(-LAG)
    assets['next_2m_ret_tlt'] = assets["TLT"].pct_change(8).shift(-LAG)
    assets['next_2m_diff_hy'] = assets["BAMLH0A0HYM2"].diff(8).shift(-LAG)
    combined = pd.concat([df_oof_ev, assets[[
        'next_2m_ret_sp500', "next_2m_ret_tlt",'next_2m_diff_hy']]], axis=1).dropna()

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
            'next_2m_ret_sp500': ['mean', 'std', 'min', 'max', "count", lambda x: (x > 0).mean()],
            'next_2m_ret_tlt': ['mean', 'std', 'min', 'max'],
            'next_2m_diff_hy': ['mean', 'std', 'min', 'max']
            })
        stats.columns = [
            "sp500_mean", "sp500_std", "sp500_min", "sp500_max", "counts", "勝率",
            "tlt_mean", "tlt_std", "tlt_min", "tlt_max",
            "hy_mean", "hy_std", "hy_min", "hy_max"]
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
