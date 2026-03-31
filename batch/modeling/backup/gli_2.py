########################################################
# GLI予測モデル
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

########################################################
# メインプロセス
########################################################
def get_gli_model_beta(df_index):
    # --- データの取得 ---
    df = df_index[gli_index]
    #check_nan_time(df,"1990-01-01")
    #pd.set_option('display.max_rows', None)
    #print(df["CP"].tail(300))

    # --- 目的変数の生成:NDFACBM027SBOGはそのまま。通貨スワップ・ベーシスはSOFRとTEDRATE併用 ---
    #df_target_var = _make_target_variable(df)
    #pd.set_option('display.max_rows', None)
    #print(df.tail(300))

    # --- データ集計-日次は月次に、四半期は月次に線形補完する ---
    #df = _aggregation(df)
    #check_nan_time(df,"1990-01-01")
    #pd.set_option('display.max_rows', None)
    #print(df["CP"].tail(300))

    # --- 教師ラベルの生成 ---
    #df_label = _make_label(df_target_var["NDFACBM027SBOG"].dropna())

    # --- 特徴量抽出 ---
    #df_a, df_b, df_c, df_d =  _featuring(df)
    #check_nan_time(df_a,"1990-01-01")

    # --- 特徴量とGLIのラグ相関分析 ---
    #_lag_corr_check(df_a, df_b, df_c, df_d, df["NDFACBM027SBOG"])

    # --- DFA前にラグを調整する ---
    #df_a, df_b, df_c, df_d = _lag_adjustment(df_a, df_b, df_c, df_d)
    #check_nan_time(df_d,"1990-01-01")

    # --- 特徴量を追加する ---
    #df_features = pd.concat([df_a, df_b, df_c, df_d], axis=1)
    #df_features = _add_features(df_features)
    #check_nan_time(df_features,"1990-01-01")


    df_features = df_features[[
        # --- target_diff52, 逆張りモデル ---
        #'BUSLOANS_yoy_sync',
        #'yoy_PCE_sync',
        #'spd_SOFR_TB3MS_sync',
        #'diff_SOFR_sync',
        #'spd_BBB_A_sync',
        #'yoy_Net_Liquidity_sync_l0',
        #'Res_Ratio_sync',

        # --- target_diff13 ---
        #'level_BUSLOANS_sync',
        #'level_CP_sync',
        #'level_PNFIC1_sync',
        #'qoq_BUSLOANS_sync',
        #'qoq_CP_sync',
        #'qoq_PNFIC1_sync',
        #'yoy_BUSLOANS_sync',
        'yoy_CP_sync',
        #'yoy_PNFIC1_sync',
        #'z52_qoq_BUSLOANS_sync',
        #'z52_qoq_CP_sync',
        #'z52_qoq_PNFIC1_sync',
        #'z52_yoy_BUSLOANS_sync',
        #'z52_yoy_CP_sync',
        #'z52_BUSLOANS_sync',
        #'z52_CP_sync',
        #'z52_PNFIC1_sync',
        #'z104_BUSLOANS_sync',
        #'z104_CP_sync',
        #'z104_PNFIC1_sync',
        'mom4_BUSLOANS_sync',
        #'mom4_CP_sync',
        #'mom4_PNFIC1_sync',

        #'level_PAYEMS_sync',
        #'level_PCE_sync',
        #'qoq_PAYEMS_sync',###
        #'qoq_PCE_sync',###
        #'yoy_PAYEMS_sync',
        #'yoy_PCE_sync',
        #'z52_qoq_PAYEMS_sync',
        #'z52_qoq_PCE_sync',
        #'z52_yoy_PAYEMS_sync',
        #'z52_yoy_PCE_sync',
        #'z52_PAYEMS_sync',
        #'z52_PCE_sync',
        #'z104_PAYEMS_sync',
        #'z104_PCE_sync',
        #'mom4_PAYEMS_sync',
        #'mom4_PCE_sync',

        #'level_SOFR_sync',
        #'level_DXY_sync',
        #'spd_SOFR_TB3MS_sync',
        #'spd_BBB_A_sync',
        #'diff13_SOFR_sync',
        #'qoq_DXY_sync',
        #'diff13_DXY_sync',
        #'diff13_spd_SOFR_TB3MS_sync',
        #'diff13_spd_BBB_A_sync',
        #'diff52_SOFR_sync',
        #'yoy_DXY_sync',
        #'diff52_spd_SOFR_TB3MS_sync',
        #'diff52_spd_BBB_A_sync',
        #'z52_diff13_SOFR_sync',
        #'z52_qoq_DXY_sync',
        #'z52_diff13_spd_SOFR_TB3MS_sync',
        #'z52_diff13_spd_BBB_A_sync',
        #'z52_diff52_SOFR_sync',
        #'z52_yoy_DXY_sync',
        #'z52_diff52_spd_SOFR_TB3MS_sync',
        #'z52_diff52_spd_BBB_A_sync',
        'z52_SOFR_sync',
        #'z52_DXY_sync',
        #'z52_spd_SOFR_TB3MS_sync',
        #'z52_spd_BBB_A_sync',
        #'z104_SOFR_sync',
        #'z104_DXY_sync',
        #'z104_spd_SOFR_TB3MS_sync',
        #'z104_spd_BBB_A_sync',
        #'mom4_SOFR_sync',
        #'mom4_DXY_sync',
        #'mom4_spd_SOFR_TB3MS_sync',
        #'mom4_spd_BBB_A_sync',

        #'Net_Liquidity_sync',
        #'Res_Ratio_sync',
        #'Abs_Rate_sync',
        'qoq_Net_Liquidity_sync',
        ##'qoq_Res_Ratio_sync',
        #'qoq_Abs_Rate_sync',
        #'yoy_Net_Liquidity_sync',
        #'yoy_Res_Ratio_sync',
        'yoy_Abs_Rate_sync',
        #'z52_qoq_Net_Liquidity_sync',
        #'z52_qoq_Res_Ratio_sync',
        #'z52_qoq_Abs_Rate_sync',
        #'z52_yoy_Net_Liquidity_sync',
        #'z52_yoy_Res_Ratio_sync',
        #'z52_yoy_Abs_Rate_sync',
        #'z52_Net_Liquidity_sync',
        #'z52_Res_Ratio_sync',
        #'z52_Abs_Rate_sync',
        #'z104_Net_Liquidity_sync',
        'z104_Res_Ratio_sync',
        #'z104_Abs_Rate_sync',
        #'mom4_Net_Liquidity_sync',
        'mom4_Res_Ratio_sync',
        #'mom4_Abs_Rate_sync'
    ]]

    # --- 学習（1か月予測と3か月予測でgap設定をかえる） ---
    #df_master = df_label.to_frame().join(df_features, how='left')
    #df_master = df_master.loc["2007-01-01":]
    #print(df_master)
    #check_nan_time(df_master,"1990-01-01")


    #print(f"特徴量のリスト: {df_features.columns}")

    # --- diff52基準の逆張りモデル ---
    """df_oof_all, final_shap_dfs, df_oof_ev = learning_lgbm_test_gli(
        df_master, target_col="gli_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=2, gap=13,
        n_estimators=1000,learning_rate=0.001, num_leaves=15, min_data_in_leaf=65,
        reg_alpha=0.5, reg_lambda=7, max_depth=3,#feature_fraction=0.6,bagging_fraction=0.5,bagging_freq=1,
        class_weight="balanced",
        importance_type="gain",stopping_rounds=30,#min_gain_to_split=0.1,
        learning_curve=True,
    )"""


    """df_oof_all, final_shap_dfs, df_oof_ev = learning_lgbm_test_gli(
        df_master, target_col="gli_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=2, gap=13,
        n_estimators=1000,learning_rate=0.001, num_leaves=15, min_data_in_leaf=50,
        reg_alpha=0.5, reg_lambda=7, max_depth=2,#feature_fraction=0.6,bagging_fraction=0.5,bagging_freq=1,
        class_weight="balanced",
        importance_type="gain",stopping_rounds=30,#min_gain_to_split=0.1,
        learning_curve=True,
    )"""
    """for label, shap_df in final_shap_dfs.items():
        print(f"\n=== レジーム: {label} の符号検証 ===")
        # 検証データ期間の元の特徴量を取得
        original_X = df_master.loc[shap_df.index, df_features.columns]

        logic_results = []
        for col in df_features.columns:
            # 元の値とSHAP値の相関を計算
            correlation = original_X[col].corr(shap_df[col])

            # 方向性の判定
            direction = "正の相関 (+)" if correlation > 0 else "負の相関 (-)"
            logic_results.append({
                "特徴量": col,
                "方向性": direction,
                "相関係数": f"{correlation:.3f}"
            })

        print(pd.DataFrame(logic_results))"""

    """mean_coefs, all_y_probs, all_y_test = learning_logistic_lasso_test(
        df_master, target_col="gli_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=2, gap=13,solver='saga',max_iter=5000,
        C=0.5, penalty="l1",class_weight="balanced",
    )"""

    # --- 学習結果の分析・可視化 ---
    #plot_gli_trajectory(df_trajectory, df_index["gli"].ffill(),df_index["^GSPC"], start_date="2010-01-01")

    #return df_oof_all

def _make_target_variable(df):
    # スプレッド
    df['SOFR_Spread'] = df['SOFR'].ffill() - df['DTB3'].ffill()
    df['TED_Spread'] = df['TEDRATE'].ffill()
    # つなぎ目
    overlap_start, overlap_end = "2018-04-01", "2018-12-31"
    offset = (df.loc[overlap_start:overlap_end, 'TED_Spread'] - 
              df.loc[overlap_start:overlap_end, 'SOFR_Spread']).mean()
    #print(f"Detected Level Shift (Offset): {offset:.4f}")
    switch_date = "2018-04-01"
    df['Final_Price_Stress'] = df['SOFR_Spread'] # 基本はSOFRベース

    df.loc[df.index < switch_date, 'Final_Price_Stress'] = df.loc[df.index < switch_date, 'TED_Spread'] - offset

    return df[["NDFACBM027SBOG","Final_Price_Stress"]]

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
    #print(df_monthly_w_lagged.tail(10))
    #check_nan_time(df_monthly_w_lagged)

    # 結合
    df_combine = pd.concat([df_daily_w_s, df_weekly_w_s, df_monthly_w_lagged, df_quarterly_w], axis=1)

    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_combine.dropna(how="all").tail(20))
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

def _lag_corr_check(df_a, df_b, df_c, df_d, target):

    
    # GLI をdiffにする
    #target = target.resample('ME').interpolate(method='linear').dropna()
    #df_gli_yoy = target.pct_change(4).dropna().rename("gli_yoy")
    target_diff = target.diff(13).dropna().rename("NDFACBM027SBOG_diff")
    #target_diff = target_diff.loc["2008-01-01":]
    target_diff = target_diff.loc["2010-01-01":]

    # GLI のインデックスに合わせる
    #df_a = standard_scalar_df(df_a)
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
    #check_nan_time(df_b_all,"1990-01-01")
    #plot_index(df_a_all)
    #print(df_lag_a.columns)

    #_plot_lag_correlation(df_lag_a)
    #_plot_lag_correlation(df_lag_b)
    #_plot_lag_correlation(df_lag_c)
    #_plot_lag_correlation(df_lag_d)

    #pd.set_option('display.max_rows', None)
    #print(df_lag_d)

def _lag_adjustment(df_a, df_b, df_c, df_d):

    # Layer A
    #print(df_a.columns)
    df_a_ = pd.DataFrame(index=df_a.index)
    df_a_['level_BUSLOANS_sync'] = df_a['level_BUSLOANS'].shift(0)
    df_a_['level_CP_sync'] = df_a['level_CP'] .shift(16)
    df_a_['level_PNFIC1_sync'] = df_a['level_PNFIC1'].shift(5)
    df_a_['qoq_BUSLOANS_sync'] = df_a['qoq_BUSLOANS'].shift(0)
    df_a_['qoq_CP_sync'] = df_a['qoq_CP'].shift(25)
    df_a_['qoq_PNFIC1_sync'] = df_a['qoq_PNFIC1'].shift(32)
    df_a_['yoy_BUSLOANS_sync'] = df_a['yoy_BUSLOANS'].shift(0)
    df_a_['yoy_CP_sync'] = df_a['yoy_CP'].shift(61)
    df_a_['yoy_PNFIC1_sync'] = df_a['yoy_PNFIC1'].shift(10)
    df_a_['z52_qoq_BUSLOANS_sync'] = df_a['z52_qoq_BUSLOANS'].shift(2)
    df_a_['z52_qoq_CP_sync'] = df_a['z52_qoq_CP'].shift(15)
    df_a_['z52_qoq_PNFIC1_sync'] = df_a['z52_qoq_PNFIC1'].shift(39)
    df_a_['z52_yoy_BUSLOANS_sync'] = df_a['z52_yoy_BUSLOANS'].shift(0)
    df_a_['z52_yoy_CP_sync'] = df_a['z52_yoy_CP'].shift(69)
    #df_a_['z52_yoy_PNFIC1_sync'].shift()
    df_a_['z52_BUSLOANS_sync'] = df_a['z52_BUSLOANS'].shift(1)
    df_a_['z52_CP_sync'] = df_a['z52_CP'].shift(21)
    df_a_['z52_PNFIC1_sync'] = df_a['z52_PNFIC1'].shift(17)
    df_a_['z104_BUSLOANS_sync'] = df_a['z104_BUSLOANS'].shift(0)
    df_a_['z104_CP_sync'] = df_a['z104_CP'].shift(18)
    df_a_['z104_PNFIC1_sync'] = df_a['z104_PNFIC1'].shift(10)
    df_a_['mom4_BUSLOANS_sync'] = df_a['mom4_BUSLOANS'].shift(0)
    df_a_['mom4_CP_sync'] = df_a['mom4_CP'].shift(26)
    df_a_['mom4_PNFIC1_sync'] = df_a['mom4_PNFIC1'].shift(14)

    # Layer B
    #print(df_b.columns)
    df_b_ = pd.DataFrame(index=df_b.index)
    df_b_['level_PAYEMS_sync'] = df_b['level_PAYEMS'].shift(13)
    df_b_['level_PCE_sync'] = df_b['level_PCE'].shift(15)
    df_b_['qoq_PAYEMS_sync'] = df_b['qoq_PAYEMS'].shift(13)
    df_b_['qoq_PCE_sync'] = df_b['qoq_PCE'].shift(15)
    df_b_['yoy_PAYEMS_sync'] = df_b['yoy_PAYEMS'].shift(13)
    df_b_['yoy_PCE_sync'] = df_b['yoy_PCE'].shift(15)
    df_b_['z52_qoq_PAYEMS_sync'] = df_b['z52_qoq_PAYEMS'].shift(17)
    df_b_['z52_qoq_PCE_sync'] = df_b['z52_qoq_PCE'].shift(19)
    df_b_['z52_yoy_PAYEMS_sync'] = df_b['z52_yoy_PAYEMS'].shift(16)
    df_b_['z52_yoy_PCE_sync'] = df_b['z52_yoy_PCE'].shift(18)
    df_b_['z52_PAYEMS_sync'] = df_b['z52_PAYEMS'].shift(15)
    df_b_['z52_PCE_sync'] = df_b['z52_PCE'].shift(17)
    df_b_['z104_PAYEMS_sync'] = df_b['z104_PAYEMS'].shift(14)
    df_b_['z104_PCE_sync'] = df_b['z104_PCE'].shift(16)
    df_b_['mom4_PAYEMS_sync'] = df_b['mom4_PAYEMS'].shift(16)
    df_b_['mom4_PCE_sync'] = df_b['mom4_PCE'].shift(18)

    # Layer C
    #print(df_c.columns)
    df_c_ = pd.DataFrame(index=df_c.index)
    df_c_['level_SOFR_sync'] = df_c["level_SOFR"].shift(2)
    df_c_['level_DXY_sync'] = df_c["level_DXY"].shift(27)
    df_c_['spd_SOFR_TB3MS_sync'] = df_c["spd_SOFR_TB3MS"].shift(0)
    df_c_['spd_BBB_A_sync'] = df_c["spd_BBB_A"].shift(16)

    df_c_['diff13_SOFR_sync'] = df_c["diff13_SOFR"].shift(1)
    df_c_['qoq_DXY_sync'] = df_c["qoq_DXY"].shift(20)
    df_c_['diff13_DXY_sync'] = df_c["diff13_DXY"].shift()
    df_c_['diff13_spd_SOFR_TB3MS_sync'] = df_c["diff13_spd_SOFR_TB3MS"].shift(20)
    df_c_['diff13_spd_BBB_A_sync'] = df_c["diff13_spd_BBB_A"].shift(16)

    df_c_['diff52_SOFR_sync'] = df_c["diff52_SOFR"].shift(0)
    df_c_['yoy_DXY_sync'] = df_c["yoy_DXY"].shift(20)
    #df_c_['diff52_DXY_sync'] = df_c["diff52_DXY"].shift()
    df_c_['diff52_spd_SOFR_TB3MS_sync'] = df_c["diff52_spd_SOFR_TB3MS"].shift(0)
    df_c_['diff52_spd_BBB_A_sync'] = df_c["diff52_spd_BBB_A"].shift(16)

    df_c_['z52_diff13_SOFR_sync'] = df_c["z52_diff13_SOFR"].shift(2)
    df_c_['z52_qoq_DXY_sync'] = df_c["z52_qoq_DXY"].shift(20)
    #df_c_['z52_diff13_DXY_sync'] = df_c["z52_diff13_DXY"].shift()
    df_c_['z52_diff13_spd_SOFR_TB3MS_sync'] = df_c["z52_diff13_spd_SOFR_TB3MS"].shift(21)
    df_c_['z52_diff13_spd_BBB_A_sync'] = df_c["z52_diff13_spd_BBB_A"].shift(18)

    df_c_['z52_diff52_SOFR_sync'] = df_c["z52_diff52_SOFR"].shift(2)
    df_c_['z52_yoy_DXY_sync'] = df_c["z52_yoy_DXY"].shift(18)
    #df_c_['z52_diff52_DXY_sync'] = df_c["z52_diff52_DXY"].shift()
    df_c_['z52_diff52_spd_SOFR_TB3MS_sync'] = df_c["z52_diff52_spd_SOFR_TB3MS"].shift(18)
    df_c_['z52_diff52_spd_BBB_A_sync'] = df_c["z52_diff52_spd_BBB_A"].shift(17)

    df_c_['z52_SOFR_sync'] = df_c["z52_SOFR"].shift(0)
    df_c_['z52_DXY_sync'] = df_c["z52_DXY"].shift(20)
    df_c_['z52_spd_SOFR_TB3MS_sync'] = df_c["z52_spd_SOFR_TB3MS"].shift(0)
    df_c_['z52_spd_BBB_A_sync'] = df_c["z52_spd_BBB_A"].shift(18)

    df_c_['z104_SOFR_sync'] = df_c["z104_SOFR"].shift(0)
    df_c_['z104_DXY_sync'] = df_c["z104_DXY"].shift(20)
    df_c_['z104_spd_SOFR_TB3MS_sync'] = df_c["z104_spd_SOFR_TB3MS"].shift(0)
    df_c_['z104_spd_BBB_A_sync'] = df_c["z104_spd_BBB_A"].shift(17)

    df_c_['mom4_SOFR_sync'] = df_c["mom4_SOFR"].shift(5)
    df_c_['mom4_DXY_sync'] = df_c["mom4_DXY"].shift(0)
    df_c_['mom4_spd_SOFR_TB3MS_sync'] = df_c["mom4_spd_SOFR_TB3MS"].shift(25)
    df_c_['mom4_spd_BBB_A_sync'] = df_c["mom4_spd_BBB_A"].shift(19)

    # Layer D
    #print(df_d.columns)
    df_d_ = pd.DataFrame(index=df_d.index)
    df_d_['Net_Liquidity_sync'] = df_d["Net_Liquidity"].shift(0)
    df_d_['Res_Ratio_sync'] = df_d["Res_Ratio"].shift(0)
    df_d_['Abs_Rate_sync'] = df_d["Abs_Rate"].shift(4)

    df_d_['qoq_Net_Liquidity_sync'] = df_d["qoq_Net_Liquidity"].shift(0)
    df_d_['qoq_Res_Ratio_sync'] = df_d["qoq_Res_Ratio"].shift(1)
    df_d_['qoq_Abs_Rate_sync'] = df_d["qoq_Abs_Rate"].shift(42)

    df_d_['yoy_Net_Liquidity_sync'] = df_d["yoy_Net_Liquidity"].shift(0)
    df_d_['yoy_Res_Ratio_sync'] = df_d["yoy_Res_Ratio"].shift(0)
    df_d_['yoy_Abs_Rate_sync'] = df_d["yoy_Abs_Rate"].shift(9)

    df_d_['z52_qoq_Net_Liquidity_sync'] = df_d["z52_qoq_Net_Liquidity"].shift(1)
    df_d_['z52_qoq_Res_Ratio_sync'] = df_d["z52_qoq_Res_Ratio"].shift(4)
    df_d_['z52_qoq_Abs_Rate_sync'] = df_d["z52_qoq_Abs_Rate"].shift(42)

    df_d_['z52_yoy_Net_Liquidity_sync'] = df_d["z52_yoy_Net_Liquidity"].shift(0)
    df_d_['z52_yoy_Res_Ratio_sync'] = df_d["z52_yoy_Res_Ratio"].shift(2)
    df_d_['z52_yoy_Abs_Rate_sync'] = df_d["z52_yoy_Abs_Rate"].shift(40)

    df_d_['z52_Net_Liquidity_sync'] = df_d["z52_Net_Liquidity"].shift(1)
    df_d_['z52_Res_Ratio_sync'] = df_d["z52_Res_Ratio"].shift(1)
    df_d_['z52_Abs_Rate_sync'] = df_d["z52_Abs_Rate"].shift(40)

    df_d_['z104_Net_Liquidity_sync'] = df_d["z104_Net_Liquidity"].shift(0)
    df_d_['z104_Res_Ratio_sync'] = df_d["z104_Res_Ratio"].shift(0)
    df_d_['z104_Abs_Rate_sync'] = df_d["z104_Abs_Rate"].shift(37)

    df_d_['mom4_Net_Liquidity_sync'] = df_d["mom4_Net_Liquidity"].shift(3)
    df_d_['mom4_Res_Ratio_sync'] = df_d["mom4_Res_Ratio"].shift(5)
    df_d_['mom4_Abs_Rate_sync'] = df_d["mom4_Abs_Rate"].shift(7)

    start = df_a_.apply(pd.Series.first_valid_index).max()
    df_a_ = df_a_[start:]
    start = df_b_.apply(pd.Series.first_valid_index).max()
    df_b_ = df_b_[start:]
    start = df_c_.apply(pd.Series.first_valid_index).max()
    df_c_ = df_c_[start:]
    start = df_d_.apply(pd.Series.first_valid_index).max()
    df_d_ = df_d_[start:]

    #print(df_a.head(10))
    #print(df_b.head(10))
    #print(df_c.head(10))
    #print(df_d.head(10))

    return df_a_, df_b_, df_c_, df_d_

def _check_factor(df, gli, sp500):
    # GLIとSP500を因子のインデックスに合わせる
    df_gli = gli.resample('ME').interpolate(method='linear').reindex(df.index, method="ffill")
    df_gli_diff = df_gli.diff(12).dropna().rename("GLI_diff")

    df_sp500 = sp500.resample('ME').mean().dropna()
    #sp500_ret_cumsum = df_sp500.pct_change().cumsum().dropna()
    #sp500_ret_cumsum = sp500_ret_cumsum.reindex(df.index, method="ffill").rename("sp500_ret_cumsum")

    #df = df.cumsum().dropna()

    plot_df = pd.concat([df, df_gli_diff], axis=1).dropna()
    
    # Zスコア化
    scaled_cols = standard_scalar_df(plot_df[['Factor_Supply', 'GLI_diff']])
    plot_df['Factor_Z'] = scaled_cols['Factor_Supply']
    plot_df['GLI_diff_Z'] = scaled_cols['GLI_diff']
    plot_df = plot_df.drop(columns=['Factor_Supply', 'GLI_diff'])

    # グラフ描画
    plot_index(plot_df[["Factor_Z", "GLI_diff_Z"]])

########################################################
# 特徴量抽出
########################################################
def _featuring(df):

    df_a = _featuring_a(df)
    df_b = _featuring_b(df)
    df_c = _featuring_c(df)
    df_d = _featuring_d(df)

    return df_a, df_b, df_c, df_d

def _featuring_a(df):
    col = ["BUSLOANS","CP","PNFIC1"]
    df_ = df[col].dropna(how="all")

    # level
    level_BUSLOANS = df_["BUSLOANS"].dropna().rename("level_BUSLOANS")
    level_CP = df_["CP"].dropna().rename("level_CP")
    level_PNFIC1 = df_["PNFIC1"].dropna().rename("level_PNFIC1")

    # yoy / diff
    qoq_BUSLOANS = df_["BUSLOANS"].pct_change(13).dropna().rename("qoq_BUSLOANS")
    qoq_CP = df_["CP"].pct_change(13).dropna().rename("qoq_CP")
    qoq_PNFIC1 = df_["PNFIC1"].pct_change(13).dropna().rename("qoq_PNFIC1")
    yoy_BUSLOANS = df_["BUSLOANS"].pct_change(52).dropna().rename("yoy_BUSLOANS")
    yoy_CP = df_["CP"].pct_change(52).dropna().rename("yoy_CP")
    yoy_PNFIC1 = df_["PNFIC1"].pct_change(52).dropna().rename("yoy_PNFIC1")

    # yoy/diffのZスコア化
    z52_qoq_BUSLOANS = _featuring_z_score(qoq_BUSLOANS, 52).rename("z52_qoq_BUSLOANS")
    z52_qoq_CP = _featuring_z_score(qoq_CP, 52).rename("z52_qoq_CP")
    z52_qoq_PNFIC1 = _featuring_z_score(qoq_PNFIC1, 52).rename("z52_qoq_PNFIC1")
    z52_yoy_BUSLOANS = _featuring_z_score(yoy_BUSLOANS, 52).rename("z52_yoy_BUSLOANS")
    z52_yoy_CP = _featuring_z_score(yoy_CP, 52).rename("z52_yoy_CP")
    z52_yoy_PNFIC1 = _featuring_z_score(yoy_PNFIC1, 52).rename("z52_yoy_PNFIC1")

    # 生値のZスコア化
    z52_BUSLOANS = _featuring_z_score(df_["BUSLOANS"], 52).rename("z52_BUSLOANS")
    z52_CP = _featuring_z_score(df_["CP"], 52).rename("z52_CP")
    z52_PNFIC1 = _featuring_z_score(df_["PNFIC1"], 52).rename("z52_PNFIC1")

    z104_BUSLOANS = _featuring_z_score(df_["BUSLOANS"], 104).rename("z104_BUSLOANS")
    z104_CP = _featuring_z_score(df_["CP"], 104).rename("z104_CP")
    z104_PNFIC1 = _featuring_z_score(df_["PNFIC1"], 104).rename("z104_PNFIC1")

    # mom
    mom4_BUSLOANS = df_["BUSLOANS"].diff(13).rename("mom4_BUSLOANS")
    mom4_CP = df_["CP"].diff(4).rename("mom4_CP")
    mom4_PNFIC1 = df_["PNFIC1"].diff(4).rename("mom4_PNFIC1")

    df_feature = pd.concat([
        level_BUSLOANS,
        level_CP,
        level_PNFIC1,
        qoq_BUSLOANS,
        qoq_CP,
        qoq_PNFIC1,
        yoy_BUSLOANS,
        yoy_CP,
        yoy_PNFIC1,
        z52_qoq_BUSLOANS,
        z52_qoq_CP,
        z52_qoq_PNFIC1,
        z52_yoy_BUSLOANS,
        z52_yoy_CP,
        z52_yoy_PNFIC1,
        z52_BUSLOANS,
        z52_CP,
        z52_PNFIC1,
        z104_BUSLOANS,
        z104_CP,
        z104_PNFIC1,
        mom4_BUSLOANS,
        mom4_CP,
        mom4_PNFIC1
        ], axis=1).dropna(how="all")
    #print(df_feature.head(10))

    return df_feature

def _featuring_b(df):
    col = ["PAYEMS","PCE","CES0500000003"]
    df_ = df[col].dropna(how="all")

    # level
    level_PAYEMS = df_["PAYEMS"].dropna().rename("level_PAYEMS")
    level_PCE = df_["PCE"].dropna().rename("level_PCE")

    # yoy / diff
    qoq_PAYEMS = df_["PAYEMS"].pct_change(13).dropna().rename("qoq_PAYEMS")
    yoy_PAYEMS = df_["PAYEMS"].pct_change(52).dropna().rename("yoy_PAYEMS")
    qoq_PCE = df_["PCE"].pct_change(13).dropna().rename("qoq_PCE")
    yoy_PCE = df_["PCE"].pct_change(52).dropna().rename("yoy_PCE")

    # yoy/diffのZスコア化
    z52_qoq_PAYEMS = _featuring_z_score(qoq_PAYEMS, 52).rename("z52_qoq_PAYEMS")
    z52_yoy_PAYEMS = _featuring_z_score(yoy_PAYEMS, 52).rename("z52_yoy_PAYEMS")
    z52_qoq_PCE = _featuring_z_score(qoq_PCE, 52).rename("z52_qoq_PCE")
    z52_yoy_PCE = _featuring_z_score(yoy_PCE, 52).rename("z52_yoy_PCE")

    # 生値のZスコア化
    z52_PAYEMS = _featuring_z_score(df_["PAYEMS"], 52).rename("z52_PAYEMS")
    z52_PCE = _featuring_z_score(df_["PCE"], 52).rename("z52_PCE")
    z104_PAYEMS = _featuring_z_score(df_["PAYEMS"], 104).rename("z104_PAYEMS")
    z104_PCE = _featuring_z_score(df_["PCE"], 104).rename("z104_PCE")

    # mom
    mom4_PAYEMS = df_["PAYEMS"].diff(4).rename("mom4_PAYEMS")
    mom4_PCE = df_["PCE"].diff(4).rename("mom4_PCE")

    df_featured = pd.concat([
        level_PAYEMS,
        level_PCE,
        qoq_PAYEMS,
        qoq_PCE,
        yoy_PAYEMS,
        yoy_PCE,
        z52_qoq_PAYEMS,
        z52_qoq_PCE,
        z52_yoy_PAYEMS,
        z52_yoy_PCE,
        z52_PAYEMS,
        z52_PCE,
        z104_PAYEMS,
        z104_PCE,
        mom4_PAYEMS,
        mom4_PCE
        ], axis=1).dropna(how="all")
    #print(df_featured.corr())
    #check_nan_time(df_featured,"1990-01-01")

    #plot_index(df_featured.tail(10))
    return df_featured

def _featuring_c(df):
    col = ["SOFR","DFF","TB3MS","DX-Y.NYB","BAMLC0A4CBBB","BAMLC0A3CA"]
    df_ = df[col].dropna(how="all")

    # level
    df_["SOFR"] = df_["SOFR"].fillna(df_["DFF"])
    level_SOFR = df_["SOFR"].dropna().rename("level_SOFR")
    level_DXY = df_["DX-Y.NYB"].dropna().rename("level_DXY")
    spd_SOFR_TB3MS = (df_["SOFR"] - df_["TB3MS"]).dropna().rename("spd_SOFR_TB3MS")
    spd_BBB_A = (df_["BAMLC0A4CBBB"] - df_["BAMLC0A3CA"]).dropna().rename("spd_BBB_A")

    # yoy / diff
    diff13_SOFR = df_["SOFR"].diff(13).dropna().rename("diff13_SOFR")
    qoq_DXY = df_["DX-Y.NYB"].pct_change(13).dropna().rename("qoq_DXY")
    diff13_DXY = df_["DX-Y.NYB"].diff(13).dropna().rename("diff13_DXY")
    diff13_spd_SOFR_TB3MS = spd_SOFR_TB3MS.diff(13).dropna().rename("diff13_spd_SOFR_TB3MS")
    diff13_spd_BBB_A = spd_BBB_A.diff(13).dropna().rename("diff13_spd_BBB_A")
    diff52_SOFR = df_["SOFR"].diff(52).dropna().rename("diff52_SOFR")
    yoy_DXY = df_["DX-Y.NYB"].pct_change(52).dropna().rename("yoy_DXY")
    diff52_DXY = df_["DX-Y.NYB"].diff(52).dropna().rename("diff52_DXY")
    diff52_spd_SOFR_TB3MS = spd_SOFR_TB3MS.diff(52).dropna().rename("diff52_spd_SOFR_TB3MS")
    diff52_spd_BBB_A = spd_BBB_A.diff(52).dropna().rename("diff52_spd_BBB_A")

    # yoy/diffのZスコア化
    z52_diff13_SOFR = _featuring_z_score(diff13_SOFR, 52).rename("z52_diff13_SOFR")
    z52_qoq_DXY = _featuring_z_score(qoq_DXY, 52).rename("z52_qoq_DXY")
    z52_diff13_DXY = _featuring_z_score(diff13_DXY, 52).rename("z52_diff13_DXY")
    z52_diff13_spd_SOFR_TB3MS = _featuring_z_score(diff13_spd_SOFR_TB3MS, 52).rename("z52_diff13_spd_SOFR_TB3MS")
    z52_diff13_spd_BBB_A = _featuring_z_score(diff13_spd_BBB_A, 52).rename("z52_diff13_spd_BBB_A")
    z52_diff52_SOFR = _featuring_z_score(diff52_SOFR, 52).rename("z52_diff52_SOFR")
    z52_yoy_DXY = _featuring_z_score(yoy_DXY, 52).rename("z52_yoy_DXY")
    z52_diff52_DXY = _featuring_z_score(diff52_DXY, 52).rename("z52_diff52_DXY")
    z52_diff52_spd_SOFR_TB3MS = _featuring_z_score(diff52_spd_SOFR_TB3MS, 52).rename("z52_diff52_spd_SOFR_TB3MS")
    z52_diff52_spd_BBB_A = _featuring_z_score(diff52_spd_BBB_A, 52).rename("z52_diff52_spd_BBB_A")
    
    # 生値のZスコア化
    z52_SOFR = _featuring_z_score(df_["SOFR"], 52).rename("z52_SOFR")
    z52_DXY = _featuring_z_score(df_["DX-Y.NYB"], 52).rename("z52_DXY")
    z52_SOFR_TB3MS = _featuring_z_score(spd_SOFR_TB3MS, 52).rename("z52_spd_SOFR_TB3MS")
    z52_BBB_A = _featuring_z_score(spd_BBB_A, 52).rename("z52_spd_BBB_A")
    z104_SOFR = _featuring_z_score(df_["SOFR"], 104).rename("z104_SOFR")
    z104_DXY = _featuring_z_score(df_["DX-Y.NYB"], 104).rename("z104_DXY")
    z104_SOFR_TB3MS = _featuring_z_score(spd_SOFR_TB3MS, 104).rename("z104_spd_SOFR_TB3MS")
    z104_BBB_A = _featuring_z_score(spd_BBB_A, 104).rename("z104_spd_BBB_A")
    
    # mom
    mom4_SOFR = df_["SOFR"].diff(4).rename("mom4_SOFR")
    mom4_DXY = df_["DX-Y.NYB"].diff(4).rename("mom4_DXY")
    mom4_spd_SOFR_TB3MS = spd_SOFR_TB3MS.diff(4).rename("mom4_spd_SOFR_TB3MS")
    mom4_spd_BBB_A = spd_BBB_A.diff(4).rename("mom4_spd_BBB_A")

    df_featured = pd.concat([
        level_SOFR, level_DXY, spd_SOFR_TB3MS, spd_BBB_A,
        diff13_SOFR, qoq_DXY, diff13_DXY, diff13_spd_SOFR_TB3MS, diff13_spd_BBB_A,
        diff52_SOFR, yoy_DXY, diff52_DXY, diff52_spd_SOFR_TB3MS, diff52_spd_BBB_A,
        z52_diff13_SOFR, z52_qoq_DXY, z52_diff13_DXY, z52_diff13_spd_SOFR_TB3MS, z52_diff13_spd_BBB_A,
        z52_diff52_SOFR, z52_yoy_DXY, z52_diff52_DXY, z52_diff52_spd_SOFR_TB3MS, z52_diff52_spd_BBB_A,
        z52_SOFR, z52_DXY, z52_SOFR_TB3MS, z52_BBB_A,
        z104_SOFR, z104_DXY, z104_SOFR_TB3MS, z104_BBB_A,
        mom4_SOFR, mom4_DXY, mom4_spd_SOFR_TB3MS, mom4_spd_BBB_A
        ], axis=1).dropna(how="all")

    #print(df_featured.head(29))
    #plot_index(df_featured)
    #print(df_featured.corr())
    #check_nan_time(df_featured,"1990-01-01")
    return df_featured

def _featuring_d(df):
    col = ["RRPONTSYD","WALCL","RESBALNS","TOTRESNS","WDTGAL"]
    df_ = df[col].dropna(how="all")

    #　2020/8/31までは RESBALNS, それ以降は TOTRESNS で埋める
    df_['Unified_Reserves'] = df_['RESBALNS'].fillna(df_['TOTRESNS'])
    df_['Unified_Reserves'] = df_['Unified_Reserves'] * 1000
    # リバースレポ残高は2013/8/31から本格的に運用。空白を 0 で埋める (2013年以前対策)
    df_['RRP_filled'] = df_['RRPONTSYD'].fillna(0) * 1000
    df_ = df_.drop(columns=['RESBALNS', 'TOTRESNS', 'RRPONTSYD'])

    # Net Liquidity / 銀行準備金の厚み / 吸収率 (TGA+RRPが資産に占める割合)
    Net_Liquidity = (df_['WALCL'] - (df_['WDTGAL'] +  df_['RRP_filled'])).rename("Net_Liquidity")
    Res_Ratio = (df_['Unified_Reserves'] / df_['WALCL']).rename("Res_Ratio")
    Abs_Rate = ((df_['WDTGAL'] + df_['RRP_filled']) / df_['WALCL']).rename("Abs_Rate")

    # 特徴量1: 成長率 (YoY)
    qoq_Net_Liquidity = Net_Liquidity.pct_change(13).rename("qoq_Net_Liquidity")
    qoq_Res_Ratio = Res_Ratio.pct_change(13).rename("qoq_Res_Ratio")
    qoq_Abs_Rate = Abs_Rate.pct_change(13).rename("qoq_Abs_Rate")

    yoy_Net_Liquidity = Net_Liquidity.pct_change(52).rename("yoy_Net_Liquidity")
    yoy_Res_Ratio = Res_Ratio.pct_change(52).rename("yoy_Res_Ratio")
    yoy_Abs_Rate = Abs_Rate.pct_change(52).rename("yoy_Abs_Rate")

    # 特徴量2: YOYのZスコア
    z52_qoq_Net_Liquidity = _featuring_z_score(qoq_Net_Liquidity, 52).rename("z52_qoq_Net_Liquidity")
    z52_qoq_Res_Ratio = _featuring_z_score(qoq_Res_Ratio, 52).rename("z52_qoq_Res_Ratio")
    z52_qoq_Abs_Rate = _featuring_z_score(qoq_Abs_Rate, 52).rename("z52_qoq_Abs_Rate")

    z52_yoy_Net_Liquidity = _featuring_z_score(yoy_Net_Liquidity, 52).rename("z52_yoy_Net_Liquidity")
    z52_yoy_Res_Ratio = _featuring_z_score(yoy_Res_Ratio, 52).rename("z52_yoy_Res_Ratio")
    z52_yoy_Abs_Rate = _featuring_z_score(yoy_Abs_Rate, 52).rename("z52_yoy_Abs_Rate")

    # 特徴量3: 生値のZスコア
    z52_Net_Liquidity = _featuring_z_score(Net_Liquidity, 52).rename("z52_Net_Liquidity")
    z52_Res_Ratio = _featuring_z_score(Res_Ratio, 52).rename("z52_Res_Ratio")
    z52_Abs_Rate = _featuring_z_score(Abs_Rate, 52).rename("z52_Abs_Rate")
    z104_Net_Liquidity = _featuring_z_score(Net_Liquidity, 104).rename("z104_Net_Liquidity")
    z104_Res_Ratio = _featuring_z_score(Res_Ratio, 104).rename("z104_Res_Ratio")
    z104_Abs_Rate = _featuring_z_score(Abs_Rate, 104).rename("z104_Abs_Rate")

    # mom
    mom4_Net_Liquidity = Net_Liquidity.diff(4).rename("mom4_Net_Liquidity")
    mom4_Res_Ratio = Res_Ratio.diff(4).rename("mom4_Res_Ratio")
    mom4_Abs_Rate = Abs_Rate.diff(4).rename("mom4_Abs_Rate")

    df_featured = pd.concat([
        Net_Liquidity, Res_Ratio, Abs_Rate,
        qoq_Net_Liquidity, qoq_Res_Ratio, qoq_Abs_Rate,
        yoy_Net_Liquidity, yoy_Res_Ratio, yoy_Abs_Rate,
        z52_qoq_Net_Liquidity, z52_qoq_Res_Ratio, z52_qoq_Abs_Rate,
        z52_yoy_Net_Liquidity, z52_yoy_Res_Ratio, z52_yoy_Abs_Rate,
        z52_Net_Liquidity, z52_Res_Ratio, z52_Abs_Rate,
        z104_Net_Liquidity, z104_Res_Ratio, z104_Abs_Rate,
        mom4_Net_Liquidity, mom4_Res_Ratio, mom4_Abs_Rate
    ], axis=1).dropna(how="all")

    #pd.set_option('display.max_rows', None)
    #print(df_featured.tail(10))
    #check_nan_time(df_featured,"1990-01-01")
    #plot_index(df_featured)
    #print(df_featured.corr())
    return df_featured

def _add_features(df):

    #df["Financial_Stress_Index"] = df["diff_SOFR_sync"]*df["z52_yoy_Net_Liquidity_sync"]
    #df["vol4_CP_yoy_sync"] = df["CP_yoy_sync"].rolling(window=4).std()
    #df["vol4_yoy_PCE_sync"] = df["yoy_PCE_sync"].rolling(window=4).std()

    return df.dropna(how="all")

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)
########################################################
# 回帰の変数
########################################################

def _make_reg_y(df_gli):
    #print(df_gli)
    gli_monthly = df_gli.resample('ME').first().interpolate(method='linear')
    #print(gli_monthly)
    y = gli_monthly.shift(-3) - gli_monthly
    y.name = 'target_gli_change'
    #print(y)
    return y

def _make_reg_x(factor_a, factor_b, factor_c, factor_d):
    # GLIとのLAGからミラにずらした分を引いてGLIと合わせる
    fa = factor_a.shift(7-3)
    fb = factor_b.shift(14-3)
    fc = factor_c.shift(3-3)
    fd = factor_d.shift(38-3)

    x = pd.concat([fa, fb, fc, fd], axis=1).dropna(how="all")

    # インデックスを合わせる
    start = x.apply(pd.Series.first_valid_index).max()
    end  = x.apply(pd.Series.last_valid_index).min()
    x = x.loc[start:end]
    #check_nan_time(x,"1990-01-01")
    #print(x)
    return x

########################################################
# 学習の変数
########################################################

def _make_label(target_monthly):
    # 週次にします
    target_lagged = target_monthly.shift(1)
    target_weekly = target_lagged.resample('W-FRI').interpolate(method='linear').dropna()

    #target_diff52 = target_weekly.diff(52)
    target_diff13 = target_weekly.diff(13)
    target_diff13 = target_diff13.loc["2010-01-01":]

    # 3か月後予測
    #future_change = target_diff13.shift(-13) - target_diff13
    future_change = target_diff13.shift(-13)
    lower_threshold = -53.642265  # 下位25% (Down)
    upper_threshold = 47.575755   # 上位25% (Up)
    # 1か月後予測
    #future_change = target_diff52.shift(-4) - target_diff52
    #lower_threshold = -31.075355  # 下位25% (Down)
    #upper_threshold = 30.315039   # 上位25% (Up)
    #print(future_change.describe())

    # 統計データから算出した閾値
    labels = pd.cut(
        future_change,
        bins=[-np.inf, lower_threshold, upper_threshold, np.inf],
        labels=[1, 2, 3]
    ).rename("gli_label")
    #print(labels.value_counts())

    return labels.dropna()

def _make_featuring(factor_a, factor_b, factor_c, factor_d):

    # lag調整した因子セットをつくる
    fa_level = factor_a.shift(7-3).rename("a_level")
    fb_level = factor_b.shift(14-3).rename("b_level")
    fc_level = factor_c.shift(3-3).rename("c_level")
    fd_level = factor_d.shift(38-3).rename("d_level")

    # モメンタム
    fa_mom1 = fa_level.diff().rename("a_mom1")
    fb_mom1 = fb_level.diff().rename("b_mom1")
    fc_mom1 = fc_level.diff().rename("c_mom1")
    fd_mom1 = fd_level.diff().rename("d_mom1")

    # モメンタム
    fa_mom = fa_level.diff(3).rename("a_mom")
    fb_mom = fb_level.diff(3).rename("b_mom")
    fc_mom = fc_level.diff(3).rename("c_mom")
    fd_mom = fd_level.diff(3).rename("d_mom")

    # ボラ
    fa_vol = fa_level.rolling(6).std().rename("a_vol")
    fb_vol = fb_level.rolling(6).std().rename("b_vol")
    fc_vol = fc_level.rolling(6).std().rename("c_vol")
    fd_vol = fd_level.rolling(6).std().rename("d_vol")

    # 変化の加速
    fa_acc = fa_mom.diff(1).rename("a_acc")
    fb_acc = fb_mom.diff(1).rename("b_acc")
    fc_acc = fc_mom.diff(1).rename("c_acc")
    fd_acc = fd_mom.diff(1).rename("d_acc")

    # 交差項
    ab_cross = (fa_level * fb_level).rename("a_b_cross")
    ac_cross = (fa_level * fc_level).rename("a_c_cross")
    ad_cross = (fa_level * fd_level).rename("a_d_cross")
    bc_cross = (fb_level * fc_level).rename("b_c_cross")
    bd_cross = (fb_level * fd_level).rename("b_d_cross")
    cd_cross = (fc_level * fd_level).rename("c_d_cross")

    df_featured = pd.concat([
        fa_level,#
        fb_level,#
        fc_level,
        fd_level,#
        fa_mom1,
        fb_mom1,
        fc_mom1,
        fd_mom1,
        fa_mom,
        fb_mom,#
        fc_mom,
        fd_mom,
        #fa_vol,
        #fb_vol,#
        #fc_vol,
        #fd_vol,#
        #fa_acc,
        #fb_acc,
        #fc_acc,
        #fd_acc,
        #ab_cross
        #ac_cross,
        #ad_cross,
        #bc_cross,
        #bd_cross,
        #cd_cross,
        #ac_cross
        ], axis=1).dropna()

    #print(df_featured)
    return df_featured






########################################################
# 実装確認・デバッグ
########################################################

gli_index = [
    "gli",
    "BUSLOANS",
    "CP",
    "PNFIC1",
    "DSPIC96",
    "UNRATE",
    "CES0500000003",
    "SOFR",
    "DFF",
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
    "DTB3"
    ]

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












#----------------------------------------------------------------------------------------------
def get_gli_dfa_modeling_beta(
    dxy, vix, mkt_credit_spread, nfci_risk, frb_total_assets, on_rrp, tga_deposit, gli, m2
    ):
    # --- 前処理 ---
    df = _preprocess_for_gli_dfa_modeling(
        dxy, vix, mkt_credit_spread, nfci_risk,
        frb_total_assets, on_rrp, tga_deposit, gli
        )
    #print(df)

    # --- DFA ---
    res_series, results, is_inverted = learning_dfa(
        df, factors=1, factor_orders=2,
        target_col="dxy_log_diff",
        output_name="gli_dfa_model"
        )   
    #print(results.summary())
    #print(results.params)
    #print(res_series)

    # --- 寄与度の計算 ---
    contributions = df.copy()
    loadings = results.params.filter(like='loading')
    for i, col in enumerate(df.columns):
        contributions[col] = df[col] * loadings.iloc[i]
    contributions_rolling = contributions.rolling(window=4, min_periods=1).mean()
    #pd.set_option('display.max_rows', None)
    #print(contributions_rolling)

    # --- 将来予測 ---
    res_forecast, results_forecast, is_inverted_forecast = learning_dfa(
        df, factors=1, factor_orders=1,
        target_col="dxy_log_diff",
        output_name="gli_dfa_model"
        )
    forecast_mean, forecast_ci = learning_forecast(results_forecast)
    #print(results.summary())
    #print(forecast_mean)
    #print(forecast_ci)

    # --- 後処理 ---
    df_result = _postprocess_gli_dfa_modeling(res_series, forecast_mean, forecast_ci, gli, m2)
    #print(df_result)

    # --- 分析/可視化 ---
    _plot_contribution(contributions_rolling)
    #_plot_graphs(df_result["model_cumsum_z"],df_result["forecast_cumsum"])
    #_plot_graphs(df_result["model_cumsum_z"],on_rrp)
    #_plot_graphs(df_result["model_cumsum_z"],df_result["forecast_cumsum"])

def _preprocess_for_gli_dfa_modeling(
    dxy, vix, mkt_credit_spread, nfci_risk, frb_total_assets, on_rrp, tga_deposit, gli
    ):
    # --- ネットリクイディティの算出 ---
    # 日次にし、単位を合わせる
    frb_total_assets_d = frb_total_assets.resample("D").ffill()
    on_rrp_d = on_rrp.resample("D").ffill()*1000
    tga_deposit_d = tga_deposit.resample("D").ffill()
    # 計算する
    df_sub = pd.concat([frb_total_assets_d, on_rrp_d, tga_deposit_d], axis=1).dropna()
    net_liquidity = df_sub["frb_total_assets"] - (df_sub["on_rrp"] + df_sub["tga_deposit"])
    
    # --- 対数処理 ---
    dxy_log = np.log(dxy);dxy_log.name = "dxy_log"
    vix_log = np.log(vix);vix_log.name = "vix_log"
    net_liquidity_log = np.log(net_liquidity);net_liquidity_log.name = "net_liquidity_log"

    # --- データフレーム統合 ---
    df = pd.concat([dxy_log, vix_log, mkt_credit_spread, nfci_risk, net_liquidity_log], axis=1)
    # tga_depositが最も取得開始が遅いのでtga_depositに合わせる
    start_inedx = tga_deposit[tga_deposit.notna()].index[0]
    df = df.loc[start_inedx:]
    
    # --- 週次サンプリングに統一する ---
    df_w = df.resample('W-FRI').last().ffill()
    
    # --- diffをとる ---
    df_processed = pd.DataFrame(index=df_w.index)
    df_processed['dxy_log_diff'] = df_w['dxy_log'].diff()
    df_processed['vix_log_diff'] = df_w['vix_log'].diff()
    df_processed['net_liquidity_log_diff'] = df_w['net_liquidity_log'].diff()
    df_processed['mkt_credit_spread_diff'] = df_w['mkt_credit_spread'].diff()
    df_processed['nfci_risk_diff'] = df_w['nfci_risk'].diff()

    df_processed = df_processed.dropna()

    # --- 4半期データのGLI差分の結合 ---
    # 差分をとって取得開始日を合わせる
    gli_diff = gli.diff();gli_diff.name = "gli_diff"
    gli_diff = gli_diff.loc[df.index[0]:]

    # 最も近い週次の位置にデータをいれる- ffillせずにNanのままにする
    df_processed = pd.merge_asof(
        df_processed.sort_index(),
        gli_diff.sort_index(),
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('3 days') # 1週間以上離れたデータは無視
    )
    df_processed.index = pd.to_datetime(df_processed.index)
    df_processed = df_processed.asfreq('W-FRI')

    return df_processed

def _postprocess_gli_dfa_modeling(
    res_series, forecast_mean, forecast_ci, gli, m2
    ):

    df_result = pd.DataFrame(index=res_series.index)

    # --- モデルの累積和のZスコア ---
    cumsum = res_series.cumsum()
    cumsum = cumsum.rolling(window=4).mean()
    df_result["model_cumsum_z"] = learning_standard_scalar(cumsum).dropna()

    # --- 将来予測 ---
    last_actual_val = df_result["model_cumsum_z"].iloc[-1]
    first_forecast_date = df_result.index[-1]
    initial_series = pd.Series([last_actual_val], index=[first_forecast_date])

    forecast_cumsum = forecast_mean.iloc[:, 0].cumsum() + last_actual_val
    forecast_cumsum = pd.concat([initial_series, forecast_cumsum])
    forecast_cumsum.name = "forecast_cumsum"
    df_result = pd.concat([df_result, forecast_cumsum], axis=1)

    # --- GLI/M2の実績とZスコア ---
    df_result["gli_actual"] = gli.reindex(index=res_series.index, method="ffill")
    df_result["gli_z"] = learning_standard_scalar(df_result["gli_actual"])

    df_result["m2_actual"] = m2.reindex(index=res_series.index, method="ffill")
    df_result["m2_z"] = learning_standard_scalar(df_result["m2_actual"])

    return df_result

########################################################
# 金利押し上げ圧力インデックスのモデリング
########################################################
    """
    1.考え方

    2.説明変数の選定理由

    3.将来予測

    """

def get_macro_potential_index_dfa_modeling(
    tnx, t5yifr, t10y2y, cpi_us_fred, move, uup, tlt, gold_future, copper_future, tips
    ):
    # --- 前処理 ---
    df = _preprocess_macro_potential_index_dfa_modeling(
        tnx, t5yifr, t10y2y, cpi_us_fred, move, uup, tlt, gold_future, copper_future,
    )
    # --- DFA ---
    res_series, results, is_inverted = learning_dfa(
        df, factors=1, factor_orders=2, 
        sign_name="tnx_diff",
        output_name="real_interest_rate__dfa_model"
    )
    # --- 後処理 ---
    df_result = _postprocess_macro_potential_index_dfa_modeling(res_series, tnx, tips)
    latest_spread = df_result['macro_spread'].iloc[-1]
    histrical_percentage = (df_result['macro_spread'] < latest_spread).mean()

    # --- グラフ描画 ---
    #_compare_graphs(df_result["macro_spread"], df_result["macro_spread"])

    return df_result, latest_spread, histrical_percentage

def _preprocess_macro_potential_index_dfa_modeling(
    tnx, t5yifr, t10y2y, cpi_us_fred, move, uup, tlt, gold_future, copper_future
    ):

    # リサンプリング前の処理
    cpi_us_fred_mom = np.log(cpi_us_fred).diff()    # cpi_us_fred は月次発表なので先に処理をする
    cpi_us_fred_mom.name = "cpi_us_fred_mom"

    move_log = np.log(move)     # 指数急騰（スパイク）対策
    move_log.name = "move_log"

    uup_log = np.log(uup)       # 指数急騰（スパイク）対策
    uup_log.name = "uup_log"

    tlt_log = np.log(tlt)       # 指数急騰（スパイク）対策
    tlt_log.name = "tlt_log"

    cg_ratio = copper_future / gold_future     # Copper/Gold 比率
    cg_log = np.log(cg_ratio)
    cg_log.name = "cg_log"

    # データフレーム統合
    df = pd.concat([tnx, t5yifr, t10y2y, cpi_us_fred_mom, move_log, uup_log, tlt_log, cg_log], axis=1)
    # uupが最も取得開始が遅いのでuupに合わせる
    start_inedx = uup[uup.notna()].index[0]
    df = df.loc[start_inedx:]
    
    # ほとんど日次データだがcpiだけ月次なので、間をとって週次サンプリングにする
    df_w = df.resample('W-FRI').last().ffill()

    df_processed = pd.DataFrame(index=df_w.index)

    # diffをとる指標
    df_processed["tnx_diff"] = df_w["tnx"].diff()
    df_processed["t5yifr_diff"] = df_w["t5yifr"].diff()
    df_processed["t10y2y_diff"] = df_w["t10y2y"].diff()
    df_processed["move_log_diff"] = df_w["move_log"].diff()
    df_processed["uup_log_diff"] = df_w["uup_log"].diff()
    df_processed["tlt_log_diff"] = df_w["tlt_log"].diff()
    df_processed["cg_log_diff"] = df_w["cg_log"].diff()

    # 発表ラグの調整
    df_processed["cpi_us_fred_mom"] = df_w["cpi_us_fred_mom"].shift(4)

    df_processed = df_processed.dropna()

    return df_processed

def _postprocess_macro_potential_index_dfa_modeling(res_series, tnx, tips):
    
    df_result = pd.DataFrame(index=res_series.index)

    # モデルの累積和のZスコア
    cumsum = res_series.rolling(window=12,center=True).mean().cumsum()
    df_result["model_cumsum_z"] = learning_standard_scalar(cumsum).dropna()

    #TNXの実績とZスコア
    df_result["tnx_actual"] = tnx.reindex(index=res_series.index, method="ffill")
    df_result["tnx_z"] = learning_standard_scalar(df_result["tnx_actual"])

    #Macro spread
    df_result["macro_spread"] = df_result["model_cumsum_z"] - df_result["tnx_z"]
    
    return df_result
