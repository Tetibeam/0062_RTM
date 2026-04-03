########################################################
# DSR予測モデル
########################################################
from batch.modeling.featuring import (
    get_columns_by_frequency,
    cap_by_sigma,
    standard_scalar_df,
    balanced_clip,
    cap_outliers
    )
from batch.modeling.visualize import (
    plot_index,
    _plot_lag_correlation,
    plot_gli_trajectory
    )
from batch.modeling.learning import (
    lag_analysis,
    learning_dfa,
    #learning_lgbm_test,
    learning_lgbm_final
    )
import pandas as pd
import numpy as np

dsr_index = [
    "dsr",

    "BUSLOANS",
    "BAA",
    "AAA",
    "DGS10",
    "TOTALSL",
    "REALLN",
    "DRTSCLCC",

    "DSPIC96",
    "PAYEMS",
    "UNRATE",
    "ECIWAG",
    "CP",
    "AWHMAN",
    "PNFIC1",
    "PSAVERT",
    "COMPRNFB",

    "MORTGAGE30US",
    "TERMCBCCALLNS",
    "T10Y2Y"
    ]

########################################################
# メインプロセス
########################################################
def get_dsr_model_beta(df_index):
    # --- データの取得 ---
    df = df_index[dsr_index]
    #check_nan_time(df,"1990-01-01")
    #pd.set_option('display.max_rows', None)
    #print(df["CP"].tail(300))

    # --- データ集計-日次は月次に、四半期は月次に線形補完する ---
    df = _aggregation(df)
    #check_nan_time(df,"1990-01-01")
    #pd.set_option('display.max_rows', None)
    #print(df["CP"].tail(300))

    # --- 特徴量抽出 ---
    df_a, df_b, df_c =  _featuring(df)
    #check_nan_time(df_d,"1990-01-01")

    # --- gli のデータを前月末尾にする ---
    df_dsr = df_index["dsr"].dropna()
    df_dsr.index = df_dsr.index - pd.offsets.MonthEnd(1)
    #print(df_gli)

    # --- 特徴量とGLIのラグ相関分析 ---
    _lag_corr_check(df_a, df_b, df_c, df_dsr)

    # --- DFA前にラグを調整する ---
    df_a, df_b, df_c = _lag_adjustment(df_a, df_b, df_c)
    #check_nan_time(df_a,"1990-01-01")
    #check_nan_time(df_b,"1990-01-01")
    #check_nan_time(df_c,"1990-01-01")
    df_a = df_a.loc["2010-01-01":]
    df_b = df_b.loc["2010-01-01":]
    df_c = df_c.loc["2010-01-01":]

    #aaa = df_dsr.resample('ME').interpolate(method='linear').dropna()
    #df_dsr_yoy = aaa.pct_change(12).dropna().rename("dsr_yoy")
    #df_dsr_diff = aaa.diff(12).dropna().rename("dsr")
    #plot_df = pd.concat([df_b,df_dsr_diff], axis=1).dropna()
    #plot_df = standard_scalar_df(plot_df)
    #plot_index(plot_df.dropna())

    # --- 符号を反転させる ---
    # DSRを押し上げる力（＝ストレス）を正に統一する
    df_a, df_b, df_c = _sign_adjustment(df_a, df_b, df_c)

    # --- DFA ---
    factor_a, results_a, _ = learning_dfa(df_a, factors=1, factor_orders=2, target_col="DRTSCLCC_ma4_sync", target_is_positive=True, output_name="FactorA_Supply")
    factor_b, results_b, _ = learning_dfa(df_b, factors=1, factor_orders=2, target_col="UNRATE_diff_clip_sync", target_is_positive=True, output_name="FactorB_Supply")
    factor_c, results_c, _ = learning_dfa(df_c, factors=1, factor_orders=2, target_col="MORTGAGE30US_level_sync", target_is_positive=True, output_name="FactorC_Supply")
    #print(results_a.summary())
    #print(results_b.summary())
    #print(results_c.summary())
    #plot_index(pd.concat([factor_a,factor_b,factor_c, df_gli.reindex(factor_c.index)], axis=1).dropna())
    #print(factor_a.tail(), factor_b.tail(), factor_c.tail(), factor_d.tail())

    # --- 因子とGLI-diffの可視化 ---
    #_check_factor(factor_a, df_dsr, df_index["^GSPC"])
    #_check_factor(factor_b, df_dsr, df_index["^GSPC"])
    #_check_factor(factor_c, df_dsr, df_index["^GSPC"])

    # --- 教師ラベルの生成 ---
    y = _make_label_dsr(df_dsr)

    # --- 学習用特徴量の生成 ---
    X = _make_featuring(factor_a, factor_b, factor_c)
    # --- 学習 ---
    df_master = pd.concat([X, y], axis=1).dropna()
    #plot_index(df_master)
    df_oof_all = learning_lgbm_test(
        df_master, target_col="dsr_label",labels=["1:LIFT", "2:CRUISE", "3:STALL"],
        n_splits=3, gap=3,
        n_estimators=300,learning_rate=0.01,num_leaves=24, min_data_in_leaf=14,
        reg_alpha=0.5, reg_lambda=0.5,
        learning_curve=True,
    )
    #learning_lgbm_final(
    #    df_master, target_col="dsr_label",model_name="DSR Pressure", label_name_list=["1:LIFT", "2:CRUISE", "3:STALL"],
    #    option_feat_imp=True,
    #    option_tscv=True
    #)

    # --- 学習結果の分析・可視化 ---
    #plot_gli_trajectory(df_trajectory, df_index["gli"].ffill(),df_index["^GSPC"], start_date="2010-01-01")

    return df_oof_all

def _aggregation(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")].dropna(how="all")
    df_weekly = df[get_columns_by_frequency(df, target="weekly")].dropna(how="all")
    df_monthly = df[get_columns_by_frequency(df, target="monthly")].dropna(how="all")
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")].dropna(how="all")
    #pd.set_option("display.max_rows", None)
    #print(df_quarterly.columns,df_weekly.columns,df_monthly.columns,df_quarterly.columns)
    #print(df_quarterly.head(80))
    #print(df_quarterly.tail(80))
    #check_nan_time(df_quarterly,"1990-01-01")

    df_daily_m = df_daily.resample('ME').mean()
    df_weekly_m = df_weekly.resample('ME').mean()
    df_monthly = df_monthly.resample('ME').mean()

    q_m = df_quarterly.dropna(how="all")
    q_m.index = q_m.index + pd.offsets.MonthEnd(0)
    df_quarterly_m  = q_m.resample('ME').asfreq().interpolate(method='linear', limit_area="inside")

    pd.set_option("display.max_rows", None)
    #df_disp=df_quarterly_m
    #print(df_disp.columns,df_weekly.columns,df_monthly.columns,df_quarterly.columns)
    #print(df_disp.head(80))
    #print(df_disp.tail(80))
    #check_nan_time(df_disp,"1990-01-01")

    # 結合
    df_combine = pd.concat([df_daily_m, df_weekly_m, df_monthly, df_quarterly_m], axis=1)
    # 最後の月は削除 未来リーク
    df_combine = df_combine.iloc[:-1]

    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_combine.tail(20))
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine

def _lag_corr_check(df_a, df_b, df_c, df_dsr):

    # GLI をdiffにする
    df_dsr = df_dsr.resample('ME').interpolate(method='linear').dropna()
    #df_dsr_yoy = df_dsr.pct_change(4).dropna().rename("dsr_yoy")
    df_dsr_diff = df_dsr.diff(12).dropna().rename("dsr_diff")

    # GLI のインデックスに合わせる
    #df_a = standard_scalar_df(df_a)
    df_a_all = pd.concat([df_a.reindex(df_dsr_diff.index), df_dsr_diff], axis=1).dropna()
    df_b_all = pd.concat([df_b.reindex(df_dsr_diff.index), df_dsr_diff], axis=1).dropna()
    df_c_all = pd.concat([df_c.reindex(df_dsr_diff.index), df_dsr_diff], axis=1).dropna()

    # 特徴量とGLIのラグ相関分析
    df_lag_a = lag_analysis(df_a_all, target_col="dsr_diff", max_lag=72)
    df_lag_b = lag_analysis(df_b_all, target_col="dsr_diff", max_lag=72)
    df_lag_c = lag_analysis(df_c_all, target_col="dsr_diff", max_lag=72)

    # 結果の確認・デバッグ
    #check_nan_time(df_b_all,"1990-01-01")
    #print(df_b_all)

    #_plot_lag_correlation(df_lag_a)
    #_plot_lag_correlation(df_lag_b)
    #_plot_lag_correlation(df_lag_c)

    #pd.set_option('display.max_rows', None)
    #print(df_lag_a)
    #plot_index(df_b_all)

def _lag_adjustment(df_a, df_b, df_c):

    # Layer A
    df_a["BUSLOANS_yoy_clip_sync"] = df_a["BUSLOANS_yoy_clip"].shift(8)#8
    #df_a["Baa_Aaa_Spread_clip_sync"] = df_a["Baa_Aaa_Spread_clip"].shift(5)#5
    #df_a["Baa_DGS10_Spread_clip_sync"] = df_a["Baa_DGS10_Spread_clip"].shift(6)#6
    df_a["TOTALSL_yoy_sync"] = df_a["TOTALSL_yoy"].shift(5)#8
    df_a["REALLN_yoy_sync"] = df_a["REALLN_yoy"].shift(0)#8
    df_a["DRTSCLCC_ma4_sync"] = df_a["DRTSCLCC_ma4"].shift(9)#8
    df_a = df_a.drop(columns=["BUSLOANS_yoy_clip", "TOTALSL_yoy", "REALLN_yoy", "DRTSCLCC_ma4"])


    # Layer B
    #df_b["DSPIC96_yoy_clip_sync"] = df_b["DSPIC96_yoy_clip"].shift(6)#
    #df_b["PAYEMS_yoy_clip_sync"] = df_b["PAYEMS_yoy_clip"].shift(6)#
    #df_b["AWHMAN_diff_clip_sync"] = df_b["AWHMAN_diff_clip"].shift(14)#
    #df_b["ECIWAG_yoy_sync"] = df_b["ECIWAG_yoy"].shift(0)#
    df_b["UNRATE_diff_clip_sync"] = df_b["UNRATE_diff_clip"].shift(6)#
    df_b["CP_yoy_clip_sync"] = df_b["CP_yoy_clip"].shift(17)#
    #df_b["PNFIC1_yoy_clip_sync"] = df_b["PNFIC1_yoy_clip"].shift(8)#
    df_b["PSAVERT_level_clip_sync"] = df_b["PSAVERT_level_clip"].shift(0)#
    df_b["COMPRNFB_yoy_clip_sync"] = df_b["COMPRNFB_yoy_clip"].shift(3)#
    df_b = df_b.drop(columns=[
        "DSPIC96_yoy_clip", "PAYEMS_yoy_clip", "AWHMAN_diff_clip",
        "ECIWAG_yoy", "UNRATE_diff_clip", "CP_yoy_clip", "PNFIC1_yoy_clip",
        "PSAVERT_level_clip", "COMPRNFB_yoy_clip"
        ])
    #print(df_b.columns)

    # Layer C
    df_c["MORTGAGE30US_level_sync"] = df_c["MORTGAGE30US_level"].shift(22)#
    df_c["TERMCBCCALLNS_level_sync"] = df_c["TERMCBCCALLNS_level"].shift(23)#56
    df_c["T10Y2Y_level_sync"] = df_c["T10Y2Y_level"].shift(0)#

    df_c = df_c.drop(columns=["MORTGAGE30US_level", "TERMCBCCALLNS_level", "T10Y2Y_level"])

    start = df_a.apply(pd.Series.first_valid_index).max()
    df_a = df_a[start:]
    start = df_b.apply(pd.Series.first_valid_index).max()
    df_b = df_b[start:]
    start = df_c.apply(pd.Series.first_valid_index).max()
    df_c = df_c[start:]

    #plot_index(df_a)
    #plot_index(df_b)
    #plot_index(df_c)

    #print(df_a.head(10))
    #print(df_b.head(10))
    #print(df_c.head(10))

    return df_a, df_b, df_c

def _sign_adjustment(df_a, df_b, df_c):

    df_a["BUSLOANS_yoy_clip_sync"] *= -1
    #df_a["Baa_Aaa_Spread_clip_sync"] *= -1
    #df_a["Baa_DGS10_Spread_clip_sync"] *= -1
    df_a["DRTSCLCC_ma4_sync"] *= -1

    #df_b["DSPIC96_yoy_clip_sync"] *= -1
    df_b["UNRATE_diff_clip_sync"] *= -1
    #df_b["PNFIC1_yoy_clip_sync"] *= -1
    df_b["PSAVERT_level_clip_sync"] *= -1
    df_b["COMPRNFB_yoy_clip_sync"] *= -1

    df_c["MORTGAGE30US_level_sync"] *= -1
    df_c["TERMCBCCALLNS_level_sync"] *= -1
    df_c["T10Y2Y_level_sync"] *= -1

    return df_a, df_b, df_c

def _check_factor(df, dsr, sp500):
    # GLIとSP500を因子のインデックスに合わせる
    df_dsr = dsr.resample('ME').interpolate(method='linear').reindex(df.index, method="ffill")
    df_dsr_diff = df_dsr.diff(12).dropna().rename("DSR_diff")

    df_sp500 = sp500.resample('ME').mean().dropna()
    #sp500_ret_cumsum = df_sp500.pct_change().cumsum().dropna()
    #sp500_ret_cumsum = sp500_ret_cumsum.reindex(df.index, method="ffill").rename("sp500_ret_cumsum")

    #df = df.cumsum().dropna()

    plot_df = pd.concat([df, df_dsr_diff], axis=1).dropna()
    
    # Zスコア化
    scaled_cols = standard_scalar_df(plot_df[['Factor_Supply', 'DSR_diff']])
    plot_df['Factor_Z'] = scaled_cols['Factor_Supply']
    plot_df['DSR_diff_Z'] = scaled_cols['DSR_diff']
    plot_df = plot_df.drop(columns=['Factor_Supply', 'DSR_diff'])

    # グラフ描画
    plot_index(plot_df[["Factor_Z", "DSR_diff_Z"]])

########################################################
# 特徴量抽出
########################################################
def _featuring(df):

    df_a = _featuring_a(df)
    df_b = _featuring_b(df)
    df_c = _featuring_c(df)

    return df_a, df_b, df_c

def _featuring_a(df):
    #col = ["BUSLOANS","BAA","AAA","DGS10"]
    col = ["BUSLOANS","TOTALSL", "REALLN","DRTSCLCC","BAA","AAA","DGS10"]
    df_ = df[col].dropna(how="all")
    #print(df_.tail(10))
    #check_nan_time(df_,"1990-01-01")

    yoy_BUSLOANS = df_["BUSLOANS"].pct_change(12).dropna().rename("BUSLOANS_yoy")
    clip_BUSLOANS = yoy_BUSLOANS.clip(-0.15, 0.18).rename("BUSLOANS_yoy_clip").dropna()
    spd_BAA_AAA = (df_["BAA"] - df_["AAA"]).dropna().rename("Baa_Aaa_Spread")
    clip_BAA_AAA = spd_BAA_AAA.clip(0, 1.8).rename("Baa_Aaa_Spread_clip")
    spd_BAA_DH10 = (df_["BAA"] - df_["DGS10"]).dropna().rename("Baa_DGS10_Spread")
    clip_BAA_DH10 = spd_BAA_DH10.clip(0, 4).rename("Baa_DGS10_Spread_clip")

    yoy_TOTALSL = df_["TOTALSL"].pct_change(12).dropna().rename("TOTALSL_yoy")
    #clip_TOTALSL = yoy_TOTALSL.clip(-0.15, 0.18).rename("TOTALSL_yoy_clip").dropna()
    yoy_REALLN = df_["REALLN"].pct_change(12).dropna().rename("REALLN_yoy")
    #clip_REALLN = yoy_REALLN.clip(-0.15, 0.18).rename("REALLN_yoy_clip").dropna()
    level_DRTSCLCC = df_["DRTSCLCC"].dropna().rename("DRTSCLCC_level")
    ma4_DRTSCLCC = level_DRTSCLCC.clip(-20, 40).rolling(4).mean().rename("DRTSCLCC_ma4").dropna()


    df_feature = pd.concat([clip_BUSLOANS, yoy_TOTALSL, yoy_REALLN, ma4_DRTSCLCC], axis=1).dropna(how="all")

    start = df_feature.apply(pd.Series.first_valid_index).max()
    end   = df_feature.apply(pd.Series.last_valid_index).max() #末尾はNanで残してカルマンフィルタを使う
    df_feature = df_feature.loc[start:end]

    #check_nan_time(df_feature,"1990-01-01")
    #plot_index(df_feature)
    #print(df_feature.head(10))

    return df_feature

def _featuring_b(df):
    col = ["DSPIC96","PAYEMS","UNRATE", "ECIWAG", "CP", "AWHMAN", "PNFIC1", "PSAVERT", "COMPRNFB"]
    df_ = df[col].dropna(how="all")
    #print(df_.tail(10))
    #check_nan_time(df_,"1990-01-01")
    
    yoy_DSPIC96 = df_["DSPIC96"].pct_change(12).dropna().rename("DSPIC96_yoy")
    clip_DSPIC96 = yoy_DSPIC96.clip(-0.05, 0.08).rename("DSPIC96_yoy_clip")

    yoy_PAYEMS = df_["PAYEMS"].pct_change(12).dropna().rename("PAYEMS_yoy")
    clip_PAYEMS = yoy_PAYEMS.clip(-0.07, 0.06).rename("PAYEMS_yoy_clip")

    diff_AWHMAN = df_["AWHMAN"].diff(12).dropna().rename("AWHMAN_diff")
    clip_AWHMAN  = diff_AWHMAN.clip(-2, 1.5).rename("AWHMAN_diff_clip")

    yoy_ECIWAG = df_["ECIWAG"].pct_change(12).dropna().rename("ECIWAG_yoy")

    diff_UNRATE = df_["UNRATE"].diff(12).dropna().rename("UNRATE_diff")
    clip_UNRATE = diff_UNRATE.clip(-1.5, 1.5).rename("UNRATE_diff_clip")

    yoy_CP = df_["CP"].pct_change(12).dropna().rename("CP_yoy")
    clip_CPE = yoy_CP.clip(-0.1, 0.3).rename("CP_yoy_clip")

    yoy_PNFIC1 = df_["PNFIC1"].pct_change(12).dropna().rename("PNFIC1_yoy")
    clip_PNFIC1 = yoy_PNFIC1.clip(-0.1,1).rename("PNFIC1_yoy_clip")
    
    upper_limit = df_['PSAVERT'].quantile(0.95)
    clip_PSAVERT = df_['PSAVERT'].clip(upper=upper_limit).rolling(6).mean().rename("PSAVERT_level_clip")
    
    yoy_COMPRNFB = df_["COMPRNFB"].diff(12).dropna().rename("COMPRNFB_yoy")
    clip_COMPRNFB = yoy_COMPRNFB.clip(-5,5).rename("COMPRNFB_yoy_clip")

    df_feature = pd.concat([
        clip_UNRATE, clip_CPE, clip_PNFIC1, clip_PAYEMS,clip_AWHMAN,clip_DSPIC96,yoy_ECIWAG, clip_PSAVERT,clip_COMPRNFB
        ], axis=1).dropna(how="all")

    start = df_feature.apply(pd.Series.first_valid_index).max()
    end   = df_feature.apply(pd.Series.last_valid_index).max() #末尾はNanで残してカルマンフィルタを使う
    df_feature = df_feature.loc[start:end]

    #check_nan_time(df_feature,"1990-01-01")
    #plot_index(df_feature)
    #print(df_feature.head(10))

    return df_feature

def _featuring_c(df):
    col = ["MORTGAGE30US","TERMCBCCALLNS","T10Y2Y"]
    df_ = df[col].dropna(how="all")
    #print(df_.tail(10))
    #check_nan_time(df_,"1990-01-01")

    level_MORTGAGE30US= df_["MORTGAGE30US"].dropna().rename("MORTGAGE30US_level")
    level_TERMCBCCALLNS= df_["TERMCBCCALLNS"].dropna().rename("TERMCBCCALLNS_level")
    level_T10Y2Y= df_["T10Y2Y"].dropna().rename("T10Y2Y_level")

    df_feature = pd.concat([
        level_MORTGAGE30US, level_TERMCBCCALLNS, level_T10Y2Y
        ], axis=1).dropna(how="all")

    start = df_feature.apply(pd.Series.first_valid_index).max()
    end   = df_feature.apply(pd.Series.last_valid_index).max() #末尾はNanで残してカルマンフィルタを使う
    df_feature = df_feature.loc[start:end]

    #check_nan_time(df_feature,"1990-01-01")
    #plot_index(df_feature)
    #print(df_feature.head(10))

    return df_feature

########################################################
# 回帰の変数
########################################################
def _make_reg_y(df_dsr):
    #print(df_dsr)
    gli_monthly = df_dsr.resample('ME').first().interpolate(method='linear')
    #print(gli_monthly)
    y = gli_monthly.shift(-3) - gli_monthly
    y.name = 'target_dsr_change'
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
def _make_label_dsr(df_dsr):
    # 線形月次変換
    dsr_monthly = df_dsr.resample('ME').first().interpolate(method='linear')

    # カンニングラベル
    y_diff = dsr_monthly.diff(3)
    target = y_diff.shift(-6)
    print(target.describe())

    # 統計データから算出した閾値
    lower_threshold = -0.05  # 下位25% (Down)
    upper_threshold = 0.01   # 上位25% (Up)

    labels = pd.cut(
        target,
        bins=[-np.inf, lower_threshold, upper_threshold, np.inf],
        labels=[1, 2, 3]
    ).rename("dsr_label")
    #print(labels)

    return labels

def _make_featuring(factor_a, factor_b, factor_c):

    # lag調整した因子セットをつくる
    fa_level = factor_a.shift(0).rename("a_level")
    fb_level = factor_b.shift(0).rename("b_level")
    fc_level = factor_c.shift(0).rename("c_level")

    # モメンタム
    fa_mom1 = fa_level.diff().rename("a_mom1")
    fb_mom1 = fb_level.diff().rename("b_mom1")
    fc_mom1 = fc_level.diff().rename("c_mom1")

    # モメンタム
    fa_mom = fa_level.diff(3).rename("a_mom")
    fb_mom = fb_level.diff(3).rename("b_mom")
    fc_mom = fc_level.diff(3).rename("c_mom")

    # ボラ
    fa_vol = fa_level.rolling(6).std().rename("a_vol")
    fb_vol = fb_level.rolling(6).std().rename("b_vol")
    fc_vol = fc_level.rolling(6).std().rename("c_vol")

    # 変化の加速
    fa_acc = fa_mom.diff(1).rename("a_acc")
    fb_acc = fb_mom.diff(1).rename("b_acc")
    fc_acc = fc_mom.diff(1).rename("c_acc")

    # 交差項
    ab_cross = (fa_level * fb_level).rename("a_b_cross")
    ac_cross = (fa_level * fc_level).rename("a_c_cross")
    bc_cross = (fb_level * fc_level).rename("b_c_cross")
    ab_ratio = (fa_level / fb_level).rename("a_b_ratio")

    df_featured = pd.concat([
        fa_level,
        fb_level,
        fc_level,
        fa_mom1,
        fb_mom1,
        fc_mom1,
        fa_mom,
        fb_mom,#
        fc_mom,
        #fa_vol,
        #fb_vol,#
        #fc_vol,
        #fa_acc,
        #fb_acc,
        #fc_acc,
        ab_cross,
        #ac_cross,
        #bc_cross,
        #ab_ratio
        ], axis=1).dropna()

    #print(df_featured)
    return df_featured

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
