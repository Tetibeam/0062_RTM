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
    plot_index,
    _plot_lag_correlation,
    plot_gli_trajectory
    )
from batch.modeling.learning import (
    lag_analysis,
    learning_dfa,
    learning_lgbm_test,
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
    df = _make_target_variable(df)

    # --- 教師ラベルの生成 ---
    #df_label = _make_label_gli(df_gli)

    # --- データ集計-日次は月次に、四半期は月次に線形補完する ---
    df = _aggregation(df)
    #check_nan_time(df,"1990-01-01")
    #pd.set_option('display.max_rows', None)
    #print(df["CP"].tail(300))

    # --- 特徴量抽出 ---
    df_a, df_b, df_c, df_d =  _featuring(df)
    #check_nan_time(df_a,"1990-01-01")

    # --- gli のデータを前月末尾にする ---
    #df_gli = df_index["gli"].dropna()
    #df_gli.index = df_gli.index - pd.offsets.MonthEnd(1)
    #print(df_gli)

    # --- 特徴量とGLIのラグ相関分析 ---
    _lag_corr_check(df_a, df_b, df_c, df_d, df["NDFACBM027SBOG"])

    # --- DFA前にラグを調整する ---
    df_a, df_b, df_c, df_d = _lag_adjustment(df_a, df_b, df_c, df_d)
    #check_nan_time(df_c,"1990-01-01")

    # --- 学習用特徴量の生成 ---
    #X = _make_featuring(factor_a, factor_b, factor_c, factor_d)
    # --- 学習 ---
    #df_master = pd.concat([X, y], axis=1).dropna()
    #print(df_master)
    #plot_index(df_master)

    """df_oof_all, final_shap_dfs, df_oof_ev = learning_lgbm_test(
        df_master, target_col="gli_label",labels=["1:STALL", "2:CRUISE", "3:LIFT"],
        n_splits=2, gap=1,
        n_estimators=100,learning_rate=0.05,num_leaves=3, min_data_in_leaf=15,
        reg_alpha=10, reg_lambda=10, max_depth=2,min_child_samples= 5,
        learning_curve=True,
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
    df[['Final_Price_Stress']] = df[['Final_Price_Stress']].resample('MS').mean()

    return df

def _aggregation(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")]
    df_weekly = df[get_columns_by_frequency(df, target="weekly")]
    df_monthly = df[get_columns_by_frequency(df, target="monthly")]
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")]
    #print(df_daily.columns,df_weekly.columns,df_monthly.columns,df_quarterly.columns)
    #check_nan_time(df_quarterly,"1990-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.tail(80))

    df_daily_m = df_daily.resample('ME').mean()
    df_weekly_m = df_weekly.resample('ME').mean()
    df_monthly = df_monthly.resample('ME').mean()

    q_m = df_quarterly.dropna(how="all")
    q_m.index = q_m.index + pd.offsets.MonthEnd(0)
    df_quarterly_m  = q_m.resample('ME').asfreq().interpolate(method='linear', limit_area="inside")
    #check_nan_time(df_quarterly_m)
    #pd.set_option("display.max_rows", None)
    #print(df_quarterly_m.tail(10))

    # 結合
    df_combine = pd.concat([df_daily_m, df_weekly_m, df_monthly, df_quarterly_m], axis=1)
    # 最後の月は削除 未来リーク
    df_combine = df_combine.iloc[:-1] 

    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_combine.tail(20))
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine

def _lag_corr_check(df_a, df_b, df_c, df_d, target):

    # GLI をdiffにする
    #target = target.resample('ME').interpolate(method='linear').dropna()
    #df_gli_yoy = target.pct_change(4).dropna().rename("gli_yoy")
    target_diff = target.diff(12).dropna().rename("NDFACBM027SBOG_diff")

    # GLI のインデックスに合わせる
    #df_a = standard_scalar_df(df_a)
    df_a_all = pd.concat([df_a.reindex(target_diff.index), target_diff], axis=1).dropna()
    df_b_all = pd.concat([df_b.reindex(target_diff.index), target_diff], axis=1).dropna()
    df_c_all = pd.concat([df_c.reindex(target_diff.index), target_diff], axis=1).dropna()
    df_d_all = pd.concat([df_d.reindex(target_diff.index), target_diff], axis=1).dropna()

    # 特徴量とGLIのラグ相関分析
    df_lag_a = lag_analysis(df_a_all, target_col="NDFACBM027SBOG_diff", max_lag=72)
    df_lag_b = lag_analysis(df_b_all, target_col="NDFACBM027SBOG_diff", max_lag=72)
    df_lag_c = lag_analysis(df_c_all, target_col="NDFACBM027SBOG_diff", max_lag=72)
    df_lag_d = lag_analysis(df_d_all, target_col="NDFACBM027SBOG_diff", max_lag=72)

    # 結果の確認・デバッグ
    #check_nan_time(df_b_all,"1990-01-01")
    #plot_index(df_a_all)
    #print(df_d_all)

    #_plot_lag_correlation(df_lag_a)
    #_plot_lag_correlation(df_lag_b)
    #_plot_lag_correlation(df_lag_c)
    #_plot_lag_correlation(df_lag_d)

    pd.set_option('display.max_rows', None)
    #print(df_lag_d)

def _lag_adjustment(df_a, df_b, df_c, df_d):

    # Layer A
    df_a["BUSLOANS_yoy_sync"] = df_a["BUSLOANS_yoy"].shift(6)#12
    df_a["CP_yoy_sync"] = df_a["CP_yoy"].shift(0)#9
    df_a["PNFIC1_yoy_sync"] = df_a["PNFIC1_yoy"].shift(21)#28
    df_a = df_a.drop(columns=["BUSLOANS_yoy", "CP_yoy", "PNFIC1_yoy"])

    # Layer B
    #df_b["DSPIC96_yoy_clip_sync"] = df_b["DSPIC96_yoy_clip"].shift(5)#
    #df_b["UNRATE_diff_clip_sync"] = df_b["UNRATE_diff_clip"].shift(10)#
    df_b["PCE_yoy_clip_sync"] = df_b["PCE_yoy_clip"].shift(13)#
    df_b["PAYEMS_yoy_clip_sync"] = df_b["PAYEMS_yoy_clip"].shift(11)#
    df_b["CES0500000003_yoy_clip_sync"] = df_b["CES0500000003_yoy_clip"].shift(0)#
    df_b = df_b.drop(columns=["PAYEMS_yoy_clip", "PCE_yoy_clip", "CES0500000003_yoy_clip"])

    # Layer C
    df_c["SOFR_diff_clip_sync"] = df_c["SOFR_diff_clip"].shift(23)#
    df_c["DXY_yoy_clip_sync"] = df_c["DXY_yoy_clip"].shift(53)#56
    df_c["Liq_Spread_sync"] = df_c["SOFR_TB3MS_minus_clip"].shift(0)#
    df_c["Credit_Spread_sync"] = df_c["BAMLC0A4CBBB_minus_BAMLC0A3CA_clip"].shift(10)#
    df_c["Credit_Spread_diff_sync"] = df_c["BAMLC0A4CBBB_minus_BAMLC0A3CA_diff_clip"].shift(18)#
    df_c["Liq_Spread_diff_sync"] = df_c["SOFR_TB3MS_minus_diff_clip"].shift(5)#
    df_c = df_c.drop(columns=[
        "SOFR_diff_clip", "DXY_yoy_clip", "SOFR_TB3MS_minus_clip", "BAMLC0A4CBBB_minus_BAMLC0A3CA_clip",
        "BAMLC0A4CBBB_minus_BAMLC0A3CA_diff_clip", "SOFR_TB3MS_minus_diff_clip"])
    # Layer D
    df_d["Net_Liquidity_yoy_clip_sync"] = df_d["Net_Liquidity_yoy_clip"].shift(2)#
    df_d["Abs_Rate_clip_sync"] = df_d["Abs_Rate_clip"].shift(0)#
    df_d = df_d.drop(columns=["Net_Liquidity_yoy_clip", "Abs_Rate_clip"])


    start = df_a.apply(pd.Series.first_valid_index).max()
    df_a = df_a[start:]
    start = df_b.apply(pd.Series.first_valid_index).max()
    df_b = df_b[start:]
    start = df_c.apply(pd.Series.first_valid_index).max()
    df_c = df_c[start:]
    start = df_d.apply(pd.Series.first_valid_index).max()
    df_d = df_d[start:]

    #print(df_a.head(10))
    #print(df_b.head(10))
    #print(df_c.head(10))
    #print(df_d.head(10))

    return df_a, df_b, df_c, df_d

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
    #print(df_.tail(10))
    #check_nan_time(df_,"1990-01-01")
    #start = df_.apply(pd.Series.first_valid_index).max()
    #end   = df_.apply(pd.Series.last_valid_index).max() #末尾はNanで残してカルマンフィルタを使う
    #df_ = df_.loc[start:end]
    #check_nan_time(df_,"1990-01-01")
    #plot_index(df_)
    yoy_BUSLOANS = df_["BUSLOANS"].pct_change(12).dropna().rename("BUSLOANS_yoy")
    yoy_CP = df_["CP"].pct_change(12).dropna().rename("CP_yoy")
    yoy_PNFIC1 = df_["PNFIC1"].pct_change(12).dropna().rename("PNFIC1_yoy")

    #df_feature = pd.concat([yoy_BUSLOANS, yoy_CP, yoy_PNFIC1], axis=1).dropna(how="all")
    df_feature = pd.concat([yoy_BUSLOANS, yoy_CP], axis=1).dropna(how="all")
    #print(df_feature.head(10))

    return df_feature

def _featuring_b(df):
    #col = ["DSPIC96","UNRATE","CES0500000003"]
    col = ["PAYEMS","PCE","CES0500000003"]
    df_ = df[col].dropna(how="all")
    #print(df_.tail())
    #check_nan_time(df_,"1990-01-01")
    #start = df_.apply(pd.Series.first_valid_index).max()
    #end   = df_.apply(pd.Series.last_valid_index).max() #末尾はNanで残してカルマンフィルタを使う
    #df_ = df_.loc[start:end]
    #check_nan_time(df_,"1990-01-01")
    #print(df_)
    #plot_index(df_)

    #yoy_DSPIC96 = df_["DSPIC96"].pct_change(12).dropna().rename("yoy_DSPIC96")
    #clip_DSPIC96 = yoy_DSPIC96.clip(-0.15, 0.18).rename("DSPIC96_yoy_clip")

    yoy_PAYEMS = df_["PAYEMS"].pct_change(12).dropna().rename("yoy_PAYEMS")
    #clip_PAYEMS = yoy_PAYEMS.clip(-0.06, 0.06).rename("PAYEMS_yoy_clip")

    yoy_PCE = df_["PCE"].pct_change(12).dropna().rename("yoy_PCE")
    #clip_PCE = yoy_PCE.clip(-0.05, 0.16).rename("PCE_yoy_clip")

    #diff_UNRATE = df_["UNRATE"].diff(12).dropna().rename("diff_UNRATE")
    #clip_UNRATE = diff_UNRATE.clip(-2.0, 4.5).rename("UNRATE_diff_clip")

    #yoy_CES0500000003 = df_["CES0500000003"].pct_change(12).dropna().rename("yoy_CES0500000003")
    #clip_yoy_CES0500000003 = yoy_CES0500000003.clip(0.02, 0.06).rename("CES0500000003_yoy_clip")

    #df_featured = pd.concat([clip_PAYEMS,clip_PCE, clip_yoy_CES0500000003], axis=1).dropna(how="all")
    df_featured = pd.concat([yoy_PAYEMS,yoy_PCE], axis=1).dropna(how="all")
    #print(df_featured.corr())
    #check_nan_time(df_featured,"1990-01-01")

    #plot_index(df_featured.tail(10))
    return df_featured

def _featuring_c(df):
    col = ["SOFR","DFF","TB3MS","DX-Y.NYB","BAMLC0A4CBBB","BAMLC0A3CA"]
    df_ = df[col].dropna(how="all")

    # DFFで代用
    df_["SOFR"] = df_["SOFR"].fillna(df_["DFF"])
    #start = df_.apply(pd.Series.first_valid_index).max()
    #end   = df_.apply(pd.Series.last_valid_index).max() # 末尾はNanで残してカルマンフィルタを使う
    #df_ = df_.loc[start:end].drop(columns=["DFF"])
    #check_nan_time(df_,"1990-01-01")
    #print(df_.tail(10))

    #diff_SOFR = df_["SOFR"].diff(12).dropna().rename("diff_SOFR")
    #clip_SOFR = diff_SOFR.clip(-3, 3).rename("SOFR_diff_clip")

    yoy_DXY = df_["DX-Y.NYB"].pct_change(12).dropna().rename("yoy_DXY")
    #clip_yoy_DXY = cap_by_sigma(yoy_DXY, sigma=2.5).rename("DXY_yoy_clip")

    diff_DXY = df_["DX-Y.NYB"].diff().dropna().rename("diff_DXY")
    #clip_diff_DXY = cap_by_sigma(diff_DXY, sigma=2.5).rename("DXY_diff_clip")

    spd_SOFR_TB3MS = (df_["SOFR"] - df_["TB3MS"]).dropna().rename("spd_SOFR_TB3MS")
    #clip_spd_TB3MS = spd_TB3MS.clip(-0.5, 0.5).rename("SOFR_TB3MS_minus_clip")

    #diff_spd_SOFR_TB3MS = spd_SOFR_TB3MS.diff().dropna().rename("diff_spd_SOFR_TB3MS")
    #clip_diff_spd_TB3MS = diff_spd_TB3MS.clip(-0.25, 0.25).rename("SOFR_TB3MS_minus_diff_clip")

    spd_BBB_A = (df_["BAMLC0A4CBBB"] - df_["BAMLC0A3CA"]).dropna().rename("spd_BBB_A")
    #clip_spd_BBB_A = spd_BBB_A.clip(lower=0, upper=1.2).rename("BAMLC0A4CBBB_minus_BAMLC0A3CA_clip")

    diff_spd_BBB_A = spd_BBB_A.diff().dropna().rename("diff_spd_BBB_A")
    #clip_diff_spd_BBB_A = diff_spd_BBB_A.clip(lower=-0.35, upper=0.3).rename("BAMLC0A4CBBB_minus_BAMLC0A3CA_diff_clip")

    df_featured = pd.concat([
        yoy_DXY, spd_SOFR_TB3MS, spd_BBB_A, diff_spd_BBB_A
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

    #start = df_.apply(pd.Series.first_valid_index).max()
    #end   = df_.apply(pd.Series.last_valid_index).max() # 末尾はNanで残してカルマンフィルタを使う
    #df_ = df_.loc[start:end]
    #print(df_.tail(10))

    # Net Liquidity の生成
    Net_Liquidity = df_['WALCL'] - (df_['WDTGAL'] +  df_['RRP_filled'])
    # 特徴量1: 成長率 (YoY)
    yoy_Net_Liquidity = Net_Liquidity.pct_change(12).rename("yoy_Net_Liquidity")
    #clip_yoy_Net_Liquidity = yoy_Net_Liquidity.clip(-0.2, 0.5).rename("Net_Liquidity_yoy_clip")

    # 特徴量2: 吸収率 (TGA+RRPが資産に占める割合)
    #Abs_Rate = ((df_['WDTGAL'] + df_['RRP_filled']) / df_['WALCL']).rename("Abs_Rate")
    #clip_Abs_Rate = Abs_Rate.clip(0, 0.25).rename("Abs_Rate_clip")

    # 特徴量3: 銀行準備金の厚み
    Res_Ratio = (df_['Unified_Reserves'] / df_['WALCL']).rename("Res_Ratio")
    #clip_Res_Ratio = Res_Ratio.clip(0, 0.25).rename("Res_Ratio_clip")

    #df_featured = pd.concat([clip_yoy_Net_Liquidity,clip_Abs_Rate], axis=1).dropna(how="all")
    df_featured = pd.concat([yoy_Net_Liquidity,Res_Ratio], axis=1).dropna(how="all")

    #pd.set_option('display.max_rows', None)
    #print(df_["WDTGAL"])

    #print(df_featured.tail(10))
    #check_nan_time(df_featured,"1990-01-01")
    #plot_index(df_featured)
    #print(df_featured.corr())
    return df_featured

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
def _make_label_gli(df_gli):

    df = df_gli.copy()
    y_diff = df.diff()
    target = y_diff.shift(-1).dropna()
    #print(target.describe())
    # 統計データから算出した閾値
    lower_threshold = -1.099750  # 下位25% (Down)
    upper_threshold = 1.181250   # 上位25% (Up)
    labels = pd.cut(
        target,
        bins=[-np.inf, lower_threshold, upper_threshold, np.inf],
        labels=[1, 2, 3]
    ).rename("gli_label")
    #print(labels)
    return labels

    # 線形月次変換
    """gli_monthly = df_gli.resample('ME').first().interpolate(method='linear')

    # カンニングラベル
    y_diff = gli_monthly.diff(3)
    target = y_diff.shift(-3)

    # 統計データから算出した閾値
    lower_threshold = -0.911  # 下位25% (Down)
    upper_threshold = 1.073   # 上位25% (Up)

    labels = pd.cut(
        target,
        bins=[-np.inf, lower_threshold, upper_threshold, np.inf],
        labels=[1, 2, 3]
    ).rename("gli_label")
    #print(labels)

    return labels"""

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
