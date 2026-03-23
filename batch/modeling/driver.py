########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test,
    learning_lgbm_final,
    learning_get_shap_df,
    learning_get_shap_date
    )
from batch.modeling.visualize import (
    plot_driver_soft_label,
    plot_driver_diagnostic_report,
    plot_index
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################

def get_driver_beta(df_index, df_sp500):

    df_daily = df_index.copy()

    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df_daily)

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_daily, df_sp500)

    features_refined = [
        'VIX_z252',
        'VVIX_z252',
        'MOVE_z252',
        #'VIX_diff5_zscore',
        #'MOVE_diff5_zscore',
        #'MOVE_VIX_ratio_zscore',###
        #'VIX_gap_zscore',
        #'VIX_rv_zscore',
        "MOVE_vov",###########
        #"MOVE_accel",
        
        #'HY_diff5_zscore',
        'hy_z252',
        #'TED_spread_z252',
        #'TED_diff5_zscore',
        'SOFR_vol_spike',
        'Term_Premium_z252',
        'Credit_Equity_Divergence',
        #'Term_Premium_diff5',
        "Term_Premium_diff5_z252",

        'DFII10_diff5_zscore',
        #'DFII10_z252',
        #'Curve_10Y2Y_z252', #1
        #'Curve_10Y3M_z252',
        #'T10YIE_diff5_zscore',
        #'Real_Nominal_ratio_zscore',
        #'Curve_flattening_speed_zscore',

        #'DXY_diff5_zscore',
        #'DXY_z252',
        #'Stock_Bond_Corr_20d',
        'Stock_Bond_Corr_zscore',
        #"Gold_zscore",
        'Equity_Gold_Ratio_zscore',
        'Flight_to_Safety_zscore',
        #'SP500_Ret_Z',
        #"tlt_z252",
        #"tlt_ret_z252",
        #"tlt_diff_z252",
        "tlt_hy_ratio_z252",
        #"tlt_hy_diff_z252",
        #"oil_ret_z252"
        ]
    df_features = df_features[features_refined]

    # --- 学習モデル生成 ---
    # サンプルフェイト
    df_label['sample_weight'] = 1.0

    mask_credit = (df_label['driver'] == 1)
    df_label.loc[mask_credit, 'sample_weight'] = df_label['credit_score']
    mask_bond = (df_label['driver'] == 2)
    df_label.loc[mask_bond, 'sample_weight'] = df_label['bond_score']
    mask_bond = (df_label['driver'] == 3)
    df_label.loc[mask_bond, 'sample_weight'] = 0.8

    df_driver = df_features.join(df_label["driver"])

    """driver_clf, df_driver_trajectory = learning_lgbm_final(
        df_driver, "driver", model_name="Driver", label_name_list=["1:Credit", "2:Bond", "3:Mix"],
        n_estimators=2800,learning_rate=0.001,num_leaves=50, min_data_in_leaf=100,
        reg_alpha=0.3, reg_lambda=0.3,)"""

    print(f"特徴量のリスト: {df_features.columns}")
    """df_oof_all, df_shap, df_oof_ev = learning_lgbm_test(
        df_driver, "driver", labels=["1:Credit", "2:Bond", "3:Mix"],
        n_splits=5, gap =30,
        n_estimators=2800,learning_rate=0.001,num_leaves=50, min_data_in_leaf=100,
        class_weight="balanced",
        sample_weight=df_label["sample_weight"],
        reg_alpha=0.3, reg_lambda=0.3, learning_curve=False,
        )"""

     # --- ファイル保存 ---
    """df_oof_all.to_parquet("diver_oof.parquet", engine="pyarrow")
    df_shap["1:Credit"].to_parquet("diver_shap_credit.parquet", engine="pyarrow")
    df_shap["2:Bond"].to_parquet("diver_shap_bond.parquet", engine="pyarrow")
    df_shap["3:Mix"].to_parquet("diver_shap_mix.parquet", engine="pyarrow")
    df_oof_ev.to_parquet("diver_oof_ev.parquet", engine="pyarrow")"""

    # --- ファイル読み込み ---
    df_oof_all = pd.read_parquet("diver_oof.parquet", engine="pyarrow")
    df_shap = {
        "1:Credit":pd.read_parquet("diver_shap_credit.parquet", engine="pyarrow"),
        "2:Bond":pd.read_parquet("diver_shap_bond.parquet", engine="pyarrow"),
        "3:Mix":pd.read_parquet("diver_shap_mix.parquet", engine="pyarrow")
    }
    df_oof_ev = pd.read_parquet("diver_oof_ev.parquet", engine="pyarrow")

    # --- 一時分析 ---
    df_driver = df_features.join(df_label["next_20d_ret_sp500"])

    df_bt = df_oof_ev["ev_rank"].to_frame().join(df_daily["^GSPC"].pct_change().ffill().rename("sp500_ret"))
    df_oof = df_oof_all[["1:Credit", "2:Bond", "3:Mix"]].join(df_oof_ev[["risk_sum", "expected_value"]])
    plot_driver_diagnostic_report(df_bt, df_oof, start_date="2025-01-01", end_date="2026-02-01")
    #_da_miss_credit_mix(df_oof_all,df_shap,df_driver)

    #_da_CRITICAL(df_oof_all, df_oof_ev, df_shap)
    #_da_CRITICAL_detail(df_oof_all, df_oof_ev, df_shap, df_driver)
    #_da_High_Risk(df_oof_all, df_oof_ev, df_shap, df_driver)

    # --- フィルター ---
    #df_oof_ev, df_oof_all = _divergence_filter(df_oof_ev, df_driver, df_oof_all)
    #df_oof_ev, df_oof_all = _high_risk_reselection_filter(df_oof_ev, df_driver, df_oof_all)

    # --- 結果を確認 ---
    #_chk_ev_hist(df_oof_ev)
    #_chk_accuracy(df_oof_all)
    #_chk_ev(df_oof_ev)

    # --- バックテスト ---
    #_back_test(df_daily, df_oof_ev)
    """_ = plot_driver_trajectory(
        df_prob, df_daily["^GSPC"].pct_change().dropna(),
        ["1:Credit", "2:Bond", "3:Mix"],
        start_date="2010-01-01", end_date="2026-01-01"
        )"""

    #return driver_clf, df_driver_trajectory, df_driver
    #return df_oof_all

########################################################
# 特徴量抽出
########################################################

def _featuring_all(df_daily, df_sp500):

    # --- リバランスの実行基準となるマスターカレンダー ---
    master_index = df_daily["^GSPC"].dropna().index
    feats = pd.DataFrame(index=master_index)

    # ---  恐怖の先行指標 - 初期震動の検知 ---
    feats = _vol_feats(df_daily, feats, master_index)

    # --- システムの目詰まり（Credit & Liquidity）- Creditレジュームを仕留める ---
    feats = _credit_liq_feats(df_daily, feats, master_index)

    # --- マクロの重力（Rates & Inflation）- Bondレジュームを仕留める ---
    feats = _macro_gravity_feats(df_daily, feats, master_index)

    # --- 資金のうねり（Momentum & Flow）- Creditレジュームを仕留める ---
    feats = _momentum_flow_feats(df_daily, feats, master_index)

    # 未来リーク耐性
    feats = feats.shift(1).dropna(how="all")
    #check_nan_time(feats, date="2005-01-01")

    # 開始日、終了日をを決める
    start = "2005-03-16"#feats["MOVE_z252"].first_valid_index()
    end = feats.apply(pd.Series.last_valid_index).min()
    feats = feats.loc[start:end]
    
    #check_nan_time(feats, date="2005-01-01")

    return feats

    #df_b["vix_panic_duration"] = ((vix > 20).astype(int)).rolling(20).sum().reindex(master_index)

def _vol_feats(df, feats, master_index):
    # 指標
    vix = df['VIXCLS'].dropna()
    vvix = df["VVIX"].dropna()
    move = df["^MOVE"].dropna()

    # 異常性
    feats["VIX_z252"] = _featuring_z_score(vix, window=252).reindex(master_index, method="ffill")
    feats["VVIX_z252"] = _featuring_z_score(vvix, window=252).reindex(master_index, method="ffill")
    feats["MOVE_z252"] = _featuring_z_score(move, window=252).reindex(master_index, method="ffill")

    # 加速
    vix_diff = vix.diff(5)
    move_diff = move.diff(5)
    feats['VIX_diff5_zscore'] = _featuring_z_score(vix_diff, window=252).reindex(master_index, method="ffill")
    feats['MOVE_diff5_zscore'] = _featuring_z_score(move_diff, window=252).reindex(master_index, method="ffill")
    feats['MOVE_accel'] = feats['MOVE_z252'].diff(1).diff(5).reindex(master_index, method="ffill")

    # 格差
    ratio = move / vix
    ratio = ratio.ffill()
    #feats['MOVE_VIX_ratio'] = ratio.reindex(master_index, method="ffill")
    feats['MOVE_VIX_ratio_zscore'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")

    # VIXの期間構造(VIX3Mの代用)
    vix_gap = vix / vix.rolling(window=60).mean()
    feats['VIX_gap_zscore'] = _featuring_z_score(vix_gap, window=252).reindex(master_index, method="ffill")

    # ボラのボラ
    vix_rv = vix.pct_change().rolling(20).std()
    feats['VIX_rv_zscore'] = _featuring_z_score(vix_rv, window=252).reindex(master_index, method="ffill")
    move_z252 = _featuring_z_score(move, window=252).reindex(master_index, method="ffill")
    feats['MOVE_vov'] = move_z252.diff().rolling(20).std().reindex(master_index, method="ffill")
    return feats

def _credit_liq_feats(df, feats, master_index):
    # 指標
    hy = df['BAMLH0A0HYM2'].dropna()
    sofr = df["SOFR"].dropna()
    tedrate = df["TEDRATE"].dropna()
    dgs10 = df["DGS10"].dropna()
    dgs3mo = df["DGS3MO"].dropna()
    vix = df["VIXCLS"].dropna()

    # クレジットの加速
    hy_diff = hy.diff(5)
    feats['HY_diff5_zscore'] = _featuring_z_score(hy_diff, window=252).reindex(master_index, method="ffill")
    feats["hy_z252"] = _featuring_z_score(hy, window=252).reindex(master_index, method="ffill")

    # 歴史的パニックの同期 (2008年対策)
    #feats['TED_spread_level'] = tedrate.reindex(master_index, method="ffill")
    #feats['TED_spread_diff5'] = tedrate.diff(5).reindex(master_index, method="ffill")
    feats['TED_spread_z252'] = _featuring_z_score(tedrate, window=252).reindex(master_index, method="ffill")
    feats['TED_diff5_zscore'] = _featuring_z_score(tedrate.diff(5), window=252).reindex(master_index, method="ffill")

    # 現代の流動性ショック
    sofr_rolling_mean = sofr.rolling(20).mean()
    sofr_rolling_std = sofr.rolling(20).std().replace(0, np.nan)
    feats['SOFR_vol_spike'] = ((sofr - sofr_rolling_mean) / sofr_rolling_std).reindex(master_index, method="ffill")

    # 金融機関の「収益性・貸出意欲」の悪化
    term_premium = dgs10 - dgs3mo
    feats['Term_Premium_z252'] = _featuring_z_score(term_premium, window=252).reindex(master_index, method="ffill")
    feats['Term_Premium_diff5'] = feats['Term_Premium_z252'].diff(5).reindex(master_index, method="ffill")
    feats['Term_Premium_diff5_z252'] = _featuring_z_score(feats['Term_Premium_diff5'] , window=252).reindex(master_index, method="ffill")

    # クレジットとボラティリティの「乖離」
    vix_z = _featuring_z_score(vix, window=252)
    feats['Credit_Equity_Divergence'] = (feats['hy_z252'] - vix_z).reindex(master_index, method="ffill")

    return feats

def _macro_gravity_feats(df, feats, master_index):
    # 指標
    dfii10 = df["DFII10"].dropna()
    t10yie = df["T10YIE"].dropna()
    dgs10 = df["DGS10"].dropna()
    dgs2 = df["DGS2"].dropna()
    dgs3mo = df["DGS3MO"].dropna()

    # 実質金利のモメンタム
    dfii10_diff = dfii10.diff(5)
    feats['DFII10_diff5_zscore'] = _featuring_z_score(dfii10_diff, window=252).reindex(master_index, method="ffill")
    feats['DFII10_z252'] = _featuring_z_score(dfii10, window=252).reindex(master_index, method="ffill")

    # イールドカーブ
    curve10y2y = dgs10 - dgs2
    curve10y3m = dgs10 - dgs3mo
    feats['Curve_10Y2Y_z252'] = _featuring_z_score(curve10y2y, window=252).reindex(master_index, method="ffill")
    feats['Curve_10Y3M_z252'] = _featuring_z_score(curve10y3m, window=252).reindex(master_index, method="ffill")

    # インフレ期待の加速
    t10yie_diff = t10yie.diff(5)
    feats['T10YIE_diff5_zscore'] = _featuring_z_score(t10yie_diff, window=252).reindex(master_index, method="ffill")

    # 金利上昇の「質」の分解
    real_nominal_ratio = dfii10 / dgs10
    feats['Real_Nominal_ratio_zscore'] = _featuring_z_score(real_nominal_ratio, window=252).reindex(master_index, method="ffill")

    # カーブの「フラット化」速度
    flattening_speed = curve10y2y.diff(20)
    feats['Curve_flattening_speed_zscore'] = _featuring_z_score(flattening_speed, window=252).reindex(master_index, method="ffill")

    return feats

def _momentum_flow_feats(df, feats, master_index):
    # 指標
    dxy = df["DX-Y.NYB"].dropna()
    gold = df["GC=F"].dropna()
    sp500 = df["^GSPC"].dropna()
    tlt = df["TLT"].dropna()
    hy = df["BAMLH0A0HYM2"].dropna()
    oil = df["CL=F"].dropna()

    # ドルの引力
    dxy_diff = dxy.diff(5)
    feats['DXY_diff5_zscore'] = _featuring_z_score(dxy_diff, window=252).reindex(master_index, method="ffill")
    feats['DXY_z252'] = _featuring_z_score(dxy, window=252).reindex(master_index, method="ffill")

    # アセット相関の変調
    returns_sp = sp500.pct_change()
    returns_tlt = tlt.pct_change()
    corr = returns_sp.rolling(20).corr(returns_tlt)
    feats['Stock_Bond_Corr_20d'] = corr.reindex(master_index, method="ffill")
    feats['Stock_Bond_Corr_zscore'] = _featuring_z_score(corr, window=252).reindex(master_index, method="ffill")


    # リスクオン・オフの体温計
    feats['Gold_zscore'] = _featuring_z_score(gold, window=252).reindex(master_index, method="ffill")
    equity_gold = sp500 / gold
    feats['Equity_Gold_Ratio_zscore'] = _featuring_z_score(equity_gold, window=252).reindex(master_index, method="ffill")

    # 資金の逃避速度
    fts_index = returns_tlt - returns_sp
    feats['Flight_to_Safety_zscore'] = _featuring_z_score(fts_index, window=252).reindex(master_index, method="ffill")

    # 市場のオーバーシュート
    feats['SP500_Ret_Z'] = _featuring_z_score(returns_sp.rolling(20).sum(), window=252).reindex(master_index, method="ffill")

    feats['tlt_z252'] = _featuring_z_score(tlt, window=252).reindex(master_index, method="ffill")
    feats['tlt_ret_z252'] = _featuring_z_score(tlt.pct_change(), window=252).reindex(master_index, method="ffill")
    feats['tlt_diff_z252'] = _featuring_z_score(tlt.diff(5), window=252).reindex(master_index, method="ffill")
    ratio = tlt / hy
    ratio = ratio.ffill()
    feats['tlt_hy_ratio_z252'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")
    tlt_hy_diff = np.log(tlt)- np.log(hy)
    feats['tlt_hy_diff_z252'] = _featuring_z_score(tlt_hy_diff, window=252).reindex(master_index, method="ffill")

    feats['oil_ret_z252'] = _featuring_z_score(oil.pct_change(), window=252).reindex(master_index, method="ffill")


    return feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# 教師ラベル作成 - カンニングラベル
########################################################

def _make_label(df_daily, smear_days=5):

    # リバランスの実行基準となるマスターカレンダー
    master_index = df_daily["^GSPC"].dropna().index
    df = pd.DataFrame(index=master_index)

    # 指標の生成
    tlt_ret = df_daily["TLT"].pct_change()
    tlt_vol = tlt_ret.rolling(60,min_periods=20).std().reindex(master_index, method='ffill')

    sp500_ret = df_daily['^GSPC'].pct_change()
    sp500_vol = sp500_ret.rolling(60,min_periods=20).std().reindex(master_index, method='ffill')

    hy_diff = df_daily["BAMLH0A0HYM2"].diff()
    hy_diff_vol = hy_diff.rolling(60,min_periods=20).std().reindex(master_index, method='ffill')

    for col, asset_name in zip(["^GSPC", "TLT"], ["sp500", "tlt"]):
        # NAを落として、その資産の純粋な営業日だけで未来20日を計算
        asset_clean = df_daily[col].dropna()
        future_ret = asset_clean.pct_change(20).shift(-20)
        df[f'next_20d_ret_{asset_name}'] = future_ret.reindex(master_index, method="ffill")

    # HYスプレッドは「差分(diff)」で計算する
    hy_clean = df_daily["BAMLH0A0HYM2"].dropna()
    future_diff = hy_clean.diff(20).shift(-20) # 20日後に何ポイント拡大したか
    df['next_20d_diff_hy'] = future_diff.reindex(master_index, method="ffill")

    # --- Step 1: 生のフラグ（Raw Flags）を立てる ---
    # Credit: HYスプレッドの拡大（※絶対値ではなく拡大側にバイアスをかけるのが戦略的）
    raw_credit = (
        (df['next_20d_diff_hy'] > (2.0 * hy_diff_vol * np.sqrt(20))) & # 拡大のみ
        ((df['next_20d_diff_hy'] / hy_diff_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5))
    )

    # Bond: TLTの激しい動き
    raw_bond = (
        (df['next_20d_ret_tlt'].abs() > (1.5 * tlt_vol * np.sqrt(20))) &
        ((df['next_20d_ret_tlt'].abs() / tlt_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5))
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
                    # 指数減衰：イベント当日(d=0)が1.0、遡るほど小さくなる
                    decay_val = np.exp(-d / tau)
                    scores[idx - d] = max(scores[idx - d], decay_val)
        return scores

    # スコア（確信度 0.0 ~ 1.0）を算出
    df['credit_score'] = calculate_decay_score(raw_credit, smear_days)
    df['bond_score'] = calculate_decay_score(raw_bond, smear_days)

    # --- Step 3: スコアに基づく動的ラベル付与 ---
    # 単なる0/1ではなく、閾値（例: 0.5）を超えた期間をレジュームとして認定
    # これにより「あまりに遠い予兆」を無理に学習することを防ぐ
    threshold = 0.4

    df["driver"] = 3 # Neutral

    # 1. まず、閾値を超えている場所を特定
    is_bond_candidate = df['bond_score'] > threshold
    is_credit_candidate = df['credit_score'] > threshold
    
    #sp500 = df_daily["^GSPC"].dropna()
    #vix = df_daily["VIXCLS"].dropna().reindex(master_index, method='ffill')
    #sp500_sma = sp500.rolling(25).mean().reindex(master_index, method='ffill')
    #is_market_calm = (sp500 > sp500_sma) & (vix < 20)

    #raw_credit_filtered = is_credit_candidate & (~is_market_calm)

    # 基本はスコアが高い方を採用（勝者総取り）
    df.loc[is_bond_candidate, 'driver'] = 2
    df.loc[is_credit_candidate & (df['credit_score'] >= df['bond_score']), 'driver'] = 1
    df.loc[is_bond_candidate & (df['bond_score'] > df['credit_score']), 'driver'] = 2

    """
    df["driver"] = 3 # Neutral
    df["driver_name"] = "Neutral"

    # 優先度低
    significant_bond_move = df['next_20d_ret_tlt'].abs() > (1.35 * tlt_vol * np.sqrt(20))
    bond_dominance = (df['next_20d_ret_tlt'].abs() / tlt_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5)
    is_bond_move = significant_bond_move & bond_dominance
    df.loc[is_bond_move, 'driver'] = 2
    df.loc[is_bond_move, 'driver_name'] = "Bond"

    # 優先度高
    significant_credit_move = df['next_20d_diff_hy'].abs() > (2.0 * hy_diff_vol * np.sqrt(20))
    credit_dominance = (df['next_20d_diff_hy'].abs() / hy_diff_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5)

    is_credit_move = significant_credit_move & credit_dominance
    df.loc[is_credit_move, 'driver'] = 1
    df.loc[is_credit_move, 'driver_name'] = "Credit"

    """
    df = df.dropna()
    #check_nan_time(df)

    print("\nDriver教師ラベルの期間: ",df.index[0].date(), df.index[-1].date())

    #_analysis_label(df, df_daily)

    return df

def _analysis_label(df, df_daily):
    # 分析・可視化
    stats = df['driver'].value_counts().to_frame(name='Count')
    stats['Percentage (%)'] = (df['driver'].value_counts(normalize=True) * 100).round(2)
    print(stats)

    market_summary = df.groupby('driver').agg({
        'next_20d_ret_sp500': ['mean', 'std', 'min', 'max'],
        'next_20d_ret_tlt': ['mean', 'std'],
        'next_20d_diff_hy': ['mean']
    }).round(4)
    print(market_summary)

    # 継続日数の算出
    df['change'] = df['driver'] != df['driver'].shift()
    df['regime_id'] = df['change'].cumsum()

    # 各期間の長さをカウント
    duration_stats = df.groupby(['regime_id', 'driver']).size().reset_index(name='duration')
    avg_duration = duration_stats.groupby('driver')['duration'].mean().round(1)
    print(f"平均継続日数:\n{avg_duration}")

    # 遷移マトリクス（現在の状態 -> 次の状態）
    transition_matrix = pd.crosstab(
        df['driver'], 
        df['driver'].shift(-1), 
        normalize='index'
    ).round(2)

    print("遷移マトリクス（行：現在 -> 列：次）:")
    print(transition_matrix)
    plot_driver_soft_label(df, df_daily, start_date="2024-01-01", end_date="2026-02-01")

########################################################
# フィルタ -　AI 解釈の微修正
########################################################
def _divergence_filter(df_oof_ev, df_driver, df_oof_all):

    # evラベルの修正
    df_oof_ev = df_oof_ev.join(df_driver[[
        'Credit_Equity_Divergence',
        'Equity_Gold_Ratio_zscore',
        'VVIX_z252'
        ]], how='left')

    df_oof_ev['ev_rank'] = df_oof_ev['ev_rank'].astype(str)
    #mask_rescue = (
    #    (df_oof_ev['ev_rank'] == 'CRITICAL') &
    #    (df_oof_ev['Credit_Equity_Divergence'] <= 1.0)
    #)
    mask_rescue = (
        (df_oof_ev['ev_rank'] == 'CRITICAL') & (
            (df_oof_ev['Equity_Gold_Ratio_zscore'] <= 0.5) | # 株が金より明らかに強い（救済）
            (df_oof_ev['VVIX_z252'] >= 0.2)                  # オプション市場が冷静（救済）
        )
    )

    df_oof_ev.loc[mask_rescue, 'ev_rank'] = "Neutral"
    df_oof_ev = df_oof_ev.drop(columns=[
        'Credit_Equity_Divergence', 'Equity_Gold_Ratio_zscore', 'VVIX_z252'])

    # 確率の修正
    df_oof_all.loc[mask_rescue, '1:Credit'] = 0.25  # Creditリスクを中程度に抑制
    df_oof_all.loc[mask_rescue, '2:Bond'] = 0.25  # Bondリスクを中程度に抑制
    df_oof_all.loc[mask_rescue, '3:Mix'] = 0.50  # Neutral確率を50%確保
    df_oof_ev.loc[mask_rescue, 'risk_sum'] = 0.50

    return df_oof_ev, df_oof_all

def _high_risk_reselection_filter(df_oof_ev, df_driver, df_oof_all):
    
    df_oof_ev = df_oof_ev.join(df_driver[[
    'Term_Premium_z252',
    'Credit_Equity_Divergence',
    'Equity_Gold_Ratio_zscore'
    ]], how='left')
    
    df_oof_ev['ev_rank'] = df_oof_ev['ev_rank'].astype(str)
    
    # High Risk のなかから「お宝（Win）」を特定する条件
    # 1. 期間プレミアムが健全（崩壊していない）
    # 2. 株がクレジットを無視して暴走していない（同期している）
    # 3. 株/金比率に一定の生命力が残っている
    
    mask = (
        (df_oof_ev['ev_rank'] == 'High Risk') & 
        (df_oof_ev['Term_Premium_z252'] > -0.2) &        # 金利構造の健全性
        (df_oof_ev['Credit_Equity_Divergence'] < 0.1) &   # 株の強欲さがない
        (df_oof_ev['Equity_Gold_Ratio_zscore'] > 0.8)     # 相対的な強さ
    )
    
    # 救済：これらを Neutral (市場参加) に格下げ
    df_oof_ev.loc[mask, 'ev_rank'] = "Neutral"
    df_oof_ev = df_oof_ev.drop(columns=[
        'Term_Premium_z252', 'Credit_Equity_Divergence', 'Equity_Gold_Ratio_zscore'])
    
    # 確率の修正
    df_oof_all.loc[mask, '1:Credit'] = 0.25  # Creditリスクを中程度に抑制
    df_oof_all.loc[mask, '2:Bond'] = 0.25  # Bondリスクを中程度に抑制
    df_oof_all.loc[mask, '3:Mix'] = 0.50  # Neutral確率を50%確保
    df_oof_ev.loc[mask, 'risk_sum'] = 0.50
    print(f"High Risk から救済されたお宝: {mask.sum()} 日")
    
    return df_oof_ev, df_oof_all

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

########################################################
# モデル分析・検証
########################################################

def _da_CRITICAL(df_oof_all, df_oof_ev, df_shap, df_driver):
    critical_df = df_oof_ev[df_oof_ev['ev_rank'] == 'CRITICAL'].copy()
    critical_df = critical_df.join(df_oof_all[["1:Credit","2:Bond","3:Mix"]])
    #print(critical_df)

    rebound_df = critical_df[critical_df['actual_return'] > 0]
    hit_df = critical_df[critical_df['actual_return'] <= 0]

    print(f"=== CRITICAL 58日間 の内訳分析 ===")
    print(f"リバウンド日: {len(rebound_df)}日 / 的中日: {len(hit_df)}日")
    print("-" * 40)

    for name, target in [("リバウンド", rebound_df), ("的中", hit_df)]:
        if not target.empty:
            # 予測確率のカラム名はモデルの labels 指定に基づきます
            # 一般的には 'prob_1.0', 'prob_2.0' 等
            p_credit = target['1:Credit'].mean()
            p_bond   = target['2:Bond'].mean()
            print(f"【{name}】の平均確率 -> Credit: {p_credit:.3f} / Bond: {p_bond:.3f}")

    # 2. どちらのレジュームが「ノイズ」を呼んでいるか？
    if rebound_df['2:Bond'].mean() > rebound_df['1:Credit'].mean():
        print("\n[診断] リバウンドは主に『Bond（金利急変）』由来で発生しています。")
    else:
        print("\n[診断] リバウンドは主に『Credit（スプレッド拡大）』由来で発生しています。")
    # リバウンド日 (rebound_df) のインデックスを使って SHAP値を抽出
    rebound_indices = rebound_df.index
    rebound_shap = df_shap["1:Credit"].loc[rebound_indices]

    print("\n=== リバウンドを誘発した特徴量 Top 5 (SHAP平均) ===")
    # 各特徴量の絶対値平均（または実数平均）を算出し、リスクを押し上げた犯人を探す
    # ※クラスごとのSHAPがある場合は、Credit/BondそれぞれのSHAPを見てください
    print(rebound_shap.mean().sort_values(ascending=False).head(5))

def _da_CRITICAL_detail(df_oof_all, df_oof_ev, df_shap, df_driver):
    df_oof_ev = df_oof_ev.join(df_driver, how='left')
    mask_target = (df_oof_ev['ev_rank'] == 'CRITICAL') & (df_oof_ev['Credit_Equity_Divergence'] <= 1.0)
    mask_rebound = (df_oof_ev['ev_rank'] == 'CRITICAL') & (df_oof_ev['Credit_Equity_Divergence'] > 1.0)
    features = [
        'VIX_z252', 'VVIX_z252', 'MOVE_z252', 'MOVE_vov', 'hy_z252',
        'SOFR_vol_spike', 'Term_Premium_z252', 'Credit_Equity_Divergence',
        'Term_Premium_diff5_z252', 'DFII10_diff5_zscore',
        'Stock_Bond_Corr_zscore', 'Equity_Gold_Ratio_zscore',
        'Flight_to_Safety_zscore', 'tlt_hy_ratio_z252'
        ]
    stats_target = df_oof_ev.loc[mask_target, features].describe().T
    stats_rebound = df_oof_ev.loc[mask_rebound, features].describe().T
    
    comparison = stats_target[['mean', '50%']].join(
        stats_rebound[['mean', '50%']], 
        lsuffix='_Target(21d)', 
        rsuffix='_Rebound(37d)'
    )
    comparison['diff_mean'] = comparison['mean_Target(21d)'] - comparison['mean_Rebound(37d)']
    print("=== 的中(21日) vs リバウンド(37日) 特徴量比較レポート ===")
    print(comparison.sort_values('diff_mean', key=abs, ascending=False))

def _da_High_Risk(df_oof_all, df_oof_ev, df_shap, df_driver):
    df_oof_ev = df_oof_ev.join(df_driver, how='left')
    # 1. High Risk群（382日）を抽出
    df_hr = df_oof_ev[df_oof_ev['ev_rank'] == 'High Risk'].copy()

    # 2. 20日後のリターン（教師ラベルの元データ）に基づいて「お宝」と「ゴミ」に分割
    # ※ next_20d_ret_sp500 などのリターンカラムを使用
    mask_otakara = df_hr['next_20d_ret_sp500'] > 0
    mask_gomi    = df_hr['next_20d_ret_sp500'] <= 0

    # 3. 比較する特徴量の厳選セット
    features_to_analyze = [
        'VIX_z252', 'VVIX_z252', 'MOVE_z252', 'MOVE_vov', 'hy_z252',
        'SOFR_vol_spike', 'Term_Premium_z252', 'Credit_Equity_Divergence',
        'Term_Premium_diff5_z252', 'DFII10_diff5_zscore',
        'Stock_Bond_Corr_zscore', 'Equity_Gold_Ratio_zscore',
        'Flight_to_Safety_zscore', 'tlt_hy_ratio_z252'
        ]

    # 4. 統計量の算出
    stats_otakara = df_hr.loc[mask_otakara, features_to_analyze].describe().T
    stats_gomi    = df_hr.loc[mask_gomi, features_to_analyze].describe().T

    # 5. 比較レポートの作成
    hr_analysis = pd.DataFrame({
        'mean_Otakara(Win)': stats_otakara['mean'],
        'mean_Gomi(Loss)': stats_gomi['mean'],
        'median_Otakara': stats_otakara['50%'],
        'median_Gomi': stats_gomi['50%']
    })

    # 差分（お宝 - ゴミ）を計算。この値が大きいほど、その指標は「買い時」を見分ける武器になる
    hr_analysis['diff_mean'] = hr_analysis['mean_Otakara(Win)'] - hr_analysis['mean_Gomi(Loss)']

    print(f"=== High Riskバケツ（n=382）内部解剖レポート ===")
    print(f"お宝(リターン正): {mask_otakara.sum()} 日")
    print(f"ゴミ(リターン負): {mask_gomi.sum()} 日")
    print("-" * 50)
    print(hr_analysis.sort_values('diff_mean', key=abs, ascending=False))

########################################################
# 結果・確認
########################################################
def _chk_ev_hist(df_oof_ev):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. 基本統計量の確認（平均、最小、最大、四分位数）
    print("=== risk_sum の基本統計量 ===")
    print(df_oof_ev['risk_sum'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df_oof_ev['risk_sum'], bins=50, kde=True, color='royalblue')
    plt.title('Distribution of Risk Sum (Credit + Bond Probability)')
    plt.xlabel('Risk Sum Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.show()

def _chk_accuracy(df_oof_all):
    df_prob = df_oof_all[["1:Credit", "2:Bond", "3:Mix"]]
    df_prob['dominant_regime'] = df_prob.idxmax(axis=1).str.split(':').str[0].astype(int)
    teacher = df_oof_all["actual_regime"]
    ai = df_prob["dominant_regime"]
    teacher, ai = teacher.align(ai, join="inner")

    from sklearn.metrics import classification_report,confusion_matrix
    print("\nAccuracyの結果レポート")
    print(classification_report(teacher, ai))
    print(confusion_matrix(teacher, ai))

def _chk_ev(df_oof_ev):
    ev_summary = df_oof_ev.groupby('ev_rank', observed=True)['actual_return'].agg(['mean',"median", 'count'])
    print("\n期待値の結果レポート")
    print(ev_summary.reindex(['Safe', 'Neutral', 'Caution', 'High Risk', 'CRITICAL']))

########################################################
# バックテスト
########################################################
def _back_test(df_daily, df_oof_ev):
    import matplotlib.pyplot as plt
    # 1. データ準備とラグの処理
    sp500_ret = df_daily["^GSPC"].pct_change().dropna().rename("sp500_ret")
    df_bt = df_oof_ev.join(sp500_ret, how='left').sort_index().copy()

    # --- 重要：シグナルの1日ラグ（Execution Lag） ---
    lag = 10

    # 前日の終値で出た判定（ev_rank）を、今日のリターン（sp500_ret）に適用する
    df_bt['signal'] = df_bt['ev_rank'].shift(lag)

    # 2. 戦略リターンの計算
    danger_ranks = ['CRITICAL', 'High Risk'] # まずはCRITICALのみ
    
    # 昨日のシグナルがCRITICALなら、今日のリターンは0（現金退避）
    df_bt['return_strategy'] = np.where(
        df_bt['signal'].isin(danger_ranks),
        0.0,
        df_bt['sp500_ret']
    )
    # イメージ：100%避難せず、リスクの高さに応じて「アクセルを緩める」
    # df_bt['return_strategy'] = (1 - df_bt['signal_prob']) * df_bt['sp500_ret']

    # 3. 累積リターンの計算
    df_bt['cum_benchmark'] = (1 + df_bt['sp500_ret'].fillna(0)).cumprod()
    df_bt['cum_strategy']  = (1 + df_bt['return_strategy'].fillna(0)).cumprod()

    # 4. パフォーマンス・メトリクスの算出（ドローダウンの確認）
    def calc_mdd(cum_series):
        return (cum_series / cum_series.cummax() - 1).min()

    mdd_bench = calc_mdd(df_bt['cum_benchmark'])
    mdd_strat = calc_mdd(df_bt['cum_strategy'])

    print(f"=== 現実的バックテスト結果 ({lag}日ラグ) ===")
    print(f"Buy & Hold    最終: {df_bt['cum_benchmark'].iloc[-1]:.2f}倍 / MDD: {mdd_bench:.1%}")
    print(f"RTM Strategy  最終: {df_bt['cum_strategy'].iloc[-1]:.2f}倍 / MDD: {mdd_strat:.1%}")

    # 5. 可視化
    plt.figure(figsize=(12, 6))
    plt.plot(df_bt.index, df_bt['cum_benchmark'], label='Buy & Hold', color='gray', alpha=0.5)
    plt.plot(df_bt.index, df_bt['cum_strategy'], label='RTM Strategy (Shield)', color='red', linewidth=2)
    plt.title('Realistic Backtest: Signal Lag 1-day')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# ------------------------------------------------------
def _da_miss_credit_mix(df_oof_all,df_shap,df_driver):
    periods = [
        ("2013-05-24","2013-09-17"),("2013-10-02","2013-10-17"),
        ("2013-12-13","2013-12-18"),("2014-01-24","2014-02-10"),
        ("2015-03-04","2015-03-18"),("2017-01-04","2017-01-11"),
        ("2017-01-30","2017-02-13"),("2017-09-27","2017-10-02"),
    ]
    for start, end in periods:
        label="1:Credit"
        shap = df_shap[label].loc[start:end]
        top10_shap = shap.mean().sort_values(ascending=False).head(5)
        VIX_z252 = df_driver.loc[start:end, "VIX_z252"].mean()

        print(f"\nラベル{label}の{start}～{end}の平均確率と平均寄与度トップ5、およびVIX_z252の平均値")
        print(df_oof_all.loc[start:end].mean().round(2))
        print(f"\n{top10_shap}")
        print(f"\nVIX_z252の平均値: {VIX_z252}")

def _apply_probability_refinement(df_prob, df_shap, df_features):

    refined_prob = df_prob.copy()

    # 修正関数のリスト（今後、新しい修正モードが増えたらここに追加）
    refiners = [
        _refine_mode_interest_afterglow, # モードA: 金利の残像
        #_refine_mode_bond_vol_noise      # モードB: 債券ボラの過剰反応
    ]

    for refiner in refiners:
        refined_prob = refiner(refined_prob, df_shap, df_features)

    return refined_prob

def _refine_mode_interest_afterglow(df_prob, df_shap, df_features):

    VIX_SAFE_LIMIT = -0.5
    TP_SHAP_THRESHOLD = 0.35

    shap = df_shap["1:Credit"]

    # 1. 必要な中間変数を一括計算
    total_shap = shap.abs().sum(axis=1)
    # ゼロ除算を避けるために一応対策
    tp_ratio = shap['Term_Premium_z252'] / total_shap.replace(0, np.inf)
    vix_z = df_features['VIX_z252']

    # 2. 条件に合致するインデックスを特定する「マスク」を作成
    mask = (
        (df_prob["1:Credit"] >= 0.3) &
        (tp_ratio > TP_SHAP_THRESHOLD) &
        (vix_z < VIX_SAFE_LIMIT)
    )

    # 3. マスクされた行に対してのみ、ペナルティと移動量を一括計算
    if mask.any():
        # ペナルティ係数を計算 (0.0〜1.0の範囲にクリップ)
        penalty = ((vix_z[mask] - VIX_SAFE_LIMIT).abs() * tp_ratio[mask] * 2.0).clip(upper=1.0)

        # 移動量（デルタ）を算出
        delta = df_prob.loc[mask, '1:Credit'] * penalty

        # 確率の再分配
        df_prob.loc[mask, '1:Credit'] -= delta
        df_prob.loc[mask, '3:Mix'] += delta

    print(f"Mode A (Interest Afterglow) applied to: {mask.sum()} samples")

    return df_prob

def _refine_mode_bond_vol_noise(df_prob, df_shap, df_features):

    VIX_NEUTRAL_LIMIT = 0.5
    VOV_SHAP_THRESHOLD = 0.15
    
    shap = df_shap["1:Credit"]

    # 1. 必要な中間変数を一括計算
    total_shap = shap.abs().sum(axis=1)
    # MOVE_vovが判定に与えた影響のシェアを算出
    vov_ratio = shap['MOVE_vov'].abs() / total_shap.replace(0, np.inf)
    vix_z = df_features['VIX_z252']

    # 2. 条件に合致する「マスク」を作成
    # ・Credit確率が一定以上
    # ・債券ボラの寄与が閾値超え
    # ・VIXがパニック圏外（0.5以下）
    mask = (
        (df_prob['1:Credit'] >= 0.3) & 
        (vov_ratio > VOV_SHAP_THRESHOLD) & 
        (vix_z < VIX_NEUTRAL_LIMIT)
    )

    # 3. マスクされた行に対してのみ、一括で確率を移送
    if mask.any():
        # ペナルティ係数を計算 (Mode Bは最大0.8程度に抑える設定)
        penalty = (vov_ratio[mask] * 1.5).clip(upper=0.8)
        
        # 移動量（デルタ）を算出
        delta = df_prob.loc[mask, '1:Credit'] * penalty

        # 確率の再分配（Mode Bは債券要因なので 2:Bond へ移送）
        df_prob.loc[mask, '1:Credit'] -= delta
        df_prob.loc[mask, '2:Bond'] += delta

    # デバッグ用に修正件数を出力
    print(f"Mode B (Bond Vol Noise) applied to: {mask.sum()} samples")

    return df_prob
