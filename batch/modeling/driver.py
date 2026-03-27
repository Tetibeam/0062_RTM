########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test_driver,
    )
from batch.modeling.visualize import (
    plot_driver_soft_label,
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
        #'VIX_diff5_z252',
        #'MOVE_diff5_z252',
        #'MOVE_VIX_ratio_z252',
        #'VIX_gap_z252',
        #'VIX_rv_z252',
        "MOVE_vov",
        #"MOVE_accel",
        
        #'HY_diff5_z252',
        'hy_z252',
        #"HY_Acceleration_z252",
        #'TED_spread_z252',
        #'TED_diff5_z252',
        'SOFR_vol_spike',
        'Term_Premium_z252',
        'Credit_Equity_Divergence',
        #'Term_Premium_diff5',
        "Term_Premium_diff5_z252",

        'DFII10_diff5_z252',
        #'DFII10_z252',
        #'Curve_10Y2Y_z252', 
        #'Curve_10Y3M_z252',
        #'T10YIE_diff5_z252',
        #'Real_Nominal_ratio_z252',
        #'Curve_flattening_speed_z252',

        'DXY_diff5_z252',######
        #'DXY_z252',
        #'Stock_Bond_Corr_10d_z252',
        'Stock_Bond_Corr_z252',
        #"Gold_z252",
        'Equity_Gold_Ratio_z252',
        'Flight_to_Safety_z252',
        #'SP500_Ret_Z',
        #"tlt_z252",
        #"tlt_ret_z252",
        "tlt_diff_z252",######
        "tlt_hy_ratio_z252",
        #"tlt_hy_diff_z252",
        #"oil_ret_z252",
        #"SPX_vs_SMA200"
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

    df_driver = df_features.join(df_label[["driver", "next_20d_ret_sp500"]])

    """driver_clf, df_driver_trajectory = learning_lgbm_final(
        df_driver, "driver", model_name="Driver", label_name_list=["1:Credit", "2:Bond", "3:Mix"],
        n_estimators=2800,learning_rate=0.001,num_leaves=50, min_data_in_leaf=100,
        reg_alpha=0.3, reg_lambda=0.3,)"""

    print(f"特徴量のリスト: {df_features.columns}")
    df_oof_all, df_shap, df_oof_ev = learning_lgbm_test_driver(
        df_driver, "driver", labels=["1:Credit", "2:Bond", "3:Mix"],
        n_splits=5, gap =30,
        n_estimators=2500,learning_rate=0.001,num_leaves=50, min_data_in_leaf=90,
        class_weight="balanced",
        sample_weight=df_label["sample_weight"],
        reg_alpha=0.3, reg_lambda=0.3, learning_curve=False,
        )

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

    #_chk_ev_hist(df_oof_ev)
 

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
    feats['VIX_diff5_z252'] = _featuring_z_score(vix_diff, window=252).reindex(master_index, method="ffill")
    feats['MOVE_diff5_z252'] = _featuring_z_score(move_diff, window=252).reindex(master_index, method="ffill")
    feats['MOVE_accel'] = feats['MOVE_z252'].diff(1).diff(5).reindex(master_index, method="ffill")

    # 格差
    ratio = move / vix
    ratio = ratio.ffill()
    #feats['MOVE_VIX_ratio'] = ratio.reindex(master_index, method="ffill")
    feats['MOVE_VIX_ratio_z252'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")

    # VIXの期間構造(VIX3Mの代用)
    vix_gap = vix / vix.rolling(window=60).mean()
    feats['VIX_gap_z252'] = _featuring_z_score(vix_gap, window=252).reindex(master_index, method="ffill")

    # ボラのボラ
    vix_rv = vix.pct_change().rolling(20).std()
    feats['VIX_rv_z252'] = _featuring_z_score(vix_rv, window=252).reindex(master_index, method="ffill")
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
    feats['HY_diff5_z252'] = _featuring_z_score(hy_diff, window=252).reindex(master_index, method="ffill")
    feats["hy_z252"] = _featuring_z_score(hy, window=252).reindex(master_index, method="ffill")
    smoothed_hy = hy.ewm(span=5, adjust=False).mean()
    acceleration = smoothed_hy.diff().diff()
    feats['HY_Acceleration_z252'] =_featuring_z_score(acceleration, window=252).reindex(master_index, method="ffill")

    # 歴史的パニックの同期 (2008年対策)
    #feats['TED_spread_level'] = tedrate.reindex(master_index, method="ffill")
    #feats['TED_spread_diff5'] = tedrate.diff(5).reindex(master_index, method="ffill")
    feats['TED_spread_z252'] = _featuring_z_score(tedrate, window=252).reindex(master_index, method="ffill")
    feats['TED_diff5_z252'] = _featuring_z_score(tedrate.diff(5), window=252).reindex(master_index, method="ffill")

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
    feats['DFII10_diff5_z252'] = _featuring_z_score(dfii10_diff, window=252).reindex(master_index, method="ffill")
    feats['DFII10_z252'] = _featuring_z_score(dfii10, window=252).reindex(master_index, method="ffill")

    # イールドカーブ
    curve10y2y = dgs10 - dgs2
    curve10y3m = dgs10 - dgs3mo
    feats['Curve_10Y2Y_z252'] = _featuring_z_score(curve10y2y, window=252).reindex(master_index, method="ffill")
    feats['Curve_10Y3M_z252'] = _featuring_z_score(curve10y3m, window=252).reindex(master_index, method="ffill")

    # インフレ期待の加速
    t10yie_diff = t10yie.diff(5)
    feats['T10YIE_diff5_z252'] = _featuring_z_score(t10yie_diff, window=252).reindex(master_index, method="ffill")

    # 金利上昇の「質」の分解
    real_nominal_ratio = dfii10 / dgs10
    feats['Real_Nominal_ratio_z252'] = _featuring_z_score(real_nominal_ratio, window=252).reindex(master_index, method="ffill")

    # カーブの「フラット化」速度
    flattening_speed = curve10y2y.diff(20)
    feats['Curve_flattening_speed_z252'] = _featuring_z_score(flattening_speed, window=252).reindex(master_index, method="ffill")

    return feats

def _momentum_flow_feats(df, feats, master_index):
    # 指標
    dxy = df["DX-Y.NYB"].dropna()
    gold = df["GC=F"].dropna()
    sp500 = df["^GSPC"].dropna()
    tlt = df["TLT"].dropna()
    hy = df["BAMLH0A0HYM2"].dropna()
    oil = df["CL=F"].dropna()

    sma200 = sp500.rolling(window=200).mean()
    feats['SPX_vs_SMA200'] = _featuring_z_score((sp500 - sma200) / sma200, window=252).reindex(master_index, method="ffill")


    # ドルの引力
    dxy_diff = dxy.diff(5)
    feats['DXY_diff5_z252'] = _featuring_z_score(dxy_diff, window=252).reindex(master_index, method="ffill")
    feats['DXY_z252'] = _featuring_z_score(dxy, window=252).reindex(master_index, method="ffill")

    # アセット相関の変調
    returns_sp = sp500.pct_change()
    returns_tlt = tlt.pct_change()
    corr10 = returns_sp.rolling(10).corr(returns_tlt)
    corr20 = returns_sp.rolling(20).corr(returns_tlt)
    feats['Stock_Bond_Corr_10d_z252'] = _featuring_z_score(corr10, window=252).reindex(master_index, method="ffill")
    feats['Stock_Bond_Corr_z252'] = _featuring_z_score(corr20, window=252).reindex(master_index, method="ffill")


    # リスクオン・オフの体温計
    feats['Gold_z252'] = _featuring_z_score(gold, window=252).reindex(master_index, method="ffill")
    equity_gold = sp500 / gold
    feats['Equity_Gold_Ratio_z252'] = _featuring_z_score(equity_gold, window=252).reindex(master_index, method="ffill")

    # 資金の逃避速度
    fts_index = returns_tlt - returns_sp
    feats['Flight_to_Safety_z252'] = _featuring_z_score(fts_index, window=252).reindex(master_index, method="ffill")

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
        (df['next_20d_ret_sp500'] < 0) &
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
        'Equity_Gold_Ratio_z252',
        'VVIX_z252'
        ]], how='left')

    df_oof_ev['ev_rank'] = df_oof_ev['ev_rank'].astype(str)
    #mask_rescue = (
    #    (df_oof_ev['ev_rank'] == 'CRITICAL') &
    #    (df_oof_ev['Credit_Equity_Divergence'] <= 1.0)
    #)
    mask_rescue = (
        (df_oof_ev['ev_rank'] == 'CRITICAL') & (
            (df_oof_ev['Equity_Gold_Ratio_z252'] <= 0.5) | # 株が金より明らかに強い（救済）
            (df_oof_ev['VVIX_z252'] >= 0.2)                  # オプション市場が冷静（救済）
        )
    )

    df_oof_ev.loc[mask_rescue, 'ev_rank'] = "Neutral"
    df_oof_ev = df_oof_ev.drop(columns=[
        'Credit_Equity_Divergence', 'Equity_Gold_Ratio_z252', 'VVIX_z252'])

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
    'Equity_Gold_Ratio_z252'
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
        (df_oof_ev['Equity_Gold_Ratio_z252'] > 0.8)     # 相対的な強さ
    )
    
    # 救済：これらを Neutral (市場参加) に格下げ
    df_oof_ev.loc[mask, 'ev_rank'] = "Neutral"
    df_oof_ev = df_oof_ev.drop(columns=[
        'Term_Premium_z252', 'Credit_Equity_Divergence', 'Equity_Gold_Ratio_z252'])
    
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