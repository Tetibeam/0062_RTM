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
    plot_driver_trajectory,
    plot_index
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################

def get_credit_driver_beta(df_index, df_sp500):

    df_daily = df_index.copy()

    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df_daily)

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_daily, df_sp500)

    features_refined = [
        'VIX_z21',
        #'VVIX_z21',
        'MOVE_z21',
        #'VIX_diff5_zscore',
        #'MOVE_diff5_zscore',
        #'MOVE_VIX_ratio_zscore',###
        'VIX_gap_zscore',
        #'VIX_rv_zscore',
        "MOVE_vov",###########
        #"MOVE_accel",

        #'HY_diff5_zscore',
        'hy_z21',
        'TED_spread_z21',
        #'TED_diff5_zscore',
        #'SOFR_vol_spike',
        'Term_Premium_z21',
        #'Credit_Equity_Divergence',
        #'Term_Premium_diff5',
        #"Term_Premium_diff5_z21",

        #'DFII10_diff5_zscore',
        #'DFII10_z21',
        #'Curve_10Y2Y_z21', #1
        #'Curve_10Y3M_z21',
        #'T10YIE_diff5_zscore',
        #'Real_Nominal_ratio_zscore',
        #'Curve_flattening_speed_zscore',

        #'DXY_diff5_zscore',
        #'DXY_z21',
        #'Stock_Bond_Corr_20d',
        #'Stock_Bond_Corr_zscore',
        #"Gold_zscore",
        #'Equity_Gold_Ratio_zscore',
        #'Flight_to_Safety_zscore',
        #'SP500_Ret_Z',
        #"tlt_z21",
        #"tlt_ret_z21",
        #"tlt_diff_z21",
        #"tlt_hy_ratio_z21",
        "tlt_hy_diff_z21",
        #"oil_ret_z21"
        ]
    df_features = df_features[features_refined]

    # --- 学習モデル生成 ---
    # サンプルフェイト
    df_label['sample_weight'] = 1.0

    mask_credit = (df_label['driver'] == 1)
    df_label.loc[mask_credit, 'sample_weight'] = df_label['score']

    df_driver = df_features.join(df_label["driver"])

    """driver_clf, df_driver_trajectory = learning_lgbm_final(
        df_driver, "driver", model_name="Driver", label_name_list=["0:Safe", "1:Credit"],
        n_estimators=2800,learning_rate=0.001,num_leaves=50, min_data_in_leaf=100,
        reg_alpha=0.3, reg_lambda=0.3,)"""

    print(f"特徴量のリスト: {df_features.columns}")
    df_oof_all, df_shap = learning_lgbm_test(
        df_driver, "driver", labels=["0:Safe", "1:Credit"],
        n_splits=5, gap =30,
        n_estimators=2000,learning_rate=0.001,num_leaves=10, min_data_in_leaf=50,
        class_weight="balanced",objective="binary",
        sample_weight=df_label["sample_weight"],
        reg_alpha=0.5, reg_lambda=0.5, learning_curve=True,
        )

    # 分析、可視化
    """_ = plot_driver_trajectory(
        df_oof_all, df_daily["^GSPC"].pct_change().dropna(),
        ["0:Safe", "1:Credit"],
        start_date="2021-10-01", end_date="2023-01-01"
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
    start = "2005-03-16"#feats["MOVE_z21"].first_valid_index()
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
    feats["VIX_z21"] = _featuring_z_score(vix, window=21).reindex(master_index, method="ffill")
    feats["VVIX_z21"] = _featuring_z_score(vvix, window=21).reindex(master_index, method="ffill")
    feats["MOVE_z21"] = _featuring_z_score(move, window=21).reindex(master_index, method="ffill")

    # 加速
    vix_diff = vix.diff(5)
    move_diff = move.diff(5)
    feats['VIX_diff5_zscore'] = _featuring_z_score(vix_diff, window=21).reindex(master_index, method="ffill")
    feats['MOVE_diff5_zscore'] = _featuring_z_score(move_diff, window=21).reindex(master_index, method="ffill")
    feats['MOVE_accel'] = feats['MOVE_z21'].diff(1).diff(5).reindex(master_index, method="ffill")

    # 格差
    ratio = move / vix
    ratio = ratio.ffill()
    #feats['MOVE_VIX_ratio'] = ratio.reindex(master_index, method="ffill")
    feats['MOVE_VIX_ratio_zscore'] = _featuring_z_score(ratio, window=21).reindex(master_index, method="ffill")

    # VIXの期間構造(VIX3Mの代用)
    vix_gap = vix / vix.rolling(window=60).mean()
    feats['VIX_gap_zscore'] = _featuring_z_score(vix_gap, window=21).reindex(master_index, method="ffill")

    # ボラのボラ
    vix_rv = vix.pct_change().rolling(20).std()
    feats['VIX_rv_zscore'] = _featuring_z_score(vix_rv, window=21).reindex(master_index, method="ffill")
    move_z21 = _featuring_z_score(move, window=21).reindex(master_index, method="ffill")
    feats['MOVE_vov'] = move_z21.diff().rolling(20).std().reindex(master_index, method="ffill")
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
    feats['HY_diff5_zscore'] = _featuring_z_score(hy_diff, window=21).reindex(master_index, method="ffill")
    feats["hy_z21"] = _featuring_z_score(hy, window=21).reindex(master_index, method="ffill")

    # 歴史的パニックの同期 (2008年対策)
    #feats['TED_spread_level'] = tedrate.reindex(master_index, method="ffill")
    #feats['TED_spread_diff5'] = tedrate.diff(5).reindex(master_index, method="ffill")
    feats['TED_spread_z21'] = _featuring_z_score(tedrate, window=21).reindex(master_index, method="ffill")
    feats['TED_diff5_zscore'] = _featuring_z_score(tedrate.diff(5), window=21).reindex(master_index, method="ffill")

    # 現代の流動性ショック
    sofr_rolling_mean = sofr.rolling(20).mean()
    sofr_rolling_std = sofr.rolling(20).std().replace(0, np.nan)
    feats['SOFR_vol_spike'] = ((sofr - sofr_rolling_mean) / sofr_rolling_std).reindex(master_index, method="ffill")

    # 金融機関の「収益性・貸出意欲」の悪化
    term_premium = dgs10 - dgs3mo
    feats['Term_Premium_z21'] = _featuring_z_score(term_premium, window=21).reindex(master_index, method="ffill")
    feats['Term_Premium_diff5'] = feats['Term_Premium_z21'].diff(5).reindex(master_index, method="ffill")
    feats['Term_Premium_diff5_z21'] = _featuring_z_score(feats['Term_Premium_diff5'] , window=21).reindex(master_index, method="ffill")

    # クレジットとボラティリティの「乖離」
    vix_z = _featuring_z_score(vix, window=21)
    feats['Credit_Equity_Divergence'] = (feats['hy_z21'] - vix_z).reindex(master_index, method="ffill")

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
    feats['DFII10_diff5_zscore'] = _featuring_z_score(dfii10_diff, window=21).reindex(master_index, method="ffill")
    feats['DFII10_z21'] = _featuring_z_score(dfii10, window=21).reindex(master_index, method="ffill")

    # イールドカーブ
    curve10y2y = dgs10 - dgs2
    curve10y3m = dgs10 - dgs3mo
    feats['Curve_10Y2Y_z21'] = _featuring_z_score(curve10y2y, window=21).reindex(master_index, method="ffill")
    feats['Curve_10Y3M_z21'] = _featuring_z_score(curve10y3m, window=21).reindex(master_index, method="ffill")

    # インフレ期待の加速
    t10yie_diff = t10yie.diff(5)
    feats['T10YIE_diff5_zscore'] = _featuring_z_score(t10yie_diff, window=21).reindex(master_index, method="ffill")

    # 金利上昇の「質」の分解
    real_nominal_ratio = dfii10 / dgs10
    feats['Real_Nominal_ratio_zscore'] = _featuring_z_score(real_nominal_ratio, window=21).reindex(master_index, method="ffill")

    # カーブの「フラット化」速度
    flattening_speed = curve10y2y.diff(20)
    feats['Curve_flattening_speed_zscore'] = _featuring_z_score(flattening_speed, window=21).reindex(master_index, method="ffill")

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
    feats['DXY_diff5_zscore'] = _featuring_z_score(dxy_diff, window=21).reindex(master_index, method="ffill")
    feats['DXY_z21'] = _featuring_z_score(dxy, window=21).reindex(master_index, method="ffill")

    # アセット相関の変調
    returns_sp = sp500.pct_change()
    returns_tlt = tlt.pct_change()
    corr = returns_sp.rolling(20).corr(returns_tlt)
    feats['Stock_Bond_Corr_20d'] = corr.reindex(master_index, method="ffill")
    feats['Stock_Bond_Corr_zscore'] = _featuring_z_score(corr, window=21).reindex(master_index, method="ffill")


    # リスクオン・オフの体温計
    feats['Gold_zscore'] = _featuring_z_score(gold, window=21).reindex(master_index, method="ffill")
    equity_gold = sp500 / gold
    feats['Equity_Gold_Ratio_zscore'] = _featuring_z_score(equity_gold, window=21).reindex(master_index, method="ffill")

    # 資金の逃避速度
    fts_index = returns_tlt - returns_sp
    feats['Flight_to_Safety_zscore'] = _featuring_z_score(fts_index, window=21).reindex(master_index, method="ffill")

    # 市場のオーバーシュート
    feats['SP500_Ret_Z'] = _featuring_z_score(returns_sp.rolling(20).sum(), window=21).reindex(master_index, method="ffill")

    feats['tlt_z21'] = _featuring_z_score(tlt, window=21).reindex(master_index, method="ffill")
    feats['tlt_ret_z21'] = _featuring_z_score(tlt.pct_change(), window=21).reindex(master_index, method="ffill")
    feats['tlt_diff_z21'] = _featuring_z_score(tlt.diff(5), window=21).reindex(master_index, method="ffill")
    ratio = tlt / hy
    ratio = ratio.ffill()
    feats['tlt_hy_ratio_z21'] = _featuring_z_score(ratio, window=21).reindex(master_index, method="ffill")
    tlt_hy_diff = np.log(tlt)- np.log(hy)
    feats['tlt_hy_diff_z21'] = _featuring_z_score(tlt_hy_diff, window=21).reindex(master_index, method="ffill")

    feats['oil_ret_z21'] = _featuring_z_score(oil.pct_change(), window=21).reindex(master_index, method="ffill")


    return feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

def _featuring_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / (avg_loss + 1e-9) # ゼロ除算防止

    return 100 - (100 / (1 + rs))

def _featuring_m_breadth(df, window=10):
    df = df.dropna(how="all")
    returns = df.pct_change()

    up = (returns > 0).sum(axis=1)
    down = (returns < 0).sum(axis=1)
    total = returns.count(axis=1)

    breadth_up = up / total
    breadth_down = down / total
    breadth_diff = (breadth_up - breadth_down)

    return breadth_diff.rolling(window=window).mean()

def _featuring_sector_var(df, window=5):
    sector_prices = df.dropna(how="all")
    sector_returns = sector_prices.pct_change()
    # 日次セクター横断分散（モデル入力用・生値）
    sector_dispersion = sector_returns.var(axis=1)
    return sector_dispersion.rolling(window=window).mean()

########################################################
# 教師ラベル作成 - カンニングラベル
########################################################
def _make_label(df_daily, smear_days=5, threshold=0.4):

    # マスターカレンダー
    master_index = df_daily["^GSPC"].dropna().index
    df = pd.DataFrame(index=master_index)

    # 1. 指標の計算
    hy_spread = df_daily["BAMLH0A0HYM2"].dropna()
    sp500 = df_daily["^GSPC"].dropna()
    vix = df_daily["VIXCLS"].dropna().reindex(master_index, method='ffill')

    # ボラティリティ計算
    hy_diff_vol = hy_spread.diff().rolling(60, min_periods=20).std().reindex(master_index, method='ffill')
    sp500_vol = sp500.pct_change().rolling(60, min_periods=20).std().reindex(master_index, method='ffill')

    # 未来20日の変化
    df['next_20d_diff_hy'] = hy_spread.diff(20).shift(-20).reindex(master_index, method="ffill")
    df['next_20d_ret_sp500'] = sp500.pct_change(20).shift(-20).reindex(master_index, method="ffill")

    # 2. Raw Flag（生のフラグ）の算出
    # 基本条件：HYスプレッドが統計的に有意に拡大している
    is_hy_expanding = df['next_20d_diff_hy'] > (2.0 * hy_diff_vol * np.sqrt(20))

    # 優位性条件：S&P500の下落率に対して、スプレッドの拡大の方が相対的に深刻（ショック局面）
    is_credit_dominant = (df['next_20d_diff_hy'] / hy_diff_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5)

    raw_credit = is_hy_expanding & is_credit_dominant

    # --- Step 3: Censorship（検閲ロジック） ---
    # 2013/2017年対策：株価が強く（25日線の上）、かつVIXが低い（平穏）なら、
    # スプレッドが拡大していてもそれは「真の危機」ではないとみなしてフラグを折る。
    sp500_sma = sp500.rolling(25).mean().reindex(master_index, method='ffill')
    is_market_calm = (sp500 > sp500_sma) & (vix < 20)

    # 市場が平穏な中でのスプレッド拡大は「ノイズ」として除外
    raw_credit_filtered = raw_credit & (~is_market_calm)

    # --- Step 4: 数学的Smearing（減衰スコアの計算） ---
    def calculate_decay_score(raw_series, window):
        scores = np.zeros(len(raw_series))
        event_indices = np.where(raw_series)[0]
        tau = window / 1.5
        for idx in event_indices:
            for d in range(window + 1):
                if idx - d >= 0:
                    decay_val = np.exp(-d / tau)
                    scores[idx - d] = max(scores[idx - d], decay_val)
        return scores

    # スコア（確信度）の算出
    df['score'] = calculate_decay_score(raw_credit, smear_days)

    # --- Step 5: ターゲットと重みの確定 ---
    # ターゲット：スコアが閾値を超えたら「1（Credit）」
    df['driver'] = (df['score'] > threshold).astype(int)

    # 学習の重み：スコアそのものを重みにすることで、イベント直前の学習を強化
    # 平時(0)の重みは1.0、Credit(1)の重みはスコア（0.4〜1.0）
    # ※不均衡調整が必要な場合はここで調整
    df['sample_weight'] = 1.0
    df.loc[df['driver'] == 1, 'sample_weight'] = df['score']

    # 不要な列を整理
    result = df[['driver', 'sample_weight', 'score']]

    #print(f"Credit Sentinel Label Created. Event Ratio: {result['driver'].mean():.2%}")
    #_analysis_label(df, df_daily)

    return result.dropna()

def _analysis_label(df, df_daily):
    # 分析・可視化
    stats = df['driver'].value_counts().to_frame(name='Count')
    stats['Percentage (%)'] = (df['driver'].value_counts(normalize=True) * 100).round(2)
    print(stats)

    market_summary = df.groupby('driver').agg({
        'next_20d_ret_sp500': ['mean', 'std', 'min', 'max'],
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
    plot_driver_soft_label(df, df_daily, target="Credit", start_date="2024-01-01", end_date="2026-01-01")

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
