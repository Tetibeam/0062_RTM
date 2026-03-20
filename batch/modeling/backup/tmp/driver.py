########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test,
    learning_lgbm_final,
    explain_prediction,
    )
from batch.modeling.visualize import (
    plot_driver_label
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################

def get_driver_model_beta(df_index, df_sp500):

    df_daily = df_index.copy()

    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df_daily)

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_daily, df_sp500)
    selected_features = [
        'TED_spread_level', 'Term_Premium_Stress', 'Equity_Gold_Ratio', 
        'Real_Nominal_ratio', 'Curve_10Y2Y', 'VVIX_z252', 
        'Stock_Bond_Corr_20d', 'DXY_z252', 'DFII10_z252'
    ]
    #df_driver = df_features[selected_features]

    # --- 学習モデル生成 ---
    df_driver = df_features.join(df_label["driver"])

    #driver_clf, df_driver_trajectory = learning_lgbm_final(
    #    df_driver, "driver", model_name="Driver", label_name_list=["1:Credit", "2:Bond", "3:Equity", "4:Mix"],
    #    n_estimators=1000,learning_rate=0.01,num_leaves=30, min_data_in_leaf=50,
    #    reg_alpha=0.5, reg_lambda=0.5,

    print(f"特徴量のリスト: {df_features.columns}")
    df_oof_all = learning_lgbm_test(
        df_driver, "driver", labels=["1:Credit", "2:Bond", "3:Mix"],
        n_splits=5, gap =20,
        n_estimators=1000,learning_rate=0.002,num_leaves=50, min_data_in_leaf=100,
        class_weight=None,#"balanced",
        reg_alpha=1, reg_lambda=1,learning_curve=True,
        )


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
    feats['VIX_diff5'] = vix.diff(5).reindex(master_index, method="ffill")
    feats['MOVE_diff5'] = move.diff(5).reindex(master_index, method="ffill")

    # 格差
    ratio = move / vix
    ratio = ratio.ffill()
    feats['MOVE_VIX_ratio'] = ratio.reindex(master_index, method="ffill")
    feats['MOVE_VIX_ratio_zscore'] = _featuring_z_score(ratio, window=252).reindex(master_index, method="ffill")

    # VIXの期間構造(VIX3Mの代用)
    feats['VIX_avg_gap'] = (vix / vix.rolling(window=60).mean()).reindex(master_index, method="ffill")

    # ボラのボラ
    feats['VIX_rv_20d'] = vix.pct_change(fill_method=None).rolling(20).std().reindex(master_index, method="ffill")

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
    feats['HY_diff5'] = hy.diff(5).reindex(master_index, method="ffill")
    feats["hy_z252"] = _featuring_z_score(hy, window=252).reindex(master_index, method="ffill")

    # 歴史的パニックの同期 (2008年対策)
    feats['TED_spread_level'] = tedrate.reindex(master_index, method="ffill")
    feats['TED_spread_diff5'] = tedrate.diff(5).reindex(master_index, method="ffill")

    # 現代の流動性ショック
    sofr_rolling_mean = sofr.rolling(20).mean()
    sofr_rolling_std = sofr.rolling(20).std().replace(0, np.nan)
    feats['SOFR_vol_spike'] = ((sofr - sofr_rolling_mean) / sofr_rolling_std).reindex(master_index, method="ffill")

    # 金融機関の「収益性・貸出意欲」の悪化
    feats['Term_Premium_Stress'] = (dgs10 - dgs3mo).reindex(master_index, method="ffill")

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
    feats['DFII10_diff5'] = dfii10.diff(5).reindex(master_index, method="ffill")
    feats['DFII10_z252'] = _featuring_z_score(dfii10, window=252).reindex(master_index, method="ffill")

    # イールドカーブ
    feats['Curve_10Y2Y'] = (dgs10 - dgs2).reindex(master_index, method="ffill")
    feats['Curve_10Y3M'] = (dgs10 - dgs3mo).reindex(master_index, method="ffill")

    # インフレ期待の加速
    feats['T10YIE_diff5'] = t10yie.diff(5).reindex(master_index, method="ffill")

    # 金利上昇の「質」の分解
    feats['Real_Nominal_ratio'] = (dfii10 / dgs10).reindex(master_index, method="ffill")

    # カーブの「フラット化」速度
    feats['Curve_flattening_speed'] = feats['Curve_10Y2Y'].diff(20).reindex(master_index, method="ffill")

    return feats

def _momentum_flow_feats(df, feats, master_index):
    # 指標
    dxy = df["DX-Y.NYB"].dropna()
    gold = df["GC=F"].dropna()
    sp500 = df["^GSPC"].dropna()
    tlt = df["TLT"].dropna()

    # ドルの引力
    feats['DXY_diff5'] = dxy.diff(5).reindex(master_index, method="ffill")
    feats['DXY_z252'] = _featuring_z_score(dxy, window=252).reindex(master_index, method="ffill")

    # アセット相関の変調
    returns_sp = sp500.pct_change()
    returns_tlt = tlt.pct_change()
    feats['Stock_Bond_Corr_20d'] = returns_sp.rolling(20).corr(returns_tlt).reindex(master_index, method="ffill")

    # リスクオン・オフの体温計
    feats['Equity_Gold_Ratio'] = (sp500 / gold).reindex(master_index, method="ffill")
    feats['Equity_Gold_Ratio_zscore'] = _featuring_z_score((sp500 / gold), window=252).reindex(master_index, method="ffill")

    # 資金の逃避速度
    feats['Flight_to_Safety_Index'] = (returns_tlt - returns_sp).reindex(master_index, method="ffill")

    # 市場のオーバーシュート
    feats['SP500_Ret_Z'] = _featuring_z_score(returns_sp.rolling(20).sum(), window=252).reindex(master_index, method="ffill")

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
def _make_label(df_daily,smear_days=10):

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
        (df['next_20d_ret_tlt'].abs() > (1.35 * tlt_vol * np.sqrt(20))) &
        ((df['next_20d_ret_tlt'].abs() / tlt_vol) > (df['next_20d_ret_sp500'].abs() / sp500_vol * 0.5))
    )

    # --- Step 2: Label Smearing (期間の拡張) ---
    # 未来に危機が起きるなら、現在(t)からその期間までを「予兆期間」として同じラベルにする
    # rolling().max() を shift(-smear_days) することで、過去方向にラベルを伸ばす
    smeared_credit = raw_credit.rolling(window=smear_days + 1, min_periods=1).max().shift(-smear_days).fillna(0).astype(bool)
    smeared_bond = raw_bond.rolling(window=smear_days + 1, min_periods=1).max().shift(-smear_days).fillna(0).astype(bool)

    # --- Step 3: 優先順位に基づいた最終ラベルの付与 ---
    df["driver"] = 3 # Neutral (Mix)
    df["driver_name"]= "Mix"

    # 優先度低：Bond
    df.loc[smeared_bond, 'driver'] = 2
    df.loc[smeared_bond, 'driver_name'] = "Bond"

    # 優先度高：Credit (Bondの上書き)
    df.loc[smeared_credit, 'driver'] = 1
    df.loc[smeared_credit, 'driver_name'] = "Credit"
    #_analysis_label(df, df_daily)

    return df.dropna()

def _analysis_label(df, df_daily):
        # 分析・可視化
    stats = df['driver_name'].value_counts().to_frame(name='Count')
    stats['Percentage (%)'] = (df['driver_name'].value_counts(normalize=True) * 100).round(2)
    print(stats)

    market_summary = df.groupby('driver_name').agg({
        'next_20d_ret_sp500': ['mean', 'std', 'min', 'max'],
        'next_20d_ret_tlt': ['mean', 'std'],
        'next_20d_diff_hy': ['mean']
    }).round(4)
    print(market_summary)

    # 継続日数の算出
    df['change'] = df['driver_name'] != df['driver_name'].shift()
    df['regime_id'] = df['change'].cumsum()

    # 各期間の長さをカウント
    duration_stats = df.groupby(['regime_id', 'driver_name']).size().reset_index(name='duration')
    avg_duration = duration_stats.groupby('driver_name')['duration'].mean().round(1)
    print(f"平均継続日数:\n{avg_duration}")

    # 遷移マトリクス（現在の状態 -> 次の状態）
    transition_matrix = pd.crosstab(
        df['driver_name'], 
        df['driver_name'].shift(-1), 
        normalize='index'
    ).round(2)

    print("遷移マトリクス（行：現在 -> 列：次）:")
    print(transition_matrix)
    plot_driver_label(df, df_daily, start_date="2021-01-01", end_date="2023-01-01")

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
