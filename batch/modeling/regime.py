########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test,
    learning_lgbm_final,
    explain_prediction,
    )
from batch.modeling.visualize import (
    plot_regime_trajectory,
    plot_driver_trajectory,
    plot_shap_explanation,
    plot_regime_label,
    plot_factor_label,
    plot_index
    )
from batch.modeling.inference_engine import (
    RTMInferenceEngine,
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################

def get_market_regime_model_beta(df_index, df_sp500):

    df_daily = df_index.copy()

    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df_daily)

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_daily, df_sp500)

    # --- 学習モデル生成 ---

    # Regime Prism
    df_regime = df_features.join(df_label["regime"]).dropna()

    #regime_clf, df_regime_trajectory = learning_lgbm_final(
    #    df_regime, "regime", model_name="Regime", label_name_list=["1: Golden Dip", "2: Crash Flash", "3: Slow Bleed", "4: Liquidity In", "5: Healthy/Neutral"],
    #    n_estimators=1000,learning_rate=0.01,num_leaves=25, min_data_in_leaf=20,
    #    reg_alpha=0.4, reg_lambda=0.4,
    #    )

    df_oof_all = learning_lgbm_test(
        df_regime, "regime", labels=["1: Golden Dip", "2: Crash Flash", "3: Slow Bleed", "4: Liquidity In", "5: Healthy/Neutral"],
        n_splits=5, gap =40,
        n_estimators=1000,learning_rate=0.01,num_leaves=25, min_data_in_leaf=20,
        reg_alpha=0.4, reg_lambda=0.4,
        learning_curve=False,
        )

    # --- 可視化 ---
    # Market Navigator
    #df_all = df_features.join(df_label[["regime", "driver"]])
    #plot_market_navigator(df_all, regime_clf=regime_clf, driver_clf=driver_clf)

    # Regime Prism
    #plot_regime_trajectory(df_oof_all_driver, df_regime["sp500_ret_1d"].dropna(), labels=driver_labels, start_date="2022-01-01", end_date="2023-01-01") # 
    #plot_regime_trajectory(df_regime, regime_clf, start_date="2008-09-01", end_date="2009-03-01") # リーマン
    #plot_regime_trajectory(df_regime, regime_clf, start_date="2020-01-01", end_date="2020-06-01") # コロナ
    #plot_regime_trajectory(df_regime, regime_clf, start_date="2022-01-01", end_date="2022-12-01") # インフレショック
    #plot_regime_trajectory(df_regime, regime_clf, start_date="2018-09-01", end_date="2019-03-01") # クリスマスショック

    # Driver Profiler
    #plot_driver_trajectory(df_features, driver_clf, start_date="2006-10-01", end_date="2009-10-01")

    # --- SHAP ---
    # Driver Profiler
    #importance_df, pred_class, display_date = explain_prediction(driver_clf, df_driver,  target_date="2022-10-14")
    #plot_shap_explanation(importance_df, pred_class)


    # --- リバランスシミュレーター ---
    #run_modulator_backtest(df_trajectory, df_regime_features["sp500_ret_1d"])

    #return regime_clf, df_regime_trajectory, df_regime, driver_clf, df_driver_trajectory, df_driver
    return df_oof_all

def run_modulator_backtest(df_trajectory, sp500_ret_1d):
    # 1. 各レジュームのウェイト設定（ここを調整して自分好みの戦略にできます）
    weights = {
        '1: Golden Dip': 1.2,
        '2: Flash Crash': 0.4,
        '3: Slow Bleed': 0.6,
        '4: Liquidity In': 0.8,
        '5: Healthy': 1.0
    }

    # 2. 毎日のターゲット・エクスポージャー（露出度）を算出
    # 確率 * ウェイト の合計
    exposure = df_trajectory[list(weights.keys())].mul(list(weights.values())).sum(axis=1)

    # 3. リターンの計算
    # 昨日の予測に基づいて、今日のリターンが決まる（shift(1)が重要）
    strategy_ret = exposure.shift(1) * sp500_ret_1d

    # 4. 累積リターンの計算（ベンチマークとの比較）
    cum_strategy = (1 + strategy_ret.fillna(0)).cumprod()
    cum_bh = (1 + sp500_ret_1d.fillna(0)).cumprod()

    # 5. ドローダウンの計算
    dd_strategy = cum_strategy / cum_strategy.cummax() - 1
    dd_bh = cum_bh / cum_bh.cummax() - 1

    return cum_strategy, cum_bh, dd_strategy, dd_bh, exposure

########################################################
# 特徴量抽出
########################################################
def _featuring_all(df_daily, df_sp500):

    # --- リバランスの実行基準となるマスターカレンダー ---
    master_index = df_daily["^GSPC"].dropna().index

    # --- 指標の取得 ---
    sp500 = df_daily["^GSPC"].dropna()
    vix = df_daily['VIXCLS'].dropna()
    hy = df_daily['BAMLH0A0HYM2'].dropna()
    ig = df_daily["BAMLC0A0CM"].dropna()
    tlt = df_daily['TLT'].dropna()
    sector = df_daily[["XLF", "XLK", "XLE", "XLY", "XLP", "XLU"]].dropna(how="all")
    nasdaq = df_daily["^IXIC"].dropna()
    russell = df_daily["^RUT"].dropna()
    gold = df_daily["GC=F"].dropna()
    dxy = df_daily["DX-Y.NYB"].dropna()
    vvix = df_daily["VVIX"].dropna()
    skew = df_daily["^SKEW"].dropna()
    xlk = df_daily["XLK"].dropna()
    xlp = df_daily["XLP"].dropna()
    usdjpy = df_daily["DEXJPUS"].dropna()
    usdchf = df_daily["DEXSZUS"].dropna()

    # --- 準備計算 ---
    sp500_ret_1d = sp500.pct_change(fill_method=None)
    vvix = vvix["2006-08-01":]  # CBOEで安定して日次でとれるのはこの月から

    # --- LayerA - リスク資産の動き ---
    df_a = pd.DataFrame(index=master_index)
    df_a["sp500_ret_1d"] = sp500_ret_1d.reindex(master_index)
    df_a["sp500_rsi_14d"] = _featuring_rsi(sp500, window=14).reindex(master_index)
    df_a["sp500_breadth_diff_10d"] = _featuring_m_breadth(df_sp500, window=10).reindex(master_index)

    df_a["sector_dispersion_5d"] = _featuring_sector_var(sector, window=1).reindex(master_index)
    df_a["nasdaq_ret_diff_sp500"] = (nasdaq.pct_change(fill_method=None) - sp500_ret_1d).reindex(master_index)
    df_a["russell_ret_diff_sp500"] = (russell.pct_change(fill_method=None) - sp500_ret_1d).reindex(master_index)
    df_a["nasdaq_ret_5d"] = nasdaq.pct_change(5,fill_method=None).reindex(master_index)
    df_a["russell_ret_5d"] = russell.pct_change(5,fill_method=None).reindex(master_index)

    # Driver Profiler専用
    #df_a["xlk/xlp_ret_5d"] = (xlk/xlp).pct_change(5,fill_method=None).reindex(master_index)

    #print(df_a)
    #check_nan_time(df_a, "2005-01-01")

    # --- LayerB - ボラティリティ・不確実性 ---
    df_b = pd.DataFrame(index=master_index)
    df_b['vix_level'] = vix.reindex(master_index)   # レベル
    df_b['vix_chg_5d'] = vix.diff(5).reindex(master_index)  # 直近5日変化
    df_b["vix_zscore_10d"] = _featuring_z_score(vix, window=10).reindex(master_index)
    df_b["vix_panic_duration"] = ((vix > 20).astype(int)).rolling(20).sum().reindex(master_index)

    # Driver Profiler専用
    #df_b["vvix_zscore_10d"] = _featuring_z_score(vvix, window=10).reindex(master_index)

    #print(df_b)
    #check_nan_time(df_b, "2005-01-01")

    # --- LayerC - クレジット市場 ---
    df_c = pd.DataFrame(index=master_index)
    df_c["hy_level"] = hy.reindex(master_index)
    df_c["ig_level"] = ig.reindex(master_index)
    df_c['hy_diff_5d'] = hy.diff(5).reindex(master_index) # 直近5日変化
    df_c['ig_diff_5d'] = ig.diff(5).reindex(master_index) # 直近5日変化

    #print(df_c)
    #check_nan_time(df_c, "2005-01-01")

    # --- LayerD - 安全資産・ヘッジ資産 ---
    df_d = pd.DataFrame(index=master_index)
    df_d['tlt_ret_20d'] = tlt.pct_change(20).reindex(master_index)          # 直近20日変化率
    df_d["dxy_ret_1d"] = dxy.pct_change(5,fill_method=None).reindex(master_index)
    df_d["sp500_gold_corr_10d"] = sp500_ret_1d.rolling(window=10).corr(gold.pct_change(fill_method=None)).reindex(master_index)
    # Driver Profiler専用
    #df_d["udfjpy_ret_1d"] = usdjpy.pct_change(fill_method=None).reindex(master_index)
    #df_d["udfjpy_ret_5d"] = usdjpy.pct_change(5,fill_method=None).reindex(master_index)
    #df_d["usdchf_ret_1d"] = usdchf.pct_change(fill_method=None).reindex(master_index)
    #df_d["usdchf_ret_5d"] = usdchf.pct_change(5,fill_method=None).reindex(master_index)
    #df_d["udfjpy_ret_vol"] = usdjpy.rolling(window=20).std().reindex(master_index)
    #df_d["usdchf_ret_vol"] = usdchf.rolling(window=20).std().reindex(master_index)
    #print(df_d)
    #check_nan_time(df_d, "2005-01-01")

    # 結合
    df_all = pd.concat([df_a, df_b, df_c, df_d], axis=1).sort_index()

    # 未来リーク耐性
    df_all = df_all.shift(1).dropna(how="all")

    #check_nan_time(df_all, "2005-01-01")

    # --- 全特徴量の共通有効期間 ---
    #start = df_all.apply(pd.Series.first_valid_index).max()
    start = df_all["tlt_ret_20d"].first_valid_index()
    end   = df_all.apply(pd.Series.last_valid_index).min()
    print("\nレジューム学習の特徴量の期間: ",start, end)
    df_all = df_all.loc[start:end].ffill()

    #check_nan_time(df_all, "2005-01-01")
    #print(df_all)
    #plot_index(pd.concat([sp500, hy, df_all["tlt_ret_20d"], dxy],axis=1))

    return df_all

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
    returns = df.pct_change(fill_method=None)

    up = (returns > 0).sum(axis=1)
    down = (returns < 0).sum(axis=1)
    total = returns.count(axis=1)

    breadth_up = up / total
    breadth_down = down / total
    breadth_diff = (breadth_up - breadth_down)

    return breadth_diff.rolling(window=window).mean()

def _featuring_sector_var(df, window=5):
    sector_prices = df.dropna(how="all")
    sector_returns = sector_prices.pct_change(fill_method=None)
    # 日次セクター横断分散（モデル入力用・生値）
    sector_dispersion = sector_returns.var(axis=1)
    return sector_dispersion.rolling(window=window).mean()

########################################################
# 教師ラベル作成 - カンニングラベル
########################################################
def _make_label(df_daily):

    # リバランスの実行基準となるマスターカレンダー
    master_index = df_daily["^GSPC"].dropna().index
    df = pd.DataFrame(index=master_index)

    # --- カンニングデータの計算 ---
    # SP500, TLT, USD/JPY の計算（それぞれ生のまま計算してから、dfに同期）
    for col, asset_name in zip(["^GSPC", "TLT", "DEXJPUS"], ["sp500", "tlt", "usd_jpy"]):
        # NAを落として、その資産の純粋な営業日だけで未来20日を計算
        asset_clean = df_daily[col].dropna()
        future_ret = asset_clean.pct_change(20).shift(-20)
        df[f'next_20d_ret_{asset_name}'] = future_ret.reindex(master_index, method="ffill")
    #print(df.tail(50))

    # HYスプレッドは「差分(diff)」で計算する
    hy_clean = df_daily["BAMLH0A0HYM2"].dropna()
    future_diff = hy_clean.diff(20).shift(-20) # 20日後に何ポイント拡大したか
    df['next_20d_diff_hy'] = future_diff.reindex(master_index, method="ffill")

    # MAE
    sp500_clean = df_daily["^GSPC"].dropna()
    future_min_5d = sp500_clean.rolling(window=5, min_periods=1).min().shift(-5)
    df['next_5d_mae'] = (future_min_5d / sp500_clean) - 1
    df['next_5d_mae'] = df['next_5d_mae'].clip(upper=0)

    # --- レジューム分類 ---
    #df["regime"] = "5: Healthy/Neutral"
    df["regime"] = 5
    df["regime_name"] = "5: Healthy/Neutral"
    df = make_regime_label(df)
    #print(df.tail(30))

    # インデックスを有効な日付に合わせる
    df = df.reindex(df["next_20d_ret_usd_jpy"].dropna().index)
    #check_nan_time(df, "2005-01-01")

    print("\nレジューム学習の教師ラベルの期間: ",df.index[0].date(), df.index[-1].date())

    return df

def make_regime_label(df):
    # Flash Crash
    is_flash_crash = (df["next_5d_mae"] < -0.05)
    df.loc[is_flash_crash, "regime_name"] = "2: Flash Crash"
    df.loc[is_flash_crash, "regime"] = 2

    # Slow Bleed (忍耐のベア: 崖ではないが、20日で4%以上下落)
    is_slow_bleed = (df["next_20d_ret_sp500"] < -0.04) & (~is_flash_crash)
    df.loc[is_slow_bleed, "regime_name"] = "3: Slow Bleed"
    df.loc[is_slow_bleed, "regime"] = 3

    # Golden Dip (押し目)
    is_golden_dip = (df["next_20d_ret_sp500"] > 0.05) & (df["next_5d_mae"] > -0.02)
    df.loc[is_golden_dip, "regime_name"] = "1: Golden Dip"
    df.loc[is_golden_dip, "regime"] = 1

    # Liquidity In (復活の予兆)
    is_liquidity_in = (df["next_20d_ret_sp500"] > 0.03) & (~is_golden_dip) & (~is_flash_crash)
    df.loc[is_liquidity_in, "regime_name"] = "4: Liquidity In"
    df.loc[is_liquidity_in, "regime"] = 4

    return df

def evaluate_regimes(df):
    # --- 1. 統計サマリテーブルの作成 ---
    summary = df.groupby('regime').agg(
        Count=('next_20d_ret_sp500', 'count'),                           # 発生日数
        Mean_Return_20d=('next_20d_ret_sp500', 'mean'),                  # 20日リターンの平均
        Median_Return_20d=('next_20d_ret_sp500', 'median'),              # 20日リターンの中央値
        Std_Return_20d=('next_20d_ret_sp500', 'std'),                    # 20日リターンのボラティリティ(標準偏差)
        Win_Rate_20d=('next_20d_ret_sp500', lambda x: (x > 0).mean()),   # 勝率（20日後にプラスである確率）
        Mean_MAE_5d=('next_5d_mae', 'mean'),                             # 5日以内の最大逆行（ドローダウン）の平均
        Worst_MAE_5d=('next_5d_mae', 'min')                              # 5日以内の最大逆行の最悪値
    ).reset_index()
    print("\n" + "="*60)
    print("【1. 統計サマリテーブル】")
    print("="*60)
    print(summary)

    # ---  2.遷移マトリクス ---
    df['next_regime'] = df['regime'].shift(-1)
    # 行が「現在」、列が「次期」の確率分布
    transition_matrix = pd.crosstab(df['regime'], df['next_regime'], normalize='index')
    print("\n" + "="*60)
    print("【2.遷移マトリクス】")
    print("="*60)
    print(transition_matrix)

    # --- 3.平均継続日数 ---
    changes = df['regime'].ne(df['regime'].shift()).cumsum()
    durations = df.groupby(changes)['regime'].agg(['count', 'first'])
    avg_durations = durations.groupby('first')['count'].mean()
    print("\n" + "="*60)
    print("【3.平均継続日数")
    print("="*60)
    print(avg_durations)

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
