########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test,
    learning_lgbm_final,
    explain_prediction,
    )
from batch.modeling.visualize import (
    plot_market_navigator,
    plot_driver_trajectory,
    plot_shap_explanation,
    plot_factor_label,
    plot_index
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

    # --- 学習モデル生成 ---

    # Driver Profiler
    df_driver = df_features.join(df_label["driver"])
    #driver_clf, df_driver_trajectory = learning_lgbm_final(
    #    df_driver, "driver", model_name="Driver", label_name_list=["1:Credit", "2:Bond", "3:Equity", "4:Mix"],
    #    n_estimators=1000,learning_rate=0.01,num_leaves=30, min_data_in_leaf=50,
    #    reg_alpha=0.5, reg_lambda=0.5,

    df_oof_all = learning_lgbm_test(
        df_driver, "driver", labels=["1:Credit", "2:Bond", "3:Equity", "4:Mix"],
        n_splits=5, gap =20,
        n_estimators=1000,learning_rate=0.01,num_leaves=30, min_data_in_leaf=50,
        reg_alpha=0.5, reg_lambda=0.5,
        )

    #return driver_clf, df_driver_trajectory, df_driver
    return df_oof_all


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
    dxy = df_daily["DX=F"].dropna()
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

    # 指標
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

    # --- Driver 教師ラベル ---
    next_20d_ret_sp500_abs = df['next_20d_ret_sp500'].abs()

    df['driver_name'] = 'Neutral'
    df['driver'] = 4  # または仕様に合わせたデフォルト値

    #【優先度中】債券: TLTの変動がSP500の半分以上なら (2022年はこれが主役になる)
    df.loc[df['next_20d_ret_tlt'].abs() > (next_20d_ret_sp500_abs * 0.5), 'driver_name'] = 'Bond'
    df.loc[df['next_20d_ret_tlt'].abs() > (next_20d_ret_sp500_abs * 0.5), 'driver'] = 2

    #【優先度高】信用: スプレッドが20日で0.5ポイント(50bps)以上「拡大」したら本当の危機
    df.loc[df['next_20d_diff_hy'].abs() > 0.5, 'driver_name'] = 'Credit'
    df.loc[df['next_20d_diff_hy'].abs() > 0.5, 'driver'] = 1

    no_external_driver = (df['driver_name'] == 'Neutral')
    is_equity_move = next_20d_ret_sp500_abs > 0.03
    df.loc[no_external_driver & is_equity_move, 'driver_name'] = 'Equity'
    df.loc[no_external_driver & is_equity_move, 'driver'] = 3


    # インデックスを有効な日付に合わせる
    df = df.reindex(df["next_20d_diff_hy"].dropna().index)
    #check_nan_time(df, "2005-01-01")

    # 表示・分析・評価

    print("\nDriver教師ラベルの期間: ",df.index[0].date(), df.index[-1].date())

    return df

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
