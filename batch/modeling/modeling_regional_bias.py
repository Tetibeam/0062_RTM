########################################################
# 地域バイアスモデル　- 日本
########################################################
from batch.modeling.learning import(
    learning_lgbm_test,
    explain_prediction
    )
from batch.modeling.visualize import(
    plot_regional_bias_trajectory,
    plot_shap_explanation
    )
import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################
def get_regional_bias_model_beta(df_index,regime_clf,df_regime_features, df_nikkei,df_sp500):
    df_daily = df_index.copy()

    # --- 市場レジームの教師ラベル --
    df_j_label = _make_j_label(df_daily)

    # --- 前処理（特徴量） ---
    df_j_features = _featuring_j(df_daily, regime_clf,df_regime_features,df_nikkei,df_sp500)

    # --- 学習モデル生成 ---

    # 日本バイアス
    df_j_bias = df_j_features.join(df_j_label["japan_bias_label"])
    #j_bias_clf, df_j_bias_trajectory = learning_j_bias_final(df_j_bias)
    learning_lgbm_test(
        df_j_bias, "japan_bias_label",
        n_splits=5, gap =20,
        n_estimators=500,learning_rate=0.01,num_leaves=31, min_data_in_leaf=50,
        reg_alpha=0.6, reg_lambda=0.6,
        learning_curve=True,
        )

    #run_regional_bias_backtest(
    #    df_j_bias_trajectory,
    #    df_daily["^N225"].dropna().pct_change(fill_method=None),
    #    df_daily["^GSPC"].dropna().pct_change(fill_method=None)
    #)

def run_regional_bias_backtest(df_trajectory, n225_ret, sp500_ret, 
                               threshold_c1=0.50, threshold_c2=0.35, 
                               cost_rate=0.0005): # コスト 0.05%
    """
    RTM 地域配分戦略のバックテストを実行
    """
    # 1. データの結合とラグの考慮
    # 判定が出た翌日の寄り付きでリバランスすると仮定し、ウェイトを1日シフトする
    df_bt = df_trajectory.copy()
    df_bt['n225_ret'] = n225_ret
    df_bt['sp500_ret'] = sp500_ret

    # 2. 戦略ウェイトの決定 (Modulation)
    # デフォルトは 50:50
    df_bt['w_jp'] = 0.5
    df_bt['w_us'] = 0.5

    # Overdrive (C1) -> 日本株に傾斜 (70:30)
    df_bt.loc[df_bt['1: JP Overdrive'] >= threshold_c1, ['w_jp', 'w_us']] = [0.7, 0.3]

    # Fragile (C2) -> 米国株へ全退避 (0:100)
    df_bt.loc[df_bt['2: JP Fragile'] >= threshold_c2, ['w_jp', 'w_us']] = [0.0, 1.0]

    # 【重要】シグナルは当日終値で出るため、実際の適用（リターン享受）は翌日から
    df_bt['w_jp_exec'] = df_bt['w_jp'].shift(1)
    df_bt['w_us_exec'] = df_bt['w_us'].shift(1)

    # 3. リターン計算
    # 戦略リターン = (前日のウェイト * 今日のリターン)
    df_bt['strat_ret_raw'] = (df_bt['w_jp_exec'] * df_bt['n225_ret'] + 
                              df_bt['w_us_exec'] * df_bt['sp500_ret'])

    # 取引コストの算出 (ウェイトの変化量 × コスト率)
    df_bt['turnover'] = df_bt['w_jp_exec'].diff().abs() + df_bt['w_us_exec'].diff().abs()
    df_bt['cost'] = df_bt['turnover'] * cost_rate
    df_bt['strat_ret'] = df_bt['strat_ret_raw'] - df_bt['cost'].fillna(0)

    # ベンチマーク（50:50固定リバランスなし）
    df_bt['bench_ret'] = 0.5 * df_bt['n225_ret'] + 0.5 * df_bt['sp500_ret']

    # 4. 累積リターンの算出
    df_bt['cum_strat'] = (1 + df_bt['strat_ret'].fillna(0)).cumprod()
    df_bt['cum_bench'] = (1 + df_bt['bench_ret'].fillna(0)).cumprod()
    df_bt['cum_sp500'] = (1 + df_bt['sp500_ret'].fillna(0)).cumprod()

    # 5. パフォーマンス指標の計算
    def get_stats(rets, cum):
        ann_ret = (cum.iloc[-1] ** (252 / len(rets))) - 1
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol
        mdd = (cum / cum.cummax() - 1).min()
        return ann_ret, ann_vol, sharpe, mdd

    stats = {
        "RTM Strategy": get_stats(df_bt['strat_ret'], df_bt['cum_strat']),
        "Benchmark (50:50)": get_stats(df_bt['bench_ret'], df_bt['cum_bench']),
        "S&P 500 Only": get_stats(df_bt['sp500_ret'], df_bt['cum_sp500'])
    }

    # 6. 結果の表示
    print("\n" + "="*50)
    print("      BACKTEST PERFORMANCE REPORT")
    print("="*50)
    print(f"{'Strategy':<20} | {'Return':>7} | {'Sharpe':>7} | {'MDD':>7}")
    print("-" * 50)
    for name, (r, v, s, m) in stats.items():
        print(f"{name:<20} | {r:>7.1%} | {s:>7.2f} | {m:>7.1%}")
    print("="*50)

    return df_bt, stats

########################################################
# 特徴量抽出
########################################################
def _featuring_j(df_daily, regime_clf,df_regime_features, df_nikkei, df_sp500):

    # --- リバランスの実行基準となるマスターカレンダー ---
    master_index = df_daily["^GSPC"].dropna().index

    # --- 指標の取得 ---
    sp500 = df_daily["^GSPC"].dropna()
    nikkei = df_daily["^N225"].dropna()
    usd_jpy = df_daily["DEXJPUS"].dropna()
    hy = df_daily['BAMLH0A0HYM2'].dropna()
    oil= df_daily['CL=F'].dropna()
    sox = df_daily["^SOX"].dropna()
    topix = df_daily["topix"].dropna()
    nasdaq = df_daily["^IXIC"].dropna()

    df = pd.DataFrame(index=master_index)

    # 20日間の日米リターン差
    n225_ret_20d = nikkei.reindex(master_index, method='ffill').pct_change(20, fill_method=None)
    sp500_ret_20d = sp500.pct_change(20, fill_method=None).reindex(master_index, method='ffill')

    df["momentum_diff_20d"] = (n225_ret_20d - sp500_ret_20d)

    # 日経とドル円の20日相関
    n225_ret = nikkei.reindex(master_index, method='ffill').pct_change(fill_method=None)
    usdjpy_ret = usd_jpy.reindex(master_index, method='ffill').pct_change(fill_method=None)
    df["corr_n225_usdjpy_20d"] = n225_ret.rolling(window=20).corr(usdjpy_ret)

    # 悪い円安（エネルギーコスト増）センサー
    jpy_crude_oil = oil.reindex(master_index, method='ffill') * usd_jpy.reindex(master_index, method='ffill')
    df["jpy_crude_oil_ret_20d"] = jpy_crude_oil.pct_change(20, fill_method=None)

    # 半導体サイクル・センサー (対S&P500相対モメンタム)
    sox_ret_20d = sox.reindex(master_index, method='ffill').pct_change(20, fill_method=None)
    df["sox_vs_sp500_momentum_20d"] = sox_ret_20d - sp500_ret_20d

    # 信用収縮の「突発的なパニック（初動）」を捉えるための5日間変化幅
    df["us_hy_spread_diff_5d"] = hy.reindex(master_index, method='ffill').diff(5)

    # 20日間のヒストリカル・ボラティリティ (年率換算なしの生データ)
    df["n225_vol_20d"] = n225_ret.rolling(20).std()
    df["sp500_vol_20d"] = sp500.pct_change(fill_method=None).rolling(20).std()

    # 日本市場の「歪み」 (NT倍率)
    nt_ratio = nikkei.reindex(master_index, method='ffill') / topix.reindex(master_index, method='ffill')
    df["nt_ratio_mom_20d"] = nt_ratio.pct_change(20, fill_method=None)

    # 日本市場の「歪み」の差 (breadthの差)
    df_nikkei = df_nikkei.reindex(master_index)
    df_sp500 = df_sp500.reindex(master_index)
    df["nikkei_breadth_diff_20d"] = _featuring_m_breadth(df_nikkei, window=20)
    jp_breadth = _featuring_m_breadth(df_nikkei, window=20)
    us_breadth = _featuring_m_breadth(df_sp500, window=20)
    df["breadth_diff_sp500_nikkei"] = jp_breadth - us_breadth

    df["nasdaq_ret_20d"] = nasdaq.pct_change(20,fill_method=None).reindex(master_index)

    # Regime Prismのレジューム確率
    global_probs = regime_clf.predict_proba(df_regime_features)

    df_global_probs = pd.DataFrame(
        global_probs,
        index=df_regime_features.index,
        columns=['prob_regime_1', 'prob_regime_2', 'prob_regime_3', 'prob_regime_4', 'prob_regime_5']
    )
    df_global_probs = df_global_probs[['prob_regime_1', 'prob_regime_2', 'prob_regime_3', 'prob_regime_4']]

    df_all = pd.concat([df, df_global_probs], axis=1)

    # 未来リーク耐性
    df_all = df_all.shift(1).dropna(how="all")

    # --- 全特徴量の共通有効期間 ---
    start = df_all.apply(pd.Series.first_valid_index).max()
    #start = df_all["tlt_ret_20d"].first_valid_index()
    end   = df_all.apply(pd.Series.last_valid_index).min()
    print("\n日本バイアスの特徴量の期間: ",start, end)

    df_all = df_all.loc[start:end].ffill()

    return df_all

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

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
def _make_j_label(df_daily):

    # リバランスの実行基準となるマスターカレンダー
    master_index = df_daily["^GSPC"].dropna().index
    df = pd.DataFrame(index=master_index)

    # --- カンニングデータの計算 ---
    # SP500, NIKKEI の計算（それぞれ生のまま計算してから、dfに同期）
    for col, asset_name in zip(["^GSPC", "^N225"], ["sp500", "nikkei"]):
        # NAを落として、その資産の純粋な営業日だけで未来20日を計算
        asset_clean = df_daily[col].dropna()
        future_ret = asset_clean.pct_change(20).shift(-20)
        df[f'next_20d_ret_{asset_name}'] = future_ret.reindex(master_index, method="ffill")

    # 相対リターンの計算 (日本 - 米国)
    df['ret_20d_nikkei_minus_sp500'] = df['next_20d_ret_nikkei'] - df['next_20d_ret_sp500']


    # 相対MAE (日本 - 米国)
    sp500_clean = df_daily["^GSPC"].dropna()
    nikkei_clean = df_daily["^N225"].dropna()
    sp500_future_min_5d = sp500_clean.rolling(window=5, min_periods=1).min().shift(-5)
    nikkei_future_min_5d = nikkei_clean.rolling(window=5, min_periods=1).min().shift(-5)
    df['next_5d_mae_sp500'] = (sp500_future_min_5d / sp500_clean) - 1
    df['next_5d_mae_sp500'] = df['next_5d_mae_sp500'].clip(upper=0).reindex(master_index, method="ffill")
    df['next_5d_mae_nikkei'] = (nikkei_future_min_5d / nikkei_clean) - 1
    df['next_5d_mae_nikkei'] = df['next_5d_mae_nikkei'].clip(upper=0).ffill()

    df['mae_5d_nikkei_minus_sp500'] = (df['next_5d_mae_nikkei'] - df['next_5d_mae_sp500'])

    # --- クラス分類ロジックの適用 ---
    conditions = [
        # 1: JP Overdrive (日本が独自に3%以上強い)
        (df['ret_20d_nikkei_minus_sp500'] >= 0.02),

        # 2: JP Fragile (米国急落時に日本が2%以上余計に掘る)
        (df['mae_5d_nikkei_minus_sp500'] <= -0.02),
    ]
    choices = [1, 2]

    # デフォルトは 3: Synchronized
    df['japan_bias_label'] = np.select(conditions, choices, default=3)

    #check_nan_time(df, "2005-01-01")

    start = df.apply(pd.Series.first_valid_index).max()
    end   = df.apply(pd.Series.last_valid_index).min()
    print("\n日本バイアス 教師ラベルの期間: ",start, end)
    df = df.loc[start:end].ffill()

    #pd.set_option("display.max.row", None)
    #print(df["japan_bias_label"].value_counts(normalize=True))
    #print(df.loc["2023-01-01":,"japan_bias_label"]
    #check_nan_time(df, "2005-01-01"))

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

