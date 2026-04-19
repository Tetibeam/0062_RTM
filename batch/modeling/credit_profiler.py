########################################################
# 市場レジュームモデリング
########################################################
from batch.modeling.learning import(
    learning_lgbm_test_driver,
    learning_lgbm_regression
    )
from batch.modeling.visualize import (
    plot_driver_soft_label,
    )
from batch.modeling.featuring import (
    get_columns_by_frequency,
    )

import pandas as pd
import numpy as np

########################################################
# メインプロセス
########################################################
credit_index = {
    # 教師ラベル
    "^GSPC": 1,
    "BAMLH0A0HYM2": 1,
    "BAA10Y":1,

    "VIXCLS":1,
    "^MOVE":1,
    "VVIX":1,
    "T10Y2Y":1,
    "DGS10":1,
    "T10YIE":1,
    }

def get_driver_beta(df_index, df_sp500):
    # --- データの取得：マスターデータからLiq_eff_modelに必要なデータを取り出す ---
    keys_list = list(credit_index.keys())
    df = df_index[keys_list]
    #check_nan_time(df, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df.tail(20))
    #print(df["BAMLH0A0HYM2"].dropna().head(10))
    #print(df["BAA10Y"].dropna().head(10))

    # --- 市場レジームの教師ラベル --
    df_label = _make_label(df[["^GSPC","BAA10Y"]])
    #check_nan_time(df_label, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_label.tail(50))


    # --- データ集計：日時、週次、月次を、すべて日次にする ---
    df_agg_daily = _aggregation_daily(df)
    #check_nan_time(df_agg_daily, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #pd.set_option("display.max_columns", None)
    #print(df_agg_daily.tail(50))

    # --- 前処理（特徴量） ---
    df_features = _featuring_all(df_agg_daily)
    
    #check_nan_time(df_features, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_features.tail(20))

    features_refined = {
        "BAA_Mom_5d":0,
        "BAA_Mom_20d":0,
        "VIX_Accel":0,
        "MOVE_Accel":0,
        "VIX_Mom_20d":0,
        "MOVE_Mom_20d":0,
        "Combined_Shock":0,
        "VIX_Cum_Stress_20d":0,
        "MOVE_Cum_Stress_20d":0,
        #"BAA_Days_Above_SMA60":0,
        #"YC_Inverted_Days_252d":0,
        "T10YIE_MA500_Gap":0,
        "DGS10_MA500_Gap":0,
        "Era":0,
    }
    df_features = df_features[list(features_refined.keys())]
    monotone_constraints = list(features_refined.values())
    #print(monotone_constraints)

    df_credit = df_features.join(df_label[["target_score", "next_diff_hy"]])
    start = df_credit.apply(pd.Series.first_valid_index).max()
    end = df_credit.apply(pd.Series.last_valid_index).min()
    df_credit = df_credit.loc[start:end]
    df_credit = df_credit.loc["2007-01-01":]
    #check_nan_time(df_credit, date="2005-01-01")
    #pd.set_option("display.max_rows", None)
    #print(df_driver.tail(20))

    print(f"特徴量: {df_features.columns}")
    df_oof_all, df_shap, df_oof_ev = learning_lgbm_regression(
        df_credit, target_col="target_score", 
        n_splits=5, gap =50,
        n_estimators=1000,learning_rate=0.001,
        num_leaves=15, min_data_in_leaf=7,max_depth=3,
        reg_alpha=2, reg_lambda=2,
        extra_trees="False",
        importance_type='gain',
        learning_curve=True,
        )
    #shap_stats(df_driver, df_features.columns, df_shap)

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

def _aggregation_daily(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")]
    df_weekly = df[get_columns_by_frequency(df, target="weekly")]
    df_monthly = df[get_columns_by_frequency(df, target="monthly")]
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")]

    print(f"--- 特徴量や目的変数で使う指標の頻度 ---")
    print(f"日次: {df_daily.columns.tolist()}")
    print(f"週次: {df_weekly.columns.tolist()}")
    print(f"月次: {df_monthly.columns.tolist()}")
    print(f"四半期: {df_quarterly.columns.tolist()}\n")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.dropna(how="all").tail(10))

    master_index = df["^GSPC"].dropna().index

    # 日次>日次
    lagged_series_list = []
    for col in df_daily.columns:
        # オリジナル
        s = df_daily[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = credit_index.get(col, 2)
        s.index = s.index + pd.Timedelta(days=lag_days)
        # 日次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_daily_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))

    # 週次>週次
    """lagged_series_list = []
    for col in df_weekly.columns:
        # オリジナル
        s = df_weekly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = credit_index.get(col, 7)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 週次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_weekly_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))"""

    # 月次>週次
    """df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)
    lagged_series_list = []
    for col in df_monthly.columns:
        # オリジナル
        s = df_monthly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = credit_index.get(col, 31)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 月次>週次
        s_d = s.reindex(master_index, method="ffill")
        #print(s_d.tail(20))
        lagged_series_list.append(s_d)
    df_monthly_d_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_monthly_w_lagged,"1990-01-01")
    #print(df_monthly_w_lagged.tail(20))

    # 結合
    df_combine = pd.concat([df_daily_d_lagged, df_weekly_d_lagged, df_monthly_d_lagged], axis=1)
    #check_nan_time(df_combine,"1990-01-01")"""

    df_combine = df_daily_d_lagged
    return df_combine.dropna(how="all")

def _aggregation_weekly(df):

    df_daily = df[get_columns_by_frequency(df, target="daily")]
    df_weekly = df[get_columns_by_frequency(df, target="weekly")]
    df_monthly = df[get_columns_by_frequency(df, target="monthly")]
    df_quarterly = df[get_columns_by_frequency(df, target="quarterly")]

    print(f"--- 特徴量や目的変数で使う指標の頻度 ---")
    print(f"日次: {df_daily.columns.tolist()}")
    print(f"週次: {df_weekly.columns.tolist()}")
    print(f"月次: {df_monthly.columns.tolist()}")
    print(f"四半期: {df_quarterly.columns.tolist()}\n")
    #pd.set_option("display.max_rows", None)
    #print(df_monthly.dropna(how="all").tail(10))

    # 日次>週次
    lagged_series_list = []
    for col in df_daily.columns:
        # オリジナル
        s = df_daily[col].dropna().copy()
        # ラグ
        lag_days = driver_index.get(col, 2)
        s.index = s.index + pd.Timedelta(days=lag_days)
        # 日次>週次
        s_w = s.resample("W-FRI").mean()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_daily_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_daily_w_lagged,"1990-01-01")
    #print(df_daily_w_lagged.tail(20))

    # 週次>週次
    lagged_series_list = []
    for col in df_weekly.columns:
        # オリジナル
        s = df_weekly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 7)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 週次>週次
        s_w = s.resample("W-FRI").ffill()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_weekly_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_weekly_w_lagged,"1990-01-01")
    #print(df_weekly_w_lagged.tail(20))

    # 月次>週次
    df_monthly.index = df_monthly.index + pd.offsets.MonthEnd(0)
    lagged_series_list = []
    for col in df_monthly.columns:
        # オリジナル
        s = df_monthly[col].dropna().copy()
        #print(s.tail(20))
        # ラグ
        lag_days = driver_index.get(col, 31)
        s.index = s.index + pd.Timedelta(days=lag_days)
        s = s.dropna()
        # 月次>週次
        s_w = s.resample("W-FRI").ffill()
        #print(s_w.tail(20))
        lagged_series_list.append(s_w)
    df_monthly_w_lagged = pd.concat(lagged_series_list, axis=1)
    #check_nan_time(df_monthly_w_lagged,"1990-01-01")
    #print(df_monthly_w_lagged.tail(20))

    # 結合
    df_combine = pd.concat([df_daily_w_lagged, df_weekly_w_lagged, df_monthly_w_lagged], axis=1)
    #check_nan_time(df_combine,"1990-01-01")

    return df_combine.dropna(how="all")

########################################################
# 特徴量抽出
########################################################

def _featuring_all(df_daily):

    feats = pd.DataFrame(index=df_daily.index)
    master_index = df_daily["^GSPC"].dropna().index

    # 「物理的な衝撃波」
    feats['BAA_Mom_5d'] = df_daily['BAA10Y'].diff(5)
    feats['BAA_Mom_20d'] = df_daily['BAA10Y'].diff(20)

    for col, name in zip(['VIXCLS', '^MOVE'], ['VIX', 'MOVE']):
        series = df_daily[col]
        sma20 = series.rolling(window=20).mean()
        std252 = series.rolling(window=252).std()

        # 加速度の算出
        feats[f'{name}_Accel'] = (series - sma20) / std252
        # 比較用：シンプルな20日モメンタム（変化率）も追加
        feats[f'{name}_Mom_20d'] = series.pct_change(20)
    # 債券と株の両方が同時にパニックになっている度合い
    feats['Combined_Shock'] = feats['VIX_Accel'] + feats['MOVE_Accel']

    # A. VIX & MOVE の累積ストレス (20日間)
    # 閾値の例として、過去252日の中央値（または固定値 20 や 100 など）を使用
    vix_threshold = df_daily['VIXCLS'].rolling(252).median()
    move_threshold = df_daily['^MOVE'].rolling(252).median()

    # 閾値を超えた分（超過ストレス）だけを抽出し、20日間で合計（積分）する
    feats['VIX_Cum_Stress_20d'] = np.maximum(df_daily['VIXCLS'] - vix_threshold, 0).rolling(20).sum()
    feats['MOVE_Cum_Stress_20d'] = np.maximum(df_daily['^MOVE'] - move_threshold, 0).rolling(20).sum()

    # B. BAA10Yの「茹でガエル」滞空時間 (過去60日のうち、SMA60より上にいた日数)
    baa_sma60 = df_daily['BAA10Y'].rolling(60).mean()
    is_above_sma = (df_daily['BAA10Y'] > baa_sma60).astype(int)
    feats['BAA_Days_Above_SMA60'] = is_above_sma.rolling(60).sum()
    
    # C. イールドカーブ逆転の滞留日数 (T10Y2Y が 0 未満の日数を過去1年でカウント)
    is_inverted = (df_daily['T10Y2Y'] < 0).astype(int)
    feats['YC_Inverted_Days_252d'] = is_inverted.rolling(252).sum()
    
    
    # T10YIE (実質金利/インフレ期待) の MA500乖離
    feats['T10YIE_MA500_Gap'] = df_daily['T10YIE'] - df_daily['T10YIE'].rolling(500).mean()

    # または、シンプルに10年金利のMA500乖離でもOK
    feats['DGS10_MA500_Gap'] = df_daily['DGS10'] - df_daily['DGS10'].rolling(500).mean()


    # --- Era ---
    conditions = [
        (feats.index < '2010-10-01'),
        (feats.index >= '2010-10-01') & (feats.index < '2013-06-01'),
        (feats.index >= '2013-06-01') & (feats.index < '2019-09-01'),
        (feats.index >= '2019-09-01') & (feats.index < '2020-04-01'),
        (feats.index >= '2020-04-01') & (feats.index < '2021-12-01'),
        (feats.index >= '2021-12-01') & (feats.index < '2023-10-01'),
        (feats.index >= '2023-10-01')
    ]
    choices = [0, 1, 2, 3, 4, 5, 6]

    # 2. 条件に合致しない場合は最新のEra4とする
    feats['Era'] = np.select(conditions, choices, default=4)

    # 3. 【超重要】数値型からカテゴリ型へ明示的に変換
    feats['Era'] = feats['Era'].astype('category')


    # 開始日、終了日をを決める
    start = feats.apply(pd.Series.first_valid_index).max()
    end = feats.apply(pd.Series.last_valid_index).min()
    feats = feats.loc[start:end]

    check_nan_time(feats, date="1900-01-01")

    return feats

def _featuring_z_score(df, window):

    m = df.rolling(window=window, min_periods=max(10, window//5)).mean()
    s = df.rolling(window=window, min_periods=max(10, window//5)).std()

    z = (df - m) / (s + 1e-9)# ゼロ除算防止

    return z.clip(-5, 5)

########################################################
# 教師ラベル作成 - カンニングラベル
########################################################

def _make_label(df, smear_days=10, future_lag=20):
    pd.set_option("display.max_rows", None)

    # 1. 基準となるインデックスの作成
    master_index = df["^GSPC"].dropna().index
    df_label = pd.DataFrame(index=master_index)
    df_label["next_return_sp500"] = df["^GSPC"].pct_change(future_lag).shift(-future_lag).ffill()


    # 2. クレジット・スプレッドの準備 (BAA10Yを使用)
    # ※データが欠損している場合に備え、前方埋め処理
    #print(df["BAA10Y"].isna().sum())
    hy_clean = df["BAA10Y"].ffill()
    #print(hy_clean.isna().sum())
    #print(hy_clean.tail(20))
    #print(hy_clean.head(20))

    # --- Step 1: 閾値（現在の常識）の計算 ---
    # 過去252日（1年）のボラティリティを基準に、40日間の拡大幅の「異常さ」を定義
    hy_diff_vol = hy_clean.diff(future_lag).rolling(252, min_periods=60).std().reindex(master_index, method='ffill')
    #print(hy_diff_vol.isna().sum())

    # --- Step 2: 未来の事実（ターゲット）の計算 ---
    # 40日後のスプレッドが、今よりどれだけ拡大しているか
    future_hy_diff = hy_clean.diff(future_lag).shift(-future_lag).reindex(master_index, method="ffill")
    #print(future_hy_diff.isna().sum())
    df_label['next_diff_hy'] = future_hy_diff

    # --- Step 3: 生のフラグ（Raw Flags）を立てる ---
    # スプレッドが 1.75シグマ 以上拡大した瞬間を「事件のピーク」とする
    raw_credit_event = (df_label['next_diff_hy'] > (1.75 * hy_diff_vol))

    # --- Step 4: 数学的Smearing（減衰スコアの計算） ---
    def calculate_decay_score(raw_series, window):
        scores = np.zeros(len(raw_series))
        event_indices = np.where(raw_series)[0]

        # windowを少し長めに設定（事件の10日前から予兆を教え込む）
        # tau（半減期）を調整して、不穏さのカーブを作る
        tau = window / 1.5
        for idx in event_indices:
            for d in range(window + 1):
                if idx - d >= 0:
                    # 指数減衰により、事件当日に向かって数値が 0.0 -> 1.0 に近づく
                    decay_val = np.exp(-d / tau)
                    scores[idx - d] = max(scores[idx - d], decay_val)
        return scores

    # これが回帰モデルの学習ターゲットになる
    df_label['target_score'] = calculate_decay_score(raw_credit_event, smear_days)
    #print(df_label.tail(400))

    # 学習に不要な中間カラムを削除し、ターゲットと解析用データのみ残す
    # (後で分析できるように next_diff_hy は残しておくのが吉)
    df_label = df_label.dropna()

    #_analysis_label(df_label)

    return df_label

def _analysis_label(df):
    df_origin = df.copy()
    terms = [
        ("2010-10-01","2013-06-01"),("2013-06-01","2016-10-01"),("2016-10-01","2019-09-01"),
        ("2019-09-01","2020-04-01"),("2020-04-01","2021-12-01"),("2021-12-01","2023-10-01"),
        ("2023-10-01","2026-02-01"),
        #("2010-10-01","2021-12-01"),("2021-12-01","2026-02-01"),
        #("2010-10-01","2026-02-01"),
        ]
    for start,end in terms:
        print(f"\nCredit Profiler 教師ラベルの期間 : {start}〜{end}")

        df_sub = df_origin.loc[start:end]

        # 分析・可視化
        df_sub["hy_flg"] = df_sub["target_score"] >0
        stats = df_sub["hy_flg"].value_counts().to_frame(name='Count')
        stats['Percentage (%)'] = (df_sub["hy_flg"].value_counts(normalize=True) * 100).round(2)
        print(stats)

        market_summary = df_sub.groupby('hy_flg').agg({
            'next_return_sp500': [
                'count', 'mean', 'median', 'std', 
                lambda x: x.quantile(0.05), 'min', # 下位5%と最小値でリスクの深さを測る
                lambda x: (x > 0).mean() # 勝率
            ],
            'next_diff_hy': ['mean', 'std', 'max'] # HYは拡大(max)がリスク
        }).round(4)

        # カラム名を分かりやすく整理（任意）
        market_summary.columns = [
            'count', 'ret_mean', 'ret_median', 'ret_std', 
            'ret_q05', 'ret_min', 'win_rate',
            'hy_diff_mean', 'hy_diff_std', 'hy_diff_max'
        ]
        print(market_summary)

        # 継続日数の算出
        df_sub['change'] = df_sub['hy_flg'] != df_sub['hy_flg'].shift()
        df_sub['regime_id'] = df_sub['change'].cumsum()

        # 各期間の長さをカウント
        duration_stats = df_sub.groupby(['regime_id', 'hy_flg']).size().reset_index(name='duration')
        avg_duration = duration_stats.groupby('hy_flg')['duration'].mean().round(1)
        print(f"平均継続日数:\n{avg_duration}")

        # 遷移マトリクス（現在の状態 -> 次の状態）
        transition_matrix = pd.crosstab(
            df_sub['hy_flg'], 
            df_sub['hy_flg'].shift(-1), 
            normalize='index'
        ).round(2)

        print("遷移マトリクス（行：現在 -> 列：次）:")
        print(transition_matrix)
        #plot_driver_soft_label(df, df_daily, start_date=s_date, end_date=e_date)

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
    print("=== risk_score の基本統計量 ===")
    print(df_oof_ev['risk_score'].describe())

    plt.figure(figsize=(10, 6))
    sns.histplot(df_oof_ev['risk_score'], bins=50, kde=True, color='royalblue')
    plt.title('Distribution of Risk Sum (Credit + 0.5*Bond Probability)')
    plt.xlabel('Risk Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.3)
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    plt.legend()
    plt.show()

def shap_stats(df_master, features_list, df_shap):
    for label, shap_df in df_shap.items():
        print(f"\n=== レジーム: {label} の符号検証 ===")
        # 検証データ期間の元の特徴量を取得
        original_X = df_master.loc[shap_df.index, features_list]

        logic_results = []
        for col in features_list:
            # 元の値とSHAP値の相関を計算
            correlation = original_X[col].corr(shap_df[col])

            # 方向性の判定
            direction = "正の相関 (+)" if correlation > 0 else "負の相関 (-)"
            logic_results.append({
                "特徴量": col,
                "方向性": direction,
                "相関係数": f"{correlation:.3f}"
            })

        print(pd.DataFrame(logic_results))
