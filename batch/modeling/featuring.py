import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy.stats.mstats import winsorize


############################################################
# データ集計・整理
############################################################
def get_columns_by_frequency(
    df,
    target="daily",
    min_obs=10,
    thresholds=(2, 10, 40)
):
    daily_max, weekly_max, monthly_max = thresholds
    result = []

    for col in df.columns:
        s = df[col].dropna()

        if len(s) < min_obs:
            continue

        deltas = s.index.to_series().diff().dt.days.dropna()
        median_days = deltas.median()

        freq = (
            "daily" if median_days <= daily_max else
            "weekly" if median_days <= weekly_max else
            "monthly" if median_days <= monthly_max else
            "quarterly"
        )

        if freq == target:
            result.append(col)

    return result

############################################################
# データコンディショニング
############################################################

# スパイク等外れ値のキャッピング
def cap_outliers(series: pd.Series, lower=0.01, upper=0.01) -> pd.Series:
    w = winsorize(series.values, limits=[lower, upper])
    return pd.Series(w, index=series.index, name=series.name)

def cap_by_sigma(series: pd.Series, sigma: float = 3.0) -> pd.Series:

    if series.isnull().all():
        return series

    mean = series.mean()
    std = series.std()

    lower_bound = mean - sigma * std
    upper_bound = mean + sigma * std

    # クリッピング実行
    capped_series = series.clip(lower=lower_bound, upper=upper_bound)

    # 処理のログ（どれくらい叩いたか）
    count = (series > upper_bound).sum() + (series < lower_bound).sum()
    if count > 0:
        print(f"   [Capping] {series.name}: {count} points clipped at {sigma} sigma.")

    return capped_series

def balanced_clip(series: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """
    パーセンタイルに基づいたバランス型のクリッピング。
    上下1%（最も極端な数点）だけを、その次の異常値レベルまで引き戻す。
    """
    lower_bound = series.quantile(lower_q)
    upper_bound = series.quantile(upper_q)

    # ログ出力（何が起きたかを確認するため）
    clipped_up = (series > upper_bound).sum()
    clipped_low = (series < lower_bound).sum()

    if clipped_up + clipped_low > 0:
        print(f"   [Balanced Clip] {series.name}: clipped at {lower_q*100}%/{upper_q*100}% "
              f"({clipped_low}/{clipped_up} points)")

    return series.clip(lower=lower_bound, upper=upper_bound)
# 特定のイベント期間の影響をダミー変数回帰で除去し、残差（真のトレンド）を返す
# event_periods (list of tuples): 異常値期間のリスト [(start, end), ...
def neutralize_event_impact(series, event_periods):

    # ダミー変数の作成
    dummy = pd.Series(0, index=series.index, name='event_dummy')
    for start, end in event_periods:
        dummy.loc[start:end] = 1

    # 回帰分析（ y = alpha + beta * dummy ）の準備
    # 欠損値を除外して計算
    valid_mask = series.notna()
    y = series[valid_mask]
    X = dummy[valid_mask]
    X = sm.add_constant(X)
    # モデル適合
    model = sm.OLS(y, X).fit()

    # 残差の計算
    # 観測値全体から「beta * dummy」を差し引く（alpha（定数項）は維持する設計）
    # ※model.residは(y - pred)なので、純粋な変動分（alpha含む）を残すなら以下
    beta = model.params['event_dummy']
    adjusted_series = series - (beta * dummy)

    # ログ出力
    print(f"--- Series Adjustment: {series.name if series.name else 'Unnamed'} ---")
    print(f"Detected Spike Impact (Beta): {beta:.4f}")
    print(f"R-squared: {model.rsquared:.4f}")

    # 元のSeriesの名称を維持して返す
    adjusted_series.name = f"{series.name}_adj" if series.name else "adj_series"
    return adjusted_series


############################################################
# 特徴量エンジニアリング
############################################################

# Zスコア化
def standard_scalar(series: pd.Series):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(series.to_frame())
    return pd.Series(
        scaled_values.flatten(),
        index=series.index,
        name=series.name
    )

def standard_scalar_df(df: pd.DataFrame):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df)
    return pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
