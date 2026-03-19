import pandas as pd
import numpy as np

# -----データ取得-----
def get_index_data_set():

    df_index, df_sp500 = get_index_for_regime(months=240)

    return df_index, df_sp500

# -----集計・計算-----
def get_macro_quarterly_now_casting(df_index, df_sp500):

    get_market_regime_modeling_beta(df_index, df_sp500)

def get_regime_full_score(df_index,df_macro_monthly,df_macro_quarterly):
    # regime feature
    df_features = get_regime_features(df_index)
    
    # macro feature
    df_macro = get_macro_tag(df_macro)
    df_macro = df_macro.reindex(df_features.index, method="ffill")
    
    # japan bias feature
    df_j_bias_features = get_j_bias_features(df_j_bias)
    df_j_bias_features = df_j_bias_features.reindex(df_features.index, method="ffill")

    # europe bias feature
    df_eu_bias_features = get_eu_bias_features(df_eu_bias)
    df_eu_bias_features = df_eu_bias_features.reindex(df_features.index, method="ffill")

    # em bias feature
    df_em_bias_features = get_em_bias_features(df_em_bias)
    pd.set_option('display.max_rows', None)
    
    df_features = pd.concat([df_features, df_macro, df_j_bias_features], axis=1)

    df_regime = get_regime_score(df_features)
    return df_regime

def get_regime_snapshot_data_set(df_regime):
    cur = df_regime.iloc[-1]

    return [
        ["Regime", cur["Regime"]],
        ["Raw Regime", cur["RawRegime"]],
        ["Regime Score", f"{cur['RegimeScore']:.2f}"],
        ["Regime Confidence", f"{cur['RegimeConfidence']:.2f}"],
        ["Policy Credibility", f"{cur['PolicyCredibility']:.2f}"],
        ["Inflation State", cur["inflation_state"]],
        ["Policy Stance", cur["policy_stance"]],
        ["Pre-Crisis Level", int(cur["PreCrisis"])],
        ["Accel Warning", bool(cur["AccelWarning"])],
        ["Policy Shock", bool(cur["PolicyShock"])],
        ["Market Driver", cur["Driver"]],
        ["Equity Contribution", f"{cur['Equity_contrib']:.2f}"],
        ["Bond Contribution", f"{cur['Bond_contrib']:.2f}"],
        ["Japan Bias Score", f"{cur['JapanBiasScore']:.2f}"],
        ["nikkei relative score", f"{cur['nikkei_relative_score']:.2f}"],
        ["jpy trend score", f"{cur['jpy_trend_score']:.2f}"],
        ["boj policy gap score", f"{cur['boj_policy_gap_score']:.2f}"]
    ]