import pandas as pd
import numpy as np

class RTMInferenceEngine:
    def __init__(self, regime_clf, driver_clf):
        self.regime_clf = regime_clf
        self.driver_clf = driver_clf
        # 特徴量の定義を分けて保持
        self.driver_features = [
            'sp500_ret_1d', 'sp500_rsi_14d', 'sp500_breadth_diff_10d',
            'sector_dispersion_5d', 'nasdaq_ret_diff_sp500',
            'russell_ret_diff_sp500', 'nasdaq_ret_5d', 'russell_ret_5d',
            'xlk/xlp_ret_5d', 'vix_level', 'vix_chg_5d', 'vix_zscore_10d',
            'vix_panic_duration', 'vvix_zscore_10d', 'hy_level', 'ig_level',
            'hy_diff_5d', 'ig_diff_5d', 'tlt_ret_20d', 'dxy_ret_1d',
            'sp500_gold_corr_10d'
        ]
        self.regime_features = [f for f in self.driver_features if f not in ["xlk/xlp_ret_5d", "vvix_zscore_10d"]]

    def predict_at_date(self, df_features, target_date=None):
        try:
            if target_date is None:
                # 最新行を取得
                row = df_features.tail(1)
                actual_date = row.index[0]
            else:
                # 指定日を取得（文字列でもOK）
                actual_date = pd.to_datetime(target_date)
                if actual_date not in df_features.index:
                    # ぴったりな日付がない場合、その日以前で最新の営業日を探す
                    row = df_features[:actual_date].tail(1)
                    actual_date = row.index[0]
                else:
                    row = df_features.loc[[actual_date]]

            # 1. Regime Prism (19指標)
            regime_probs = self.regime_clf.predict(row[self.regime_features])[0]
            regime_id = np.argmax(regime_probs) + 1

            # 2. Driver Profiler (21指標)
            driver_probs = self.driver_clf.predict(row[self.driver_features])[0]
            driver_id = np.argmax(driver_probs) + 1

            return {
                "date": actual_date.strftime('%Y-%m-%d'),
                "regime": int(regime_id),
                "driver": int(driver_id),
                "regime_conf": np.max(regime_probs),
                "driver_conf": np.max(driver_probs)
            }
        except Exception as e:
            return f"Error: {str(e)}"

    def get_action_guidance(self, result):
        """状態と主因を掛け合わせたアクション・マトリックス"""
        regime_map = {1:"Golden Dip", 2:"Flash Crash", 3:"Slow Bleed", 4:"Liquidity In", 5:"Healthy"}
        driver_map = {1:"Credit", 2:"Bond", 3:"Equity", 4:"Currency", 5:"Neutral"}

        r_name = regime_map[result["regime"]]
        d_name = driver_map[result["driver"]]

        # 統合ロジックの例（ここを戦略に合わせて調整）
        if r_name == "Flash Crash":
            if d_name == "Credit":
                guidance = "⚠️ システム・リスク：全資産退避を検討（Cash 100%）"
            elif d_name == "Bond":
                guidance = "📉 金利ショック：債券・株の同時下落に注意。Cash 70%"
            else:
                guidance = "⚡ 一時的ショック：パニック売りに注意。Cash 50%"

        elif r_name == "Golden Dip":
            if d_name == "Equity":
                guidance = "🚀 健全な反発：全力買い（Leverage 1.2x）"
            else:
                guidance = "✅ 外部要因解消による回復：順張り買い（Cash 0%）"

        else:
            guidance = "⚖️ 通常運用：リバランス強度は標準（Neutral）"

        return f"【{r_name} × {d_name}】: {guidance}"
