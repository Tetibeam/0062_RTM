from batch.modeling.regime import get_market_regime_model_beta
from batch.modeling.modeling_regional_bias import get_regional_bias_model_beta
from batch.modeling.vitality_engine import get_vitality_engine_beta
from batch.modeling.get_index import get_index_by_asset_class
from batch.modeling.credit_profiler import get_driver_beta

import pandas as pd
import joblib

from app import cache

regime_index_list = [
    "S&P500指数",
    "米国国債長期債",
    "ハイイールド社債OAS",
    "VIX指数",
    "ボラのボラ",
    "恐怖指数",
    "米ドル建ての代表的代替リスクフリーレート",
    "銀行信用リスクのプレミアム",
    "米国10年国債利回り",
    "米国債 3ヶ月金利",
    "10年物米国債インフレ連動債利回り",
    "期待インフレ率",
    "米国債2年利回り",
    "ドルインデックス",
    "金先物",
    "原油",
    "実効FF金利",
    "Baa-10年国債スプレッド",
    "長短金利差",

    "リバースレポ残高",
    "FRB総資産",
    "TGA",
    "流通現金",
    "シカゴ連銀金融条件指数",
    "3ヶ月物国債利回り",
    "銅",
    "海外関連拠点向け純資金ポジション",
    "金融機関の調達コスト",
    "市場全体のストレス",
    "建築許可件数",
    "シカゴ連銀全米金融コンディション指数（調整済み）",
    "グローバル流動性",
    "クレジットギャップ",
    "債務サービス比率",

    ]

@cache.cached(
    timeout=60 * 60 * 6,
    key_prefix=lambda *args, **kwargs: f"index_for_model:prices:raw:{kwargs.get('months')}"
)
def get_index_for_learning(months=24):
    # 日付
    today = pd.Timestamp.now().normalize()
    start = today - pd.DateOffset(months=months)

    df = get_index_by_asset_class(regime_index_list, start, today)
    #print(df["BAA10Y"].dropna().head(10))

    df_sp500 = get_index_by_asset_class(["S&P500構成銘柄"], start, today)
    df_nikkei = get_index_by_asset_class(["日経平均構成銘柄"], start, today)

    return df, df_sp500, df_nikkei

def cal_main():
    # 指標の取得

    df_index, df_sp500, df_nikkei = get_index_for_learning(months=360)

    #df_index = get_index_for_learning(months=360)

    # --- Macro学習モデルの作成 ---
    df_driver_prob = get_driver_beta(df_index, df_sp500)
    #df_gli_prob = get_vitality_engine_beta(df_index)

def save_model(model, trajectory, label, start_date, end_date, row_count, version, filename):
    # 保存したい情報を一つの辞書にまとめる
    model_bundle = {
	    'model': model,
        "trajectory": trajectory,
    	'label': label,
   	    'train_data_info': {
            'start_date': start_date,
            'end_date': end_date,
            'row_count': row_count
            },
    	'version': version,
    }

    # まとめて1ファイルで保存
    joblib.dump(model_bundle, filename)

if __name__ == "__main__":
    from app import create_app
    app = create_app()
    cal_main()
