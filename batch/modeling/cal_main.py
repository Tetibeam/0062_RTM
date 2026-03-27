from batch.modeling.regime import get_market_regime_model_beta
from batch.modeling.modeling_regional_bias import get_regional_bias_model_beta
from batch.modeling.gli import get_gli_model_beta
from batch.modeling.dsr import get_dsr_model_beta
from batch.modeling.get_index import get_index_by_asset_class
from batch.modeling.driver import get_driver_beta

import pandas as pd
import joblib

from app import cache

regime_index_list = [
    # DSR model
    "住宅ローン金利（30年固定）",
    "商業銀行の企業向け貸出",
    "ムーディーズ社格付けBaa級社債利回り",
    "ムーディーズ社格付けAaa級社債利回り",
    "実質可処分所得",
    "雇用者数",
    "失業率",
    "労働時間（製造業）",
    "債務サービス比率",
    "設備投資額",
    "企業利益",
    "賃金指数",
    "クレジットカード金利",
    "消費者信用残高",
    "不動産向け融資残高",
    "個人消費支出価格指数",
    "クレジットカードローンの延滞率",
    "個人貯蓄率",
    "非農業ビジネス部門の実質時間当たり報酬",
    # GLI model
    "米ドル建ての代表的代替リスクフリーレート",
    "実効フェデラルファンド金利",
    "BBB–A 社債スプレッド",
    "A-A 社債スプレッド",
    "平均時給",
    "米国3か月財務省証券利回り",
    "グローバル流動性",
    "家計・非営利団体の住宅ローン残高",
    "FRB総資産",
    "リバースレポ残高",
    "準備預金残高",
    "銀行システムの総準備金",
    "TGA",

    # 市場レジュームモデル
    "S&P500指数",
    "NASDAQ総合指数",
    "Russell2000指数",
    "S&P500 セクターETF 金融",
    "S&P500 セクターETF テクノロジー",
    "S&P500 セクターETF エネルギー",
    "S&P500 セクターETF 一般消費財",
    "S&P500 セクターETF 生活必需品",
    "S&P500 セクターETF 公益事業",
    "VIX指数",
    "VIX3M指数",
    "VIX9D指数",
    "投資適格社債OAS",
    "ハイイールド社債OAS",
    "米国10年国債利回り",
    "長短金利差",
    "金先物",
    "USD/JPY",
    "USD/CHF",
    "期待インフレ率",
    "米国国債長期債",
    "ドルインデックス",
    "恐怖指数",
    "ボラのボラ",
    "ブラックスワン指数",
    "銀行信用リスクのプレミアム",
    "企業の短期資金調達コスト",
    "3ヶ月物国債利回り",
    "米国債 3ヶ月金利",
    "米国債2年利回り",
 
    # 地域バイアスモデル
    "日経平均",
    "日本10年国債利回り",
    "原油",
    "銅",
    "SOX指数",
    "TOPIX",
    "10年物米国債インフレ連動債利回り",
    "TOPIX セクターETF 銀行",
    "TOPIX セクターETF 食品",
    "TOPIX セクターETF エネルギー",
    "TOPIX セクターETF 電気機器",
    "TOPIX セクターETF 機械",
    "TOPIX セクターETF 輸送用機器",
    "TOPIX セクターETF 鉄鋼",
    "TOPIX セクターETF 非鉄",
    "TOPIX セクターETF 化学",
    "TOPIX セクターETF 医薬品",
    "TOPIX セクターETF 小売",
    "TOPIX セクターETF 卸売",
    "TOPIX セクターETF 金融",
    "TOPIX セクターETF 不動産",
    ]

@cache.cached(
    timeout=60 * 60 * 6,
    key_prefix=lambda *args, **kwargs: f"index_for_model_2:prices:raw:{kwargs.get('months')}"
)
def get_index_for_learning(months=24):
    # 日付
    today = pd.Timestamp.now().normalize()
    start = today - pd.DateOffset(months=months)

    df = get_index_by_asset_class(regime_index_list, start, today)

    df_sp500 = get_index_by_asset_class(["S&P500構成銘柄"], start, today)
    df_nikkei = get_index_by_asset_class(["日経平均構成銘柄"], start, today)

    return df, df_sp500, df_nikkei

def cal_main():
    # 指標の取得
    df_index, df_sp500, df_nikkei = get_index_for_learning(months=360)
    #df_index = get_index_for_learning(months=360)

    # --- Macro学習モデルの作成 ---
    df_driver_prob = get_driver_beta(df_index, df_sp500)
    #df_credit_driver_prob = get_credit_driver_beta(df_index, df_sp500)
    #df_bond_driver_prob = get_bond_driver_beta(df_index, df_sp500)
    #df_gli_prob = get_gli_model_beta(df_index)
    #df_dsr_prob = get_dsr_model_beta(df_index)
    #df_regime_prob = get_market_regime_model_beta(df_index, df_sp500)

    #df_regional_pred = get_regional_bias_model_beta(df_index, regime_clf, df_regime_features, df_nikkei, df_sp500)

    """
    save_model(
        regime_clf, df_regime_trajectory, df_regime["regime"], df_regime.index.min(), df_regime.index.max(),
        len(df_regime), "0.1.0", 'regime_prism_v0_1_0.joblib')
    save_model(
        driver_clf, df_driver_trajectory, df_driver["driver"], df_driver.index.min(), df_driver.index.max(),
        len(df_driver),"0.1.0", 'driver_profiler_v0_1_0.joblib')
    """

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