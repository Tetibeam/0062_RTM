from flask import Blueprint,jsonify
from .Market_Regime_Radar_service import build_Market_Regime_Radar_payload
from .routes_helper import apply_etag
from werkzeug.exceptions import InternalServerError

import os

Market_Regime_Radar_bp = Blueprint("Market_Regime_Radar", __name__, url_prefix="/api/Market_Regime_Radar")

# API 用ルート
@Market_Regime_Radar_bp.route("/", methods=["GET"])
def index():
    """
    API root: 簡単なメタ情報を返す
    """
    payload = {
        "service": "Market_Regime_Radar",
        "version": "1.0",
        "endpoints": {
            "graphs": "/api/Market_Regime_Radar/graphs",
            "summary": "/api/Market_Regime_Radar/summary"
        }
    }
    return jsonify(payload)

@Market_Regime_Radar_bp.route("/graphs", methods=["GET"])
def graphs():
    """
    グラフ用データを返すエンドポイント。
    フロントはここから時系列データ・メタ情報を受け取り描画する。
    """
    try:
        payload = build_Market_Regime_Radar_payload(include_graphs=True, include_summary=False)

        return apply_etag(payload)
    except Exception as e:
        import traceback
        traceback.print_exc()
        # ログはアプリ側で出している想定
        raise InternalServerError(description=str(e))


@Market_Regime_Radar_bp.route("/summary", methods=["GET"])
def summary():
    """
    サマリ（軽量）だけほしいフロントのための簡易エンドポイント。
    """
    try:
        payload = build_Market_Regime_Radar_payload(include_graphs=False, include_summary=True)
        return apply_etag(payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise InternalServerError(description=str(e))
