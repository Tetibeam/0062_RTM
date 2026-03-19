from flask import Flask
from flask_cors import CORS
from flask_caching import Cache
from app.utils.config import load_settings

import os

# 初期化前に import されるファイルで利用できるよう、ファイルスコープで定義
cache = Cache()

def create_app():
    app = Flask(__name__)

    #キャッシュの設定
    cache_config = {
        # Redisをバックエンドに使用
        "CACHE_TYPE": "redis",
        # Redisの接続先URL (ローカルのデフォルトポート)
        "CACHE_REDIS_URL": "redis://localhost:6379/0",
        # キャッシュのデフォルト有効期限 (秒)。今回は関数デコレーターで指定するため不要だが、設定しておく。
        "CACHE_DEFAULT_TIMEOUT": 300
    }

    app.config.from_mapping(cache_config)
    # Cache インスタンスの初期化
    cache.init_app(app)

    # CORSを有効化（開発環境用）
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})

    # YAML設定を読み込み
    # setting.yaml is at the root, so we need to go up one level from app/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setting_path = os.path.join(base_dir, "setting.yaml")

    settings = load_settings(setting_path)

    # DBマネージャーの初期化
    from app.utils.db_manager import init_db
    init_db(base_dir)

    # まとめて Flask に登録
    for key, value in settings.items():
        app.config[key.upper()] = value

    # Blueprint登録
    from app.routes.Market_Regime_Radar_routes import Market_Regime_Radar_bp
    app.register_blueprint(Market_Regime_Radar_bp)

    # plotly template
    from app.utils.dashboard_utility import make_graph_template
    make_graph_template()

    return app
