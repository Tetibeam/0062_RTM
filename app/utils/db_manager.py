"""
データベース接続を一元管理するマネージャーモジュール

このモジュールは、SQLiteとPostgreSQLの接続を環境変数に基づいて
自動的に切り替える機能を提供します。
"""
from typing import Optional, Union
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

import os
import sqlite3
import yaml

try:
    import psycopg2
    from psycopg2 import pool
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


# 環境変数を読み込み
load_dotenv()

# プロジェクトルートディレクトリ
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 設定ファイルの読み込み (削除)
# SETTING_FILE = BASE_DIR / "setting.yaml"
# with open(SETTING_FILE, 'r', encoding='utf-8') as f:
#     SETTINGS = yaml.safe_load(f)


class DatabaseManager:
    """
    データベース接続を管理するクラス

    環境変数 DB_TYPE に基づいて、SQLiteまたはPostgreSQLへの接続を提供します。
    """

    _instance = None
    _connection_pool = None

    def __new__(cls, base_dir: Optional[Union[str, Path]] = None):
        """シングルトンパターンで実装"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, base_dir: Optional[Union[str, Path]] = None):
        """
        初期化

        Args:
            base_dir: プロジェクトルートディレクトリ。Noneの場合は自動推定。
        """
        # シングルトンなので既に初期化済みならスキップ
        if getattr(self, 'initialized', False):
             return

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        else:
            base_dir = Path(base_dir)

        setting_file = base_dir / "setting.yaml"

        if not setting_file.exists():
             raise FileNotFoundError(f"Setting file not found: {setting_file}")

        with open(setting_file, 'r', encoding='utf-8') as f:
            self.settings = yaml.safe_load(f)

        self.db_type = os.getenv("DB_TYPE", "sqlite").lower()

        if self.db_type == "postgresql":
            if not PSYCOPG2_AVAILABLE:
                raise ImportError(
                    "psycopg2 is not installed. "
                    "Install it with: pip install psycopg2-binary"
                )
            self._init_postgresql_pool()
        elif self.db_type == "sqlite":
            self._init_sqlite_config(base_dir)
        else:
            raise ValueError(f"Unsupported DB_TYPE: {self.db_type}")

        self.initialized = True

    def _init_sqlite_config(self, base_dir: Path):
        """SQLite設定の初期化"""
        db_path = self.settings['database']['sqlite']['path']
        self.sqlite_db_dir = base_dir / db_path
        self.sqlite_db_dir.mkdir(parents=True, exist_ok=True)

        # データベースファイルパス
        self.finance_db = self.sqlite_db_dir / self.settings['database']['sqlite']['finance']

    def _init_postgresql_pool(self):
        """PostgreSQL接続プールの初期化"""
        if self._connection_pool is None:
            self._connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv("DB_HOST", "localhost"),
                port=int(os.getenv("DB_PORT", "5432")),
                database=os.getenv("DB_NAME", "finance_db"),
                user=os.getenv("DB_USER", "finance_user"),
                password=os.getenv("DB_PASSWORD", "finance_password")
            )

    @contextmanager
    def get_connection(self, db_name: str = "finance"):
        """
        データベース接続を取得するコンテキストマネージャー
        """
        conn = None
        try:
            if self.db_type == "postgresql":
                conn = self._connection_pool.getconn()
                yield conn
                conn.commit()
            else:  # sqlite
                # db_name引数は無視して常にfinance_dbを使用
                db_path = self.finance_db
                conn = sqlite3.connect(str(db_path))
                yield conn
                conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                if self.db_type == "postgresql":
                    self._connection_pool.putconn(conn)
                else:
                    conn.close()

    def get_sqlalchemy_engine(self, db_name: str = "finance") -> Optional["Engine"]:
        """
        SQLAlchemyエンジンを取得
        """
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is not installed. "
                "Install it with: pip install SQLAlchemy"
            )

        if self.db_type == "postgresql":
            connection_string = (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            )
        else:  # sqlite
            # db_name引数は無視して常にfinance_dbを使用
            db_path = self.finance_db
            connection_string = f"sqlite:///{db_path}"

        return create_engine(connection_string)

    def get_db_path(self, db_name: str = "finance") -> str:
        """
        データベースパスまたは接続文字列を取得
        """
        if self.db_type == "postgresql":
            return (
                f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
                f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
            )
        else:  # sqlite
            # db_name引数は無視して常にfinance_dbを使用
            db_path = self.finance_db
            return str(db_path)

    def close_pool(self):
        """接続プールをクローズ（PostgreSQLの場合のみ）"""
        if self.db_type == "postgresql" and self._connection_pool:
            self._connection_pool.closeall()
            self._connection_pool = None


# グローバルインスタンス (初期値はNone)
db_manager: Optional[DatabaseManager] = None


def init_db(base_dir: Optional[Union[str, Path]] = None) -> DatabaseManager:
    """
    データベースマネージャーを初期化する

    Args:
        base_dir: プロジェクトルートディレクトリ

    Returns:
        初期化されたDatabaseManagerインスタンス
    """
    global db_manager

    # 環境変数を読み込み
    load_dotenv()

    if db_manager is None:
        db_manager = DatabaseManager(base_dir)

    return db_manager


def _check_db_manager():
    if db_manager is None:
        raise RuntimeError("DatabaseManager is not initialized. Call init_db() first.")


def get_connection(db_name: str = "finance"):
    """
    データベース接続を取得するヘルパー関数
    """
    _check_db_manager()
    return db_manager.get_connection(db_name)


def get_engine(db_name: str = "finance"):
    """
    SQLAlchemyエンジンを取得するヘルパー関数
    """
    _check_db_manager()
    return db_manager.get_sqlalchemy_engine(db_name)


def get_db_path(db_name: str = "finance") -> str:
    """
    データベースパスを取得するヘルパー関数
    """
    _check_db_manager()
    return db_manager.get_db_path(db_name)
