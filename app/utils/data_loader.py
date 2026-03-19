from typing import Union, List
from pathlib import Path

import pandas as pd
import sqlite3
import os

# データベース接続マネージャーをインポート
from .db_manager import get_engine, get_db_path

# DBアクセスは with 文を使うことにします
# 1. commit/rollback/close を自動化して安全
# 2. コードが短く、読みやすい
# 3. 例外発生時も DB が壊れない

# ------ 読み込み -------
def get_raw_table(table):
    engine = get_engine("finance")
    with engine.connect() as conn:
        df = pd.read_sql_table(table, conn)
    return df

def get_latest_date():
    sql = "SELECT MAX(date) AS latest_date FROM asset_profit_detail"
    engine = get_engine("finance")
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn)

    latest_date = df["latest_date"].iloc[0]

    if pd.isna(latest_date):
        return None

    return pd.to_datetime(latest_date)

def query_table_aggregated(
    table_name: str,
    aggregates: dict,
    group_by: list = None,
    start_date: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    filters: dict = None,
    order_by: list = None  # 追加
) -> pd.DataFrame:
    """
    集計付きでテーブルを取得する関数
    """
    # SELECT句作成
    select_clause = []
    if group_by:
        select_clause.extend(group_by)
    for col, func in aggregates.items():
        select_clause.append(f"{func}({col}) AS {col}")

    sql = f"SELECT {', '.join(select_clause)} FROM {table_name}"

    # WHERE句作成
    params = {}
    where_clauses = []
    # 日付範囲指定
    if start_date:
        where_clauses.append("date >= :start_date")
        params["start_date"] = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    if end_date:
        next_day = pd.to_datetime(end_date) + pd.Timedelta(days=1)
        where_clauses.append("date < :end_date")
        params["end_date"] = next_day.strftime("%Y-%m-%d")
    # フィルタリング
    if filters:
        for i, (col, val) in enumerate(filters.items()):
            if isinstance(val, (list, tuple)):
                placeholders = []
                for j, v in enumerate(val):
                    pname = f"param_{i}_{j}"
                    placeholders.append(f":{pname}")
                    params[pname] = v
                where_clauses.append(f"{col} IN ({', '.join(placeholders)})")
            else:
                param_name = f"param_{i}"
                where_clauses.append(f"{col} = :{param_name}")
                params[param_name] = val

    # WHERE句を組み立てる
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    # GROUP BY句
    if group_by:
        sql += " GROUP BY " + ", ".join(group_by)

    # ORDER BY句
    if order_by:
        sql += " ORDER BY " + ", ".join(order_by)

    # DB接続してDataFrame取得
    engine = get_engine("finance")
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    # 日付列自動変換
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col])
    return df

def query_table_date_filter(table_name, start_date:pd.Timestamp, end_date:pd.Timestamp):
    end_date = end_date + pd.Timedelta(days=1)
    sql = f"SELECT * FROM {table_name} WHERE date >= :start_date AND date < :end_date"
    engine = get_engine("finance")
    with engine.connect() as conn:
        df = pd.read_sql_query(sql, conn, params={"start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")})
    df["date"] = pd.to_datetime(df["date"])

    return df

# ------ 書き込み -------
def append_to_table(df: pd.DataFrame, table_name: str) -> int:
    """
    DataFrame の内容を指定テーブルに追記して、追加件数を返す。

    Args:
        df (pd.DataFrame): 追記する DataFrame
        table_name (str): 追記先テーブル名

    Returns:
        int: 追加した行数
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
    if df.empty:
        return 0
    if not isinstance(table_name, str) or not table_name.isidentifier():
        raise ValueError(f"Invalid table name: {table_name}")

    # --- SQLAlchemyエンジンを使ってDB接続 ---
    engine = get_engine("finance")
    try:
        with engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists="append", index=False)
        return len(df)
    except Exception as e:
        raise Exception(f"DB追加に失敗しました: {e}")

def update_from_csv(csv_path: str, table_name: str) -> int:
    """
    CSV ファイルを読み込んで指定テーブルに追記する。

    Args:
        csv_path (str): CSV ファイルのパス
        table_name (str): 追記先テーブル名

    Returns:
        int: 追加した行数
    """
    df = pd.read_csv(csv_path, parse_dates=["date"])
    return append_to_table(df, table_name)

def replace_to_table(df, table_name):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)}")
    if df.empty:
        return 0
    if not isinstance(table_name, str) or not table_name.isidentifier():
        raise ValueError(f"Invalid table name: {table_name}")

    # --- SQLAlchemyエンジンを使ってDB接続 ---
    engine = get_engine("finance")
    try:
        with engine.begin() as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        return len(df)
    except Exception as e:
        raise Exception(f"DB上書きに失敗しました: {e}")

# ------ インデックス -------
def create_index_if_not_exists(table_name, column_name):
    engine = get_engine("finance")
    index_name = f"idx_{table_name}_{column_name}"
    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({column_name});"
    with engine.connect() as conn:
        conn.execute(sql)

def create_composite_index(table_name, columns):
    """
    table_name: str
    columns: list of str
    """
    engine = get_engine("finance")
    index_name = f"idx_{table_name}_{'_'.join(columns)}"
    cols = ", ".join(columns)
    sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name}({cols});"
    with engine.connect() as conn:
        conn.execute(sql)

if __name__ == "__main__":
    base_dir = os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    # DBマネージャーの初期化
    from app.utils.db_manager import init_db
    init_db(base_dir)

    df = query_table_aggregated(
        table_name="balance_detail",
        aggregates={
            "金額": "SUM",
            "目標": "SUM"
        },
        group_by=["date", "収支タイプ", "収支カテゴリー"],
        start_date=pd.to_datetime("2024-10-01"),
        end_date = pd.to_datetime("2025-12-01"),
        filters=None,
        order_by=["date"]
    )
    print(df)
