import time
from functools import wraps
import pandas as pd

class DTypeError(TypeError):
    pass

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 高精度タイマー
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"[{func.__name__}] {end - start:.6f} sec")
        return result
    return wrapper

def require_columns(columns, df_arg_index=0):
    """指定列を持つDataFrameが渡されているかを検証するデコレーター"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            df = args[df_arg_index]  # 指定された引数位置の DataFrame を取得
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{func.__name__} の第{df_arg_index+1}引数は pandas DataFrame である必要があります")
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise KeyError(f"{func.__name__} に必要列が存在しません: {missing}")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_columns_with_dtype(expected_dtypes, df_arg_index=0):
    """
    expected_dtypes: {"列名": dtype or (dtype1, dtype2, ...)}
    df_arg_index: チェックする DataFrame の位置 (0始まり)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 対象の引数を取得
            try:
                df = args[df_arg_index]
            except IndexError:
                raise IndexError(f"{func.__name__}: 引数位置 {df_arg_index} に DataFrame が存在しません")

            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"{func.__name__}: 引数位置 {df_arg_index} は DataFrame である必要があります")

            # 列存在チェック & dtypeチェック
            for col, dtype_rule in expected_dtypes.items():
                if col not in df.columns:
                    raise DTypeError(f"{func.__name__}: 必要な列が存在しません → '{col}'")

                actual_dtype = df[col].dtype

                # dtype_rule がタプルの場合
                if isinstance(dtype_rule, tuple):
                    ok = any(pd.api.types.is_dtype_equal(actual_dtype, dt) or
                             pd.api.types.is_dtype_equal(actual_dtype, str(dt))
                             for dt in dtype_rule)
                else:
                    ok = pd.api.types.is_dtype_equal(actual_dtype, dtype_rule) or \
                         pd.api.types.is_dtype_equal(actual_dtype, str(dtype_rule))

                if not ok:
                    raise DTypeError(
                        f"{func.__name__}: 列 '{col}' の dtype が不正です。"
                        f"期待: {dtype_rule}, 実際: {actual_dtype}"
                    )

            return func(*args, **kwargs)
        return wrapper
    return decorator

def check_args_types(arg_type_map):
    """
    複数の引数の型をチェックするデコレーター。

    Parameters
    ----------
    arg_type_map : dict
        {引数インデックス: 期待型} の形式で指定。
        期待型は type または tuple of types で指定可能。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for idx, expected_type in arg_type_map.items():
                try:
                    arg = args[idx]
                except IndexError:
                    raise IndexError(f"{func.__name__}: 引数位置 {idx} に値が存在しません")

                if not isinstance(arg, expected_type):
                    raise TypeError(
                        f"{func.__name__}: 引数位置 {idx} は {expected_type} 型である必要があります。"
                        f"実際の型: {type(arg)}"
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator
