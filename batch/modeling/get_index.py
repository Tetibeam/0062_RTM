from batch.modeling.ticker_map import (
    ticker_map,
)

import yfinance as yf
from fredapi import Fred
from jpy_datareader import estat
import pandas as pd
from datetime import datetime
import io
import requests
from io import StringIO
import sys

class DataFetchError(Exception):
    pass

START_FETCH_YF = "2005-01-01"
def get_index_by_asset_class(index_name_list, start_date: pd.Timestamp, end_date: pd.Timestamp):

    df_yfinance = None
    df_fred = None
    df_etc = None

    yfinance_list = []
    fred_list = []
    etc_list = []

    for index_name in index_name_list:
        name, source, ticker = ticker_map[index_name]
        if source == "yfinance":
            yfinance_list.append(ticker)
        elif source == "fred":
            fred_list.append(ticker)
        elif source in ("bis_monthly", "bis_quarterly", "cboe", "stooq"):
            etc_list.append((ticker, name, source))
        elif source == "wiki":
            if index_name in "S&P500構成銘柄":
                return _get_yfinance_prices(_get_wiki_sp500_data(ticker), start_date, end_date)
            elif index_name in "日経平均構成銘柄":
                return _get_yfinance_prices(_get_wiki_nikkei_data(ticker), start_date, end_date)
            else:
                raise DataFetchError("不正なデータソース指定", source)
        else:
            raise DataFetchError("不正なデータソース指定", source)

    if yfinance_list:
        df_yfinance = _get_yfinance_prices(yfinance_list, start_date, end_date)

    if fred_list:
        df_fred = _get_fred_prices(fred_list, start_date, end_date)

    df_list = []
    total=len(etc_list)
    counter=0
    print("BIS / CBOE取得中...")
    for ticker, name, source in etc_list:
        msg = f"[{'*'*(counter+1):<{total}}] {counter+1} of {total} completed"
        sys.stdout.write("\r" + msg)
        sys.stdout.flush()
        print(f" {ticker}")
        match source:
            case "bis_monthly":
                df = _get_bis_prices_monthly(ticker, name, start_date, end_date)
            case "bis_quarterly":
                df = _get_bis_prices_quarterly(ticker, name, start_date, end_date)
            case "cboe":
                df = _get_cboe_prices(ticker, name, start_date, end_date)
            case "stooq":
                df = _get_stooq_prices(ticker, name, start_date, end_date)
            case _:
                raise DataFetchError("不正なデータソース指定", source)
        df_list.append(df)
        counter +=1

    if df_list:
        df_etc = pd.concat(df_list, axis=1)

    valid_dfs = [df for df in [df_yfinance, df_fred, df_etc] if isinstance(df, pd.DataFrame)]
    if not valid_dfs:
        raise DataFetchError("一つもデータが取得できませんでした")

    df_all = pd.concat(valid_dfs, axis=1).sort_index()

    all_nan_cols = df_all.columns[df_all.isna().all()]
    if len(all_nan_cols) > 0:
        raise ValueError(f"All-NaN columns detected: {list(all_nan_cols)}")

    return df_all

##########################################
#yfinance
##########################################
def _get_yfinance_prices(tickers: list, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # --- データ取得 ---
    try:
        df = yf.download(
            tickers,
            start=START_FETCH_YF,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=True
        )
        #print(df)
    except Exception as e:
        raise DataFetchError(f"yfinance 取得失敗: {e}")
    
    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("yfinanceデータが取得できませんでした", name)
    
    df = df.xs("Close", axis=1, level=1)

    start = start_date.normalize()
    end = end_date.normalize()
    
    return df.loc[start:end]

##########################################
#fred
##########################################
def _get_fred_prices(tickers: list, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # --- データ取得 ---

    print(f"FRED取得中... : {len(tickers)}指標")
    dfs = []
    errors = []
    try:
        fred = Fred(api_key="d8028eba4732e356349912d4e0f07dc3")
        for i, ticker in enumerate(tickers):
            print(f"FRED取得中... : {ticker}")

            try:
                s = fred.get_series(ticker)
                s.name = ticker  # ← これ重要
                dfs.append(s)

            except Exception as e:
                print(f"❌ エラー: {ticker} -> {e}")
                errors.append(ticker)

        df = pd.concat(dfs, axis=1)
        df.index.name = "Date"

    except Exception as e:
        raise DataFetchError(f"fred 取得失敗: {e}")

    """try:
        print(f"FRED取得中... : {len(tickers)}指標")
        fred = Fred(api_key="d8028eba4732e356349912d4e0f07dc3")
        df = pd.concat(
            {s: fred.get_series(s) for s in tickers},
            axis=1
        )
        df.columns = tickers
        df.index.name ="Date"

    except Exception as e:
        raise DataFetchError(f"fred 取得失敗: {e}")"""

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("fredデータが取得できませんでした", name)

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#bis - monthly
##########################################
def _get_bis_prices_monthly(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv'
        }
        response = requests.get(ticker, headers=headers, timeout=10)
        r_text = response.text
        if r_text:
            df = pd.read_csv(io.StringIO(r_text))
            df = df[["TIME_PERIOD", "OBS_VALUE"]]
            df = df.rename(columns={"TIME_PERIOD":"Date","OBS_VALUE":name})
            df["Date"] = pd.to_datetime(df["Date"],format="%Y-%m")
            df = df.set_index("Date").sort_index()
    except Exception as e:
        raise DataFetchError(f"BIS 取得失敗: {e}")

    if df is None or df.empty:
        raise DataFetchError("データが空でした")

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#bis - quarterly
##########################################
def _get_bis_prices_quarterly(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/csv'
        }
        response = requests.get(ticker, headers=headers, timeout=10)
        r_text = response.text
        if r_text:
            df = pd.read_csv(io.StringIO(r_text))
            df = df[["TIME_PERIOD", "OBS_VALUE"]]
            df = df.rename(columns={"TIME_PERIOD":"Date","OBS_VALUE":name})
            df["Date"] = pd.to_datetime(
                pd.PeriodIndex(df['Date'].str.replace('-', ''), freq='Q').to_timestamp()
            )
            df = df.set_index("Date").sort_index()
    except Exception as e:
        raise DataFetchError(f"BIS 取得失敗: {e}")

    if df is None or df.empty:
        raise DataFetchError("データが空でした")

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#Cboe
##########################################
def _get_cboe_prices(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        url = ticker
        df = pd.read_csv(url)

        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        if 'CLOSE' in df.columns:
            df = df["CLOSE"].rename(name)
    except Exception as e:
        raise DataFetchError(f"Cboe 取得失敗: {e}")

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("データが取得できませんでした")

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#stooq
##########################################
def _get_stooq_prices(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    try:
        url = ticker
        df = pd.read_csv(url)

        df['DATE'] = pd.to_datetime(df['Date'])
        df.set_index('DATE', inplace=True)
        if 'Close' in df.columns:
            df = df["Close"].rename(name)
    except Exception as e:
        raise DataFetchError(f"Cboe 取得失敗: {e}")

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("データが取得できませんでした")

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#wiki
##########################################
def _get_wiki_sp500_data(ticker: str) -> list:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        html = requests.get(ticker, headers=headers).text

        df = pd.read_html(StringIO(html))[0]

        tickers = (
            df["Symbol"]
            .str.replace(".", "-", regex=False)
            .tolist()
        )
    except Exception as e:
        raise DataFetchError(f"wiki 取得失敗: {e}")

    # --- 取得結果チェック ---
    if tickers is None:
        raise DataFetchError("データが取得できませんでした")

    return tickers

def _get_wiki_nikkei_data(ticker: str) -> list:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        html = requests.get(ticker, headers=headers).text

        df = pd.read_html(StringIO(html))
        tables = pd.read_html(StringIO(html))

        symbols = []

        for table in tables:
            if "証券コード" in table.columns:
                symbols.extend(table["証券コード"].astype(str).tolist())
        symbols = [f"{s}.T" for s in symbols]
    except Exception as e:
        raise DataFetchError(f"wiki 取得失敗: {e}")

    # --- 取得結果チェック ---
    if symbols is None:
        raise DataFetchError("データが取得できませんでした")
    return symbols

#---------------------------------------------------------------------------------------------------------------------
##########################################
#Test
##########################################
##########################################
#yfinance
##########################################
def _get_yfinance_prices_test(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # --- データ取得 ---
    try:
        df = yf.download(
            ticker,
            start=START_FETCH_YF,
            progress=False,
            threads=False,
            auto_adjust=False,
        )
    except Exception as e:
        raise DataFetchError(f"yfinance 取得失敗: {e}")

    return df
    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("yfinanceデータが取得できませんでした", name)

    df= df[("Close", ticker)].rename(name)

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#fred
##########################################
def _get_fred_prices_test(ticker: str, name:str, start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    # --- データ取得 ---
    try:
        print(f"FRED取得テスト... : {ticker}")
        fred = Fred(api_key="d8028eba4732e356349912d4e0f07dc3")
        df = fred.get_series(ticker,observation_start="2003-01-01")
        df.name = name
        df.index.name ="Date"

    except Exception as e:
        raise DataFetchError(f"fred 取得失敗: {e}")

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("fredデータが取得できませんでした", name)
    
    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

##########################################
#ESTAT
##########################################

#----- 日本消費者物価指数-----
def fetch_jp_cpi_prices_test() -> pd.DataFrame:

    # --- データ取得 ---
    try:
        api_key="0bbfddb466c68e94f8f1d1144634ef8b275e7fdd"
        statsdata = estat.StatsDataReader(api_key, statsDataId="0003427113", cdArea="00000", cdCat01="0001")
        df = statsdata.read()

        df = df[["時間軸（年・月）コード", "表章項目", "値"]].copy()
        df["年"] = df["時間軸（年・月）コード"].str[:4].astype(int)
        df["月"] = df["時間軸（年・月）コード"].str[6:8].astype(int)

        df = df[df["表章項目"] == "指数"]
        df = df[df["月"] != 0]

        df["Date"] = pd.to_datetime({
            "year": df["年"],
            "month": df["月"],
            "day": 1
        })

        df = df[["Date", "値"]].rename(columns={"値":"jp_cpi"}).set_index("Date")
        df = df.sort_index()

    except Exception as e:
        raise DataFetchError(f"estat 取得失敗: {e}")

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("株価データが取得できませんでした")

    return df

def get_jp_cpi_prices_test(start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:

    df = fetch_jp_cpi_prices()

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]

#----- EM 政策金利 proxy（SEML.L ETF価格） -----#
def fetch_em_policy_rate_proxy_prices_test() -> pd.DataFrame:
    # --- データ取得 ---
    try:
        df = yf.download(
            "SEML.L",
            start=START_FETCH_YF,
            progress=False,
            threads=False,
            auto_adjust=False,
        )
    except Exception as e:
        raise DataFetchError(f"yfinance 取得失敗: {e}")

    # --- 取得結果チェック ---
    if df is None or df.empty:
        raise DataFetchError("株価データが取得できませんでした")

    return df[("Close", "SEML.L")].rename("em_policy_rate_proxy")

def get_em_policy_rate_proxy_prices_test(start_date:pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:

    df = fetch_em_policy_rate_proxy_prices()

    start = start_date.normalize()
    end = end_date.normalize()

    return df.loc[start:end]


if __name__ == "__main__":
    from app import create_app
    app = create_app()

    start = pd.Timestamp("1999-01-01")
    end = pd.Timestamp("2026-04-01")
    #df = _get_yfinance_prices_test(ticker="EEM", name="EEM", start_date=start, end_date=end)
    df = _get_fred_prices_test(ticker="THREEFYTP10", name="THREEFYTP10", start_date=start, end_date=end)
    #df = _get_fred_prices(tickers=["CP"], start_date=start, end_date=end)
    print(df.dropna())