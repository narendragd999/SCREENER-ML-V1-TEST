"""History lookup regressions: provider failures are not short listings."""
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
import threading
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from core import data_fetcher
import main


def price_bars(count=40, price=100):
    closes = np.full(count, float(price))
    return pd.DataFrame({"Open": closes, "High": closes + 1, "Low": closes - 1,
                         "Close": closes, "Volume": np.full(count, 100_000)},
                        index=pd.bdate_range(end="2026-11-06", periods=count))


class EmptyTicker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, **kwargs):
        return pd.DataFrame()


@pytest.fixture(autouse=True)
def isolate_history_cache(monkeypatch):
    monkeypatch.setattr(data_fetcher, "_PRICE_CACHE", {})
    # A failed download must not reach the real provider during these tests.
    monkeypatch.setattr(data_fetcher.yf, "Ticker", EmptyTicker)


@pytest.mark.parametrize("ticker_first", [False, True])
def test_single_symbol_history_accepts_both_yahoo_column_orders(monkeypatch, ticker_first):
    raw = price_bars(price=123)
    raw.columns = pd.MultiIndex.from_product([["DIXON.NS"], raw.columns] if ticker_first
                                             else [raw.columns, ["DIXON.NS"]])
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: raw.copy())
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400)
    assert result is not None and len(result) == 40
    assert result.Close.iloc[-1] == 123


def test_requested_symbol_is_extracted_from_contaminated_multi_symbol_result(monkeypatch):
    requested, other = price_bars(price=123), price_bars(price=456)
    raw = pd.concat({"DIXON.NS": requested, "TCS.NS": other}, axis=1)
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: raw.copy())
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400)
    assert result is not None
    assert result.Close.iloc[-1] == 123
    assert result.columns.is_unique


def test_other_symbol_prices_are_never_cached_as_requested_stock(monkeypatch):
    wrong = price_bars(price=456)
    wrong.columns = pd.MultiIndex.from_product([wrong.columns, ["TCS.NS"]])
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: wrong.copy())
    with pytest.raises(data_fetcher.HistoryUnavailableError):
        data_fetcher.fetch_ticker_ohlcv("DIXON", raise_on_error=True)
    assert not any(key.startswith("DIXON_") for key in data_fetcher._PRICE_CACHE)


def test_empty_primary_download_recovers_using_requested_ticker_history(monkeypatch):
    calls = []
    class WorkingTicker:
        def __init__(self, symbol):
            calls.append(symbol)
        def history(self, **kwargs):
            assert kwargs.get("auto_adjust") is True
            return price_bars(price=321)
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(data_fetcher.yf, "Ticker", WorkingTicker)
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400, raise_on_error=True)
    assert result is not None and len(result) == 40
    assert result.Close.iloc[-1] == 321
    assert calls and set(calls) == {"DIXON.NS"}


def test_shorter_cached_window_can_support_explicit_partial_analysis(monkeypatch):
    data_fetcher._PRICE_CACHE["DIXON_1d_365"] = (time.time(), price_bars(price=123))
    def unexpected_download(*args, **kwargs):
        raise AssertionError("Valid cached analysis history should be reused")
    monkeypatch.setattr(data_fetcher.yf, "download", unexpected_download)
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400, min_bars=31,
                                          allow_cached_shorter=True, raise_on_error=True)
    assert result is not None and result.Close.iloc[-1] == 123


def test_shorter_cache_does_not_satisfy_a_strict_historical_window(monkeypatch):
    data_fetcher._PRICE_CACHE["DIXON_1d_365"] = (time.time(), price_bars(price=123))
    requested = []
    def download(*args, **kwargs):
        requested.append(args[0])
        return price_bars(price=999)
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400, allow_cached_shorter=False)
    assert requested and result is not None and result.Close.iloc[-1] == 999


def test_global_yahoo_downloads_do_not_overlap_between_requests(monkeypatch):
    mutex = threading.Lock()
    active = maximum = 0
    def download(symbol, **kwargs):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(.025)
            frame = price_bars(price=123 if symbol == "DIXON.NS" else 456)
            frame.columns = pd.MultiIndex.from_product([frame.columns, [symbol]])
            return frame
        finally:
            with mutex:
                active -= 1
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(data_fetcher.fetch_ticker_ohlcv, ["DIXON", "TCS"]))
    assert maximum == 1
    assert [result.Close.iloc[-1] for result in results] == [123, 456]


def test_single_lookup_and_batch_scan_share_download_serialization(monkeypatch):
    mutex, start = threading.Lock(), threading.Barrier(2)
    active = maximum = 0
    def download(symbols, **kwargs):
        nonlocal active, maximum
        with mutex:
            active += 1
            maximum = max(maximum, active)
        try:
            time.sleep(.075)
            names = [symbols] if isinstance(symbols, str) else symbols
            result = pd.concat({name: price_bars(price=123 if name == "DIXON.NS" else 456)
                                for name in names}, axis=1)
            return result if kwargs.get("group_by") == "ticker" else result.swaplevel(axis=1)
        finally:
            with mutex:
                active -= 1
    def singleton():
        start.wait(5)
        return data_fetcher.fetch_ticker_ohlcv("DIXON")
    def batch():
        start.wait(5)
        return data_fetcher.fetch_batch_ohlcv(["TCS"])
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    with ThreadPoolExecutor(max_workers=2) as workers:
        single_future, batch_future = workers.submit(singleton), workers.submit(batch)
        single, multiple = single_future.result(), batch_future.result()
    assert maximum == 1
    assert single.Close.iloc[-1] == 123
    assert multiple["TCS"].Close.iloc[-1] == 456


def test_genuinely_short_provider_history_remains_available_as_partial_data(monkeypatch):
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: price_bars(10))
    result = data_fetcher.fetch_ticker_ohlcv("NEWLISTING", min_bars=31, raise_on_error=True)
    assert result is not None and len(result) == 10


def test_wider_cached_history_serves_a_strict_shorter_request_without_provider(monkeypatch):
    cached = price_bars(price=123)
    data_fetcher._PRICE_CACHE["DIXON_1d_800"] = (time.time(), cached)
    calls = []
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: calls.append(args))
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", lookback_days=400)
    assert result is not None and result.Close.iloc[-1] == 123
    assert not calls and result.attrs["cache_reused"] is True
    result.loc[result.index[-1], "Close"] = 999
    assert cached.Close.iloc[-1] == 123


def test_busy_batch_uses_individual_history_without_calling_unlocked_download(monkeypatch):
    class BusyLock:
        def acquire(self, timeout):
            assert timeout == 2.0
            return False
        def release(self):
            raise AssertionError("An unacquired lock must not be released")
    class WorkingTicker:
        def __init__(self, symbol):
            assert symbol == "DIXON.NS"
        def history(self, **kwargs):
            assert kwargs.get("timeout") == 15
            return price_bars(price=321)
    calls = []
    monkeypatch.setattr(data_fetcher, "_DOWNLOAD_LOCK", BusyLock())
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(data_fetcher.yf, "Ticker", WorkingTicker)
    result = data_fetcher.fetch_ticker_ohlcv("DIXON", raise_on_error=True)
    assert not calls and result.Close.iloc[-1] == 321
    assert result.attrs["ticker"] == "DIXON"
    assert result.attrs["data_source"] == "yfinance.Ticker.history"
    assert result.attrs["available_history"]["bars"] == 40
    assert result.attrs["history_warnings"]


def test_batch_progress_counts_cached_downloaded_and_failed_tickers(monkeypatch):
    data_fetcher._PRICE_CACHE["DIXON_1d_365"] = (time.time(), price_bars(price=123))
    raw = pd.concat({"TCS.NS": price_bars(price=456)}, axis=1)
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: raw)
    events = []
    result = data_fetcher.fetch_batch_ohlcv(["DIXON", "TCS", "MISSING", "DIXON.NS"],
                                          progress=events.append)
    assert set(result) == {"DIXON", "TCS"}
    assert [event["completed"] for event in events] == [0, 1, 1, 2, 2, 3]
    assert all(event["total"] == 3 and event["phase"] == "download" for event in events)
    assert events[1]["ticker"] == "DIXON"
    assert events[-1]["ticker"] == "MISSING" and events[-1]["failed_count"] == 1


def test_empty_batch_reports_each_failed_ticker_instead_of_stalling(monkeypatch):
    monkeypatch.setattr(data_fetcher.yf, "download", lambda *args, **kwargs: pd.DataFrame())
    events = []
    result = data_fetcher.fetch_batch_ohlcv(["DIXON", "TCS"], progress=events.append)
    assert not result
    assert [event["completed"] for event in events] == [0, 0, 1, 1, 2]
    assert events[-1]["total"] == 2 and events[-1]["failed_count"] == 2


def test_progress_downloads_one_stock_and_announces_it_before_network_work(monkeypatch):
    events, requests = [], []
    prices = {"DIXON.NS": 123, "TCS.NS": 456, "INFY.NS": 789}
    def download(symbols, **kwargs):
        assert len(symbols) == 1 and kwargs["threads"] is False
        assert data_fetcher._DOWNLOAD_LOCK.locked()
        assert events[-1]["ticker"] == symbols[0][:-3]
        assert events[-1]["completed"] == len(requests)
        requests.append(symbols[0])
        return pd.concat({symbols[0]: price_bars(price=prices[symbols[0]])}, axis=1)
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    result = data_fetcher.fetch_batch_ohlcv(["DIXON", "TCS", "INFY"], chunk_size=60,
                                          progress=events.append)
    assert requests == ["DIXON.NS", "TCS.NS", "INFY.NS"]
    assert [event["completed"] for event in events] == [0, 0, 1, 1, 2, 2, 3]
    assert events[-1]["failed_count"] == 0
    assert [result[ticker].Close.iloc[-1] for ticker in ["DIXON", "TCS", "INFY"]] == [123, 456, 789]


def test_non_progress_downloads_are_still_small_serialized_groups(monkeypatch):
    requests = []
    def download(symbols, **kwargs):
        assert kwargs["threads"] is False and data_fetcher._DOWNLOAD_LOCK.locked()
        requests.append(list(symbols))
        return pd.concat({symbol: price_bars() for symbol in symbols}, axis=1)
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    tickers = ["DIXON", "TCS", "INFY", "ITC", "SBIN"]
    result = data_fetcher.fetch_batch_ohlcv(tickers, chunk_size=60)
    assert [len(request) for request in requests] == [4, 1]
    assert set(result) == set(tickers)


def test_real_yahoo_serial_downloader_ignores_foreign_fallback_error_rows(monkeypatch):
    # Exercise the installed download assembler, mocking its network worker.
    # A simultaneous Ticker.history failure can add a foreign empty frame;
    # synchronous downloads must finish every requested stock and verify keys.
    import yfinance.multi as multi
    for attribute in ("_DFS", "_ERRORS", "_TRACEBACKS", "_ISINS"):
        monkeypatch.setattr(multi.shared, attribute, {})
    monkeypatch.setattr(multi, "YfData", lambda **kwargs: None)
    monkeypatch.setattr(multi.requests, "Session", lambda **kwargs: object())
    completed = []
    def download_one(symbol, *args, **kwargs):
        assert data_fetcher._DOWNLOAD_LOCK.locked()
        completed.append(symbol)
        frame = price_bars(price=123 if symbol == "DIXON.NS" else 456)
        multi.shared._DFS[symbol] = frame
        multi.shared._DFS["FOREIGN.NS"] = pd.DataFrame()
        multi.shared._ERRORS["FOREIGN.NS"] = "simulated chart fallback failure"
        return frame
    monkeypatch.setattr(multi, "_download_one", download_one)
    monkeypatch.setattr(data_fetcher.yf, "download", multi.download)
    result = data_fetcher.fetch_batch_ohlcv(["DIXON", "TCS"])
    assert set(completed) == {"DIXON.NS", "TCS.NS"}
    assert set(result) == {"DIXON", "TCS"}
    assert result["DIXON"].Close.iloc[-1] == 123
    assert result["TCS"].Close.iloc[-1] == 456
    assert not any(key.startswith("FOREIGN_") for key in data_fetcher._PRICE_CACHE)


def test_provider_unavailable_does_not_claim_stock_has_short_history():
    with patch.object(main, "fetch_ticker_ohlcv", return_value=None):
        response = TestClient(main.app).get("/api/analysis/DIXON")
    assert response.status_code == 503
    assert "Insufficient completed daily history" not in response.text


def test_provider_exception_becomes_a_retryable_stock_analysis_response():
    failure = data_fetcher.HistoryUnavailableError("DIXON history is temporarily unavailable")
    with patch.object(main, "fetch_ticker_ohlcv", side_effect=failure):
        response = TestClient(main.app).get("/api/analysis/DIXON")
    assert response.status_code == 503
    assert "Insufficient completed daily history" not in response.text


def test_short_listing_still_returns_available_stock_research():
    with patch.object(main, "fetch_ticker_ohlcv", return_value=price_bars(10)), \
         patch.object(main, "compute_fair_value", return_value={}), \
         patch.object(main, "analyze_stock_swing_reversal", return_value=None):
        response = TestClient(main.app).get("/api/analysis/NEWLISTING")
    assert response.status_code == 200
    result = response.json()
    assert result["history_bars"] == 10
    assert result["trend"]["trend_regime"] == "UNKNOWN"
    assert result["ema_setup"] is None
    assert result["range_52week"]["sessions_available"] == 10


@pytest.mark.parametrize("endpoint", ["/api/chart/NEWLISTING", "/api/analysis/NEWLISTING",
                                      "/api/swing-reversal/chart/NEWLISTING"])
def test_short_price_views_do_not_wait_for_fundamental_downloads(endpoint):
    with patch.object(main, "fetch_ticker_ohlcv", return_value=price_bars(10)), \
         patch.object(main, "compute_fair_value", side_effect=AssertionError("Unexpected valuation download")) as valuation, \
         patch.object(main, "get_cached_fair_value", return_value={}), \
         patch.object(main, "load_signals", return_value=[]):
        response = TestClient(main.app).get(endpoint)
    assert response.status_code == 200
    result = response.json()
    assert result["ticker"] == "NEWLISTING" and result["current_price"] == 100
    if "candles" in result:
        assert len(result["candles"]) == 10
        assert all(bar["sma50"] is None for bar in result["candles"])
    valuation.assert_not_called()


def test_switching_stock_views_preserves_requested_price_and_source(monkeypatch):
    raw = pd.concat({"DIXON.NS": price_bars(price=123), "TCS.NS": price_bars(price=456)}, axis=1)
    requested = []
    def download(symbol, **kwargs):
        requested.append(symbol)
        return raw.copy()
    monkeypatch.setattr(data_fetcher.yf, "download", download)
    with patch.object(main, "get_cached_fair_value", return_value={}), \
         patch.object(main, "compute_fair_value", side_effect=AssertionError("Unexpected valuation download")), \
         patch.object(main, "load_signals", return_value=[]):
        client = TestClient(main.app)
        first = client.get("/api/analysis/DIXON").json()
        second = client.get("/api/chart/TCS").json()
        cached = client.get("/api/analysis/DIXON").json()
    assert first["ticker"] == cached["ticker"] == "DIXON"
    assert first["current_price"] == cached["current_price"] == 123
    assert first["data_source"] == cached["data_source"] == "yfinance.download"
    assert second["ticker"] == "TCS" and second["current_price"] == 456
    assert requested == ["DIXON.NS", "TCS.NS"]
    assert data_fetcher._PRICE_CACHE["DIXON_1d_400"][1].attrs["ticker"] == "DIXON"
    assert data_fetcher._PRICE_CACHE["TCS_1d_365"][1].attrs["ticker"] == "TCS"


def test_chart_script_is_served_locally_without_a_remote_script_dependency():
    class Scripts(HTMLParser):
        def __init__(self):
            super().__init__()
            self.sources = []
        def handle_starttag(self, tag, attrs):
            if tag == "script":
                source = dict(attrs).get("src")
                if source:
                    self.sources.append(source)
    client = TestClient(main.app)
    response = client.get("/")
    assert response.status_code == 200
    scripts = Scripts()
    scripts.feed(response.text)
    assert "/static/vendor/plotly.min.js" in scripts.sources
    assert all(source.startswith("/static/") for source in scripts.sources)
    asset = client.get("/static/vendor/plotly.min.js")
    assert asset.status_code == 200
    assert "javascript" in asset.headers["content-type"]
    assert "Plotly" in asset.text[:2000]
