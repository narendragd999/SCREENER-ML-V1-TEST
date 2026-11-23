"""The duration API preserves user settings and the entire supplied basket."""
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
import main


def test_duration_api_accepts_90_sessions_and_more_than_300_symbols():
    tickers = [f"STOCK{i}" for i in range(307)]
    runner = AsyncMock(return_value={"stats": {"total_entries": 0}})
    with patch.object(main, "run_duration_backtest", runner):
        response = TestClient(main.app).post("/api/backtest/duration", json={
            "tickers": tickers, "mode": "duration_next_open", "max_hold_sessions": 90,
            "avoid_near_resistance": True, "require_pullback": True,
        })
    assert response.status_code == 200
    passed = runner.await_args.kwargs
    assert passed["tickers"] == tickers
    assert passed["max_hold_sessions"] == 90
    assert passed["mode"] == "duration_next_open"
    assert passed["avoid_near_resistance"] and passed["require_pullback"]


def test_duration_api_rejects_invalid_policy_before_running():
    runner = AsyncMock()
    with patch.object(main, "run_duration_backtest", runner):
        client = TestClient(main.app)
        for bad in ({"max_hold_sessions": 0}, {"max_hold_sessions": 251},
                    {"mode": "unknown"}, {"years": 11}, {"target_pct": 0}):
            result = client.post("/api/backtest/duration", json={"tickers": ["TCS"], **bad})
            assert result.status_code == 422
    runner.assert_not_awaited()
