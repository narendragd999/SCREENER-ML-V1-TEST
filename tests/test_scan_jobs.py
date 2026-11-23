import asyncio
import threading
from unittest.mock import patch
import pandas as pd
import pytest
from pydantic import BaseModel, Field
from fastapi import HTTPException
from core import scan_jobs, screener


class Request(BaseModel):
    count: int = Field(3, ge=1, le=10)


def test_job_progress_is_real_thread_safe_and_not_complete_early(monkeypatch):
    monkeypatch.setattr(scan_jobs, "_JOBS", {})
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        async def worker(parameters, progress):
            await asyncio.to_thread(progress, {"phase": "download", "completed": 1,
                                   "total": parameters["count"], "ticker": "DIXON", "percent": 40})
            progress({"percent": 10})
            started.set()
            await release.wait()
            progress({"phase": "analysis", "completed": 3, "total": 3, "percent": 100})
            assert scan_jobs.get_job(job["job_id"])["progress"]["percent"] == 99
            return {"tickers": 3}
        scan_jobs.register_job_handler("test_progress", Request, worker)
        job = scan_jobs.submit_job("test_progress", {"count": 3})
        assert job["status"] == "queued"
        await started.wait()
        running = scan_jobs.get_job(job["job_id"])
        assert running["status"] == "running"
        assert running["progress"]["percent"] == 40
        assert running["progress"]["ticker"] == "DIXON"
        release.set()
        await asyncio.gather(*list(scan_jobs._TASKS))
        done = scan_jobs.get_job(job["job_id"])
        assert done["status"] == "completed" and done["progress"]["percent"] == 100
        assert done["result"] == {"tickers": 3}
        done["progress"]["completed"] = 999
        assert scan_jobs.get_job(job["job_id"])["progress"]["completed"] == 3
    asyncio.run(scenario())


def test_job_failure_retains_provider_reason(monkeypatch):
    monkeypatch.setattr(scan_jobs, "_JOBS", {})
    async def worker(parameters, progress):
        raise HTTPException(503, "Provider temporarily unavailable")
    async def scenario():
        scan_jobs.register_job_handler("test_failure", Request, worker)
        job = scan_jobs.submit_job("test_failure", {})
        await asyncio.gather(*list(scan_jobs._TASKS))
        state = scan_jobs.get_job(job["job_id"])
        assert state["status"] == "failed" and "Provider" in state["error"]
        assert state["progress"]["percent"] != 100
    asyncio.run(scenario())


def test_job_parameters_validated_before_work_is_queued():
    async def worker(parameters, progress):
        return {}
    scan_jobs.register_job_handler("test_validation", Request, worker)
    with pytest.raises(HTTPException) as validation:
        scan_jobs.submit_job("test_validation", {"count": 0})
    assert validation.value.status_code == 422
    with pytest.raises(HTTPException) as unknown:
        scan_jobs.submit_job("missing_kind", {})
    assert unknown.value.status_code == 400
    with pytest.raises(HTTPException) as missing:
        scan_jobs.get_job("unknown_id")
    assert missing.value.status_code == 404


def test_job_queue_limits_concurrent_bulk_work(monkeypatch):
    monkeypatch.setattr(scan_jobs, "_JOBS", {})
    async def scenario():
        release = asyncio.Event()
        entered = []
        async def worker(parameters, progress):
            entered.append(parameters["count"])
            await release.wait()
            return parameters
        scan_jobs.register_job_handler("test_queue", Request, worker)
        jobs = [scan_jobs.submit_job("test_queue", {"count": i}) for i in (1, 2, 3)]
        await asyncio.sleep(0)
        assert len(entered) == 2
        assert scan_jobs.get_job(jobs[2]["job_id"])["status"] == "queued"
        release.set()
        await asyncio.gather(*list(scan_jobs._TASKS))
        assert len(entered) == 3
    asyncio.run(scenario())


def test_screener_reports_download_and_analysis_for_every_requested_ticker(monkeypatch):
    bars = pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 100000},
                        index=pd.bdate_range(end="2026-11-06", periods=220))
    def download(tickers, interval, lookback_days, progress=None):
        frames = {"A": bars, "^NSEI": bars}
        for i, ticker in enumerate(tickers, 1):
            progress({"phase": "download", "completed": i, "total": len(tickers),
                      "ticker": ticker, "failed_count": int(ticker == "B")})
        return frames
    monkeypatch.setattr(screener, "fetch_batch_ohlcv", download)
    updates = []
    result = asyncio.run(screener.run_ema9_screener(["A", "B"], progress=updates.append))
    download_steps = [p for p in updates if p["phase"] == "download"]
    assert download_steps[-1]["completed"] == 3
    analysis = [p for p in updates if p["phase"] == "analysis"]
    assert analysis[-1]["completed"] == 2 and analysis[-1]["failed_count"] == 1
    assert result["data_failed_count"] == 1
    assert updates[-1]["percent"] == 99
