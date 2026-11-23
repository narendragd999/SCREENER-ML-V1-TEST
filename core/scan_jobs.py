"""Local asynchronous scans with progress reported by actual ticker work."""
import asyncio
import copy
import datetime as dt
import threading
import time
import uuid
import weakref
from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

router = APIRouter(prefix="/api/jobs", tags=["Scan progress"])
_HANDLERS = {}
_JOBS = {}
_TASKS = set()
_LOCK = threading.RLock()
_SEMAPHORES = weakref.WeakKeyDictionary()


def register_job_handler(kind, model, worker):
    _HANDLERS[kind] = (model, worker)


def _stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30))).isoformat(timespec="seconds")


def get_job(job_id):
    with _LOCK:
        if job_id not in _JOBS:
            raise HTTPException(404, "Scan job does not exist; restart the scan if the local service restarted.")
        return copy.deepcopy(_JOBS[job_id])


def _cleanup():
    cutoff = time.time() - 3600
    with _LOCK:
        for job_id, job in list(_JOBS.items()):
            if job.get("finished_timestamp", float("inf")) < cutoff:
                del _JOBS[job_id]


async def _run(job_id, parameters, worker):
    loop = asyncio.get_running_loop()
    semaphore = _SEMAPHORES.setdefault(loop, asyncio.Semaphore(2))
    async with semaphore:
        with _LOCK:
            _JOBS[job_id].update(status="running", started_at=_stamp())

        def progress(update):
            with _LOCK:
                job = _JOBS.get(job_id)
                if job is None or job["status"] != "running":
                    return
                current = job["progress"]
                update = dict(update)
                if "percent" in update:
                    update["percent"] = max(current.get("percent") or 0, min(99, max(0, float(update["percent"]))))
                for key in ("completed", "total", "failed_count"):
                    if key in update:
                        update[key] = max(0, int(update[key]))
                current.update(update)

        try:
            result = await worker(parameters, progress)
            with _LOCK:
                job = _JOBS[job_id]
                job.update(status="completed", result=result, finished_at=_stamp(), finished_timestamp=time.time())
                job["progress"].update(percent=100, phase="completed", message="All requested ticker work finished")
        except asyncio.CancelledError:
            with _LOCK:
                _JOBS[job_id].update(status="failed", error="The local server stopped before this scan finished. Run it again.", finished_at=_stamp(), finished_timestamp=time.time())
            raise
        except Exception as exc:
            message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            with _LOCK:
                _JOBS[job_id].update(status="failed", error=message or type(exc).__name__, finished_at=_stamp(), finished_timestamp=time.time())


def submit_job(kind, payload):
    if kind not in _HANDLERS:
        raise HTTPException(400, "Unknown scan kind")
    model, worker = _HANDLERS[kind]
    try:
        parameters = model.model_validate(payload).model_dump()
    except ValidationError as exc:
        raise HTTPException(422, detail=exc.errors(include_context=False))
    _cleanup()
    job_id = uuid.uuid4().hex
    with _LOCK:
        _JOBS[job_id] = {"job_id": job_id, "kind": kind, "status": "queued", "started_at": None,
                        "finished_at": None, "result": None, "error": None,
                        "progress": {"phase": "queued", "completed": 0, "total": 0, "ticker": None,
                                     "percent": 0, "message": "Waiting for a scan slot", "failed_count": 0}}
    task = asyncio.create_task(_run(job_id, parameters, worker))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return get_job(job_id)


@router.post("/{kind}")
async def start_scan_job(kind: str, payload: dict):
    return submit_job(kind, payload)


@router.get("/{job_id}")
async def read_scan_job(job_id: str):
    return get_job(job_id)
