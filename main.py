"""
podcast-dubber - FastAPI application
"""
import asyncio
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pipeline import run_pipeline

app = FastAPI(title="Podcast Dubber", description="YouTube video to Chinese-dubbed MP3", version="1.0.0")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict[str, Any]] = {}


class DubRequest(BaseModel):
    youtube_url: str


class DubResponse(BaseModel):
    job_id: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    stage: str | None = None
    progress: int | None = None
    created_at: str
    download_url: str | None = None
    error: str | None = None


def _make_progress_cb(job_id: str):
    def cb(stage: str, pct: int):
        if job_id in jobs:
            jobs[job_id]["stage"] = stage
            jobs[job_id]["progress"] = pct
    return cb


async def _run_job(job_id: str, youtube_url: str):
    jobs[job_id]["status"] = "processing"
    try:
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            None, run_pipeline, youtube_url, job_id, _make_progress_cb(job_id),
        )
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["output_path"] = output_path
    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


@app.post("/dub", response_model=DubResponse)
async def create_dub(req: DubRequest):
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {
        "status": "queued", "stage": None, "progress": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": None, "error": None,
    }
    asyncio.create_task(_run_job(job_id, req.youtube_url))
    return DubResponse(job_id=job_id)


@app.get("/dub/{job_id}")
async def get_dub_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    download_url = f"/dub/{job_id}/download" if job["status"] == "completed" and job["output_path"] else None
    return JobStatus(
        job_id=job_id, status=job["status"], stage=job["stage"],
        progress=job["progress"], created_at=job["created_at"],
        download_url=download_url, error=job["error"],
    )


@app.get("/dub/{job_id}/download")
async def download_dub(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = jobs[job_id]
    if job["status"] != "completed" or not job["output_path"]:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    file_path = Path(job["output_path"])
    if not file_path.exists():
        raise HTTPException(status_code=500, detail="Output file missing")
    return FileResponse(path=str(file_path), media_type="audio/mpeg", filename=f"{job_id}.mp3")


@app.get("/health")
async def health():
    return {"status": "ok"}
