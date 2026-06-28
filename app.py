from dotenv import load_dotenv
load_dotenv()

import os
import json
import shutil
import threading
import traceback
from typing import Optional
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import select_clips
import burn_subtitles



# ─────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Viral Clip Pipeline API")


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)

# In-memory job tracking: job_id -> status dict
# For production, replace with a real datastore (DB/Redis).
JOBS = {}
JOBS_LOCK = threading.Lock()

@app.get("/", tags=["UI"])
def serve_frontend():
    """Serves the main frontend UI."""
    return FileResponse("templates/index.html")

def _set_job(job_id: str, **kwargs):
    with JOBS_LOCK:
        JOBS.setdefault(job_id, {})
        JOBS[job_id].update(kwargs)


def _get_job(job_id: str) -> Optional[dict]:
    with JOBS_LOCK:
        return JOBS.get(job_id)


# ─────────────────────────────────────────────────────────────
# Request/response models
# ─────────────────────────────────────────────────────────────

class SelectClipsURLRequest(BaseModel):
    url: str
    options: Optional[dict] = None


class JobResponse(BaseModel):
    job_id: str
    status: str


# ─────────────────────────────────────────────────────────────
# Background workers
# ─────────────────────────────────────────────────────────────

def _run_selection(job_id: str, url: Optional[str], local_file_path: Optional[str], options: Optional[dict]):
    try:
        _set_job(job_id, status="running", stage="selection")

        def status_cb(msg: str):
            _set_job(job_id, message=msg)

        raw_clips, highlights, log_path = select_clips.execute_selection_workflow(
            url=url,
            local_file_path=local_file_path,
            options=options,
            status_callback=status_cb,
        )

        if not raw_clips:
            _set_job(job_id, status="failed", message="No clips were produced.", log_path=log_path)
            return

        # raw_path is now the SOURCE video, so derive job_dir from the log path
        # (output/<job_id>/DIAGNOSTIC_REPORT.txt -> output/<job_id>).
        job_dir = os.path.dirname(log_path)

        _set_job(
            job_id,
            status="selected",
            stage="selection_complete",
            job_dir=job_dir,
            highlights=highlights,
            raw_clips=raw_clips,
            total_clips=len(raw_clips),
            ready_clips=[],
            log_path=log_path,
            message="Clip selection complete. Ready for subtitle burning.",
        )
    except Exception as e:
        _set_job(job_id, status="failed", message=f"Selection crashed: {e}", error=traceback.format_exc())


def _run_subtitles(job_id: str, job_dir: str):
    try:
        _set_job(job_id, status="running", stage="subtitles")

        def status_cb(msg: str):
            _set_job(job_id, message=msg)

        # Called by the burn workflow the instant each clip is finished, so the UI
        # can show clips as they complete instead of waiting for the whole batch.
        def clip_cb(output_path: str, reason: str):
            with JOBS_LOCK:
                JOBS.setdefault(job_id, {})
                ready = JOBS[job_id].setdefault("ready_clips", [])
                fname = os.path.basename(output_path)
                if not any(c["filename"] == fname for c in ready):
                    ready.append({
                        "filename": fname,
                        "download_url": f"/jobs/{job_id}/clips/{fname}",
                        "reason": reason,
                    })

        final_clips, log_path = burn_subtitles.execute_subtitle_workflow(
            job_dir=job_dir,
            clip_callback=clip_cb,
            status_callback=status_cb,
        )

        if not final_clips:
            _set_job(job_id, status="failed", message="No subtitled clips were produced.", subtitle_log_path=log_path)
            return

        _set_job(
            job_id,
            status="done",
            stage="subtitles_complete",
            final_clips=final_clips,
            subtitle_log_path=log_path,
            message="Subtitle burning complete.",
        )
    except Exception as e:
        _set_job(job_id, status="failed", message=f"Subtitle burning crashed: {e}", error=traceback.format_exc())


def _run_full_pipeline(job_id: str, url: Optional[str], local_file_path: Optional[str], options: Optional[dict]):
    _run_selection(job_id, url, local_file_path, options)
    job = _get_job(job_id)
    if job and job.get("status") == "selected":
        _run_subtitles(job_id, job["job_dir"])


# ─────────────────────────────────────────────────────────────
# Endpoints: clip selection only
# ─────────────────────────────────────────────────────────────



@app.post("/select-clips/url", response_model=JobResponse)
def select_clips_from_url(payload: SelectClipsURLRequest):
    """Start clip selection from a video URL (yt-dlp supported source, or Google Drive link)."""
    job_id = str(uuid4())
    _set_job(job_id, status="queued", message="Job queued")

    thread = threading.Thread(
        target=_run_selection,
        args=(job_id, payload.url, None, payload.options),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


@app.post("/select-clips/upload", response_model=JobResponse)
def select_clips_from_upload(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
):
    """Start clip selection from an uploaded local video file.
    `options` (optional) should be a JSON string, e.g.
    '{"viral": true, "num_clips": "auto"}'. `num_clips` controls how many clips
    are cut: "auto" (default) scales to video length (about one per minute, up to
    ~50), or pass an integer (capped at 80)."""
    job_id = str(uuid4())

    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed_options = None
    if options:
        try:
            parsed_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="`options` must be valid JSON")

    _set_job(job_id, status="queued", message="Job queued", uploaded_path=upload_path)

    thread = threading.Thread(
        target=_run_selection,
        args=(job_id, None, upload_path, parsed_options),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


# ─────────────────────────────────────────────────────────────
# Endpoints: subtitle burning only (for an already-selected job)
# ─────────────────────────────────────────────────────────────

@app.post("/burn-subtitles/{job_id}", response_model=JobResponse)
def burn_subtitles_for_job(job_id: str):
    """Start subtitle burning for a job that has already completed clip selection
    (status must be 'selected')."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") != "selected":
        raise HTTPException(
            status_code=400,
            detail=(
                f"Job is not ready for subtitle burning (status='{job.get('status')}'). "
                f"Run /select-clips first and wait for status='selected'."
            ),
        )

    job_dir = job["job_dir"]
    _set_job(job_id, status="queued", message="Subtitle job queued")

    thread = threading.Thread(
        target=_run_subtitles,
        args=(job_id, job_dir),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


# ─────────────────────────────────────────────────────────────
# Endpoints: combined pipeline (selection + subtitles, end-to-end)
# ─────────────────────────────────────────────────────────────

@app.post("/process/url", response_model=JobResponse)
def process_from_url(payload: SelectClipsURLRequest):
    """Run the full pipeline (selection + subtitle burning) for a video URL."""
    job_id = str(uuid4())
    _set_job(job_id, status="queued", message="Job queued")

    thread = threading.Thread(
        target=_run_full_pipeline,
        args=(job_id, payload.url, None, payload.options),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


@app.post("/process/upload", response_model=JobResponse)
def process_from_upload(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
):
    """Run the full pipeline (selection + subtitle burning) for an uploaded video file."""
    job_id = str(uuid4())

    upload_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed_options = None
    if options:
        try:
            parsed_options = json.loads(options)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="`options` must be valid JSON")

    _set_job(job_id, status="queued", message="Job queued", uploaded_path=upload_path)

    thread = threading.Thread(
        target=_run_full_pipeline,
        args=(job_id, None, upload_path, parsed_options),
        daemon=True,
    )
    thread.start()

    return JobResponse(job_id=job_id, status="queued")


# ─────────────────────────────────────────────────────────────
# Endpoints: status + results
# ─────────────────────────────────────────────────────────────

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Avoid dumping huge transcript/word-level objects back to the client
    safe_job = {k: v for k, v in job.items() if k != "raw_clips"}
    if "raw_clips" in job:
        safe_job["clips"] = [
            {
                "index": c["index"],
                "start": c["start"],
                "end": c["end"],
                "score": c.get("score"),
                "reason": c.get("reason"),
            }
            for c in job["raw_clips"]
        ]
    return JSONResponse(content=safe_job)


@app.get("/jobs/{job_id}/clips")
def list_clips(job_id: str):
    """List downloadable clips. Returns clips AS THEY FINISH (partial) while the job
    is still rendering, plus a `complete` flag and progress counts. The frontend
    polls this to show clips the moment each one is ready."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    final_clips = job.get("final_clips")
    ready = job.get("ready_clips", [])
    total = job.get("total_clips")

    if final_clips:
        clips = [
            {"filename": os.path.basename(c),
             "download_url": f"/jobs/{job_id}/clips/{os.path.basename(c)}"}
            for c in final_clips
        ]
        return {"job_id": job_id, "complete": True,
                "ready": len(clips), "total": total or len(clips), "clips": clips}

    # Still rendering — hand back whatever has finished so far.
    return {"job_id": job_id, "complete": False,
            "ready": len(ready), "total": total,
            "clips": [{"filename": c["filename"], "download_url": c["download_url"]} for c in ready]}


@app.get("/jobs/{job_id}/clips.zip")
def download_all_clips(job_id: str):
    """Bundle every finished clip for a job into a single ZIP and stream it back.
    Powers the frontend 'Download all' button."""
    import io, zipfile
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    final_clips = job.get("final_clips")
    if not final_clips:
        raise HTTPException(status_code=400, detail="Subtitled clips not ready yet")

    # Build the zip in memory (clips are small; for very large batches switch to a
    # temp file on disk). Each clip is added once under its basename.
    buf = io.BytesIO()
    added = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        # ZIP_STORED (no recompression) — mp4 is already compressed, so this is
        # fast and avoids burning CPU re-zipping video.
        for c in final_clips:
            if os.path.exists(c):
                zf.write(c, arcname=os.path.basename(c))
                added += 1

    if added == 0:
        raise HTTPException(status_code=404, detail="No clip files found on disk")

    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{job_id}_clips.zip"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)


@app.get("/jobs/{job_id}/clips/{filename}")
def download_clip(job_id: str, filename: str):
    """Download a specific clip file by name."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    job_dir = job.get("job_dir")
    if not job_dir:
        raise HTTPException(status_code=400, detail="Job directory not available yet")

    # Prevent path traversal - only allow plain filenames
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    clip_path = os.path.join(job_dir, "clips", filename)
    if not os.path.exists(clip_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(clip_path, media_type="video/mp4", filename=filename)


@app.get("/jobs/{job_id}/highlights")
def get_highlights(job_id: str):
    """Return the AI-selected highlight metadata (start/end/score/reason) for a job."""
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    highlights = job.get("highlights")
    if highlights is None:
        raise HTTPException(status_code=400, detail="Highlights not ready yet")

    return {"job_id": job_id, "highlights": highlights}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str, keep_files: bool = False):
    """Remove a job when the user clicks 'remove'.

    Opt-in cleanup: nothing is deleted automatically anywhere else, so downloaded
    videos and generated clips persist on disk until this endpoint is called.

    Deletes:
      - the job's output directory (raw_video, audio, transcripts, clips, logs)
      - the uploaded source file (if the job came from an upload)
      - the in-memory job entry
    Pass ?keep_files=true to only forget the job in memory but leave files on disk.
    """
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    removed = {"job_dir": None, "uploaded_path": None, "memory": False}
    errors = []

    if not keep_files:
        job_dir = job.get("job_dir")
        if job_dir and os.path.isdir(job_dir):
            try:
                shutil.rmtree(job_dir)
                removed["job_dir"] = job_dir
            except OSError as e:
                errors.append(f"job_dir: {e}")

        uploaded_path = job.get("uploaded_path")
        if uploaded_path and os.path.exists(uploaded_path):
            try:
                os.remove(uploaded_path)
                removed["uploaded_path"] = uploaded_path
            except OSError as e:
                errors.append(f"uploaded_path: {e}")

    with JOBS_LOCK:
        if job_id in JOBS:
            del JOBS[job_id]
            removed["memory"] = True

    if errors:
        return JSONResponse(
            status_code=207,
            content={"job_id": job_id, "removed": removed,
                     "message": "Job removed with some errors.", "errors": errors},
        )
    return {"job_id": job_id, "removed": removed,
            "message": "Job removed." if not keep_files else "Job forgotten (files kept on disk)."}


@app.get("/health")
def health():
    return {"status": "ok"}
