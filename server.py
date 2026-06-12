"""Flask server: upload videos + stats, run evaluations, serve dashboard.

Routes:
  POST /api/upload              — multipart: video file + stats JSON → kicks off evaluation
  GET  /api/personas            — list of persona profiles
  GET  /api/evaluations         — list of past evaluation ids + metadata
  GET  /api/evaluations/<id>    — full evaluation result
  GET  /                        — serves the dashboard UI (static/index.html)

Storage is plain JSON files under ./storage. No DB, no infra.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

from admin_client import AdminClient, AdminError
from evaluator import evaluate_all, load_personas
from llm_backend import OPENROUTER_MODELS, get_backend


BASE_DIR = Path(__file__).parent
STORAGE_DIR = BASE_DIR / "storage"
EVAL_DIR = STORAGE_DIR / "evaluations"
VIDEO_DIR = STORAGE_DIR / "videos"
PERSONAS_PATH = BASE_DIR / "personas.json"
STATIC_DIR = BASE_DIR / "static"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Read-only PROD admin-API client (auth + player-data fetches). Single
# process, so one shared instance holding the cached JWT is fine.
admin = AdminClient()
ADMIN_DATE_FMT = "%Y-%m-%d %H:%M"


def _admin_default_range(days: int = 90) -> tuple[str, str]:
    now = datetime.now(timezone.utc)
    return (now - timedelta(days=days)).strftime(ADMIN_DATE_FMT), now.strftime(ADMIN_DATE_FMT)


# --------------------------------------------------------------------------- #
# In-memory job registry for the live request feed.
#
# Evaluations now run in a background thread so /api/upload returns
# immediately with an id, and the browser subscribes to events via SSE on
# /api/evaluation-progress/<id>. Each Job buffers all events so a reconnecting
# subscriber sees the full history, and supports multiple concurrent
# subscribers via a fan-out queue.
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    eval_id: str
    events: list[dict] = field(default_factory=list)
    subscribers: list[queue.Queue] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    status: str = "running"  # "running" | "done" | "error"
    error: str | None = None
    started_at: float = field(default_factory=time.time)

    def emit(self, ev: dict) -> None:
        with self.lock:
            self.events.append(ev)
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass  # subscriber too slow; they'll resync via .events on reconnect

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=10_000)
        with self.lock:
            backlog = list(self.events)
            self.subscribers.append(q)
        for ev in backlog:
            q.put_nowait(ev)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
# Cap the registry size; oldest non-running jobs evicted first. Prevents the
# process from growing unboundedly across many evaluations.
MAX_JOBS_RETAINED = 50


def _register_job(eval_id: str) -> Job:
    job = Job(eval_id=eval_id)
    with JOBS_LOCK:
        if len(JOBS) >= MAX_JOBS_RETAINED:
            # Evict the oldest finished job.
            for jid in list(JOBS):
                if JOBS[jid].status != "running":
                    del JOBS[jid]
                    break
        JOBS[eval_id] = job
    return job


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")


# --------------------------------------------------------------------------- #
# Static / index
# --------------------------------------------------------------------------- #


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/detail")
def detail():
    return send_from_directory(STATIC_DIR, "detail.html")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@app.get("/api/personas")
def api_personas():
    return jsonify({"personas": load_personas(str(PERSONAS_PATH))})


@app.get("/api/models")
def api_models():
    """Static default list. Falls back here when the live OpenRouter catalog
    fetch fails."""
    return jsonify({
        "openrouter_models": OPENROUTER_MODELS,
    })


# --------------------------------------------------------------------------- #
# OpenRouter live model catalog
#
# Hardcoding OpenRouter model IDs goes stale fast (and gets some wrong — the
# UI hits 500s when an upstream provider rejects an ID we guessed). We proxy
# OpenRouter's public /models endpoint and filter for vision-capable entries
# so the dropdown reflects what's actually available right now.
# --------------------------------------------------------------------------- #


@app.get("/api/openrouter/models")
def api_openrouter_models():
    """Fetch the live OpenRouter model catalog, filter to vision-capable.

    OpenRouter's /api/v1/models endpoint is public (no auth). Each entry
    includes architecture.input_modalities — we keep models that accept
    images. Curated IDs (the constants in OPENROUTER_MODELS) are floated to
    the top of the list so the default selection lands on something sensible
    even if OpenRouter's order changes underneath us.
    """
    cached_fallback = {"models": list(OPENROUTER_MODELS), "source": "fallback"}
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError) as e:
        return jsonify({**cached_fallback, "error": f"{type(e).__name__}: {e}"})

    items = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return jsonify({**cached_fallback, "error": "unexpected payload shape"})

    vision_ids: list[str] = []
    for m in items:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        if not isinstance(mid, str):
            continue
        arch = m.get("architecture") or {}
        modalities = arch.get("input_modalities") or []
        # OpenRouter sometimes lists modalities as e.g. ["text", "image"].
        if "image" in modalities:
            vision_ids.append(mid)

    if not vision_ids:
        return jsonify({**cached_fallback, "error": "no vision-capable models returned"})

    # Curated entries first (in their declared order), then the rest sorted.
    curated_set = set(OPENROUTER_MODELS)
    front = [m for m in OPENROUTER_MODELS if m in vision_ids]
    back = sorted(m for m in vision_ids if m not in curated_set)
    return jsonify({"models": front + back, "source": "live", "count": len(vision_ids)})


@app.get("/api/evaluations")
def api_evaluations():
    """List past evaluations, newest first by `created_at` timestamp.

    Filenames are random hex UUIDs so sorting on filename gives random order —
    we sort on the embedded timestamp instead. Files missing `created_at`
    (legacy or partial writes) fall to the bottom.
    """
    items = []
    for p in EVAL_DIR.glob("*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            items.append({
                "id": p.stem,
                "created_at": data.get("created_at"),
                "backend": data.get("backend"),
                "video_filename": data.get("video_filename"),
                "stats": data.get("stats", {}),
                "persona_count": len(data.get("personas", [])),
            })
        except (OSError, json.JSONDecodeError):
            continue
    items.sort(key=lambda e: e.get("created_at") or 0, reverse=True)
    return jsonify({"evaluations": items})


@app.get("/api/evaluations/<eval_id>")
def api_evaluation(eval_id: str):
    path = EVAL_DIR / f"{eval_id}.json"
    if not path.exists():
        return jsonify({"error": "not found"}), 404
    with open(path, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.post("/api/upload")
def api_upload():
    """Multipart upload: 'video' (file) + 'api_key' + optional 'model'.

    Kicks off a background evaluation and returns immediately with the
    eval_id + the SSE stream URL the UI subscribes to for live progress.
    """
    video_path: str | None = None
    video_filename: str | None = None
    eval_id = uuid.uuid4().hex[:12]
    file = request.files.get("video")
    if file and file.filename:
        suffix = Path(file.filename).suffix or ".mp4"
        video_path = str(VIDEO_DIR / f"{eval_id}{suffix}")
        file.save(video_path)
        video_filename = file.filename

    api_key = (request.form.get("api_key") or "").strip() or None
    model = (request.form.get("model") or "").strip() or None

    raw_step = (request.form.get("frame_step") or "").strip()
    try:
        frame_step = max(1, int(raw_step)) if raw_step else 30
    except ValueError:
        frame_step = 30
    raw_cap = (request.form.get("max_frames") or "").strip()
    try:
        max_frames = int(raw_cap) if raw_cap else None
    except ValueError:
        max_frames = None

    # Backend construction is sync — failures here (bad key, missing dep) must
    # surface synchronously so the UI doesn't open a stream to a dead job.
    try:
        backend = get_backend("openrouter", api_key=api_key, model=model)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400

    personas = load_personas(str(PERSONAS_PATH))
    job = _register_job(eval_id)

    def _worker() -> None:
        try:
            result = evaluate_all(
                backend,
                personas,
                video_path,
                frame_step=frame_step,
                max_frames=max_frames,
                event_sink=job.emit,
            )
            result["id"] = eval_id
            result["video_filename"] = video_filename
            result["created_at"] = int(time.time())
            result["requested_model"] = getattr(backend, "requested_model", None)
            result["model_used"] = getattr(backend, "current_model", None)
            result["fallback_log"] = getattr(backend, "fallback_log", [])

            out_path = EVAL_DIR / f"{eval_id}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)

            with job.lock:
                job.status = "done"
            job.emit({
                "type": "done",
                "eval_id": eval_id,
                "evaluation_url": f"/?id={eval_id}",
                "elapsed_seconds": result.get("elapsed_seconds"),
            })
        except Exception as e:  # noqa: BLE001 — capture for the UI; don't crash the thread
            msg = f"{type(e).__name__}: {e}"
            with job.lock:
                job.status = "error"
                job.error = msg
            job.emit({"type": "error", "message": msg})

    threading.Thread(target=_worker, name=f"eval-{eval_id}", daemon=True).start()

    return jsonify({
        "id": eval_id,
        "stream_url": f"/api/evaluation-progress/{eval_id}",
        "evaluation_url": f"/?id={eval_id}",
    })


@app.get("/api/evaluation-progress/<eval_id>")
def api_evaluation_progress(eval_id: str):
    """Server-Sent Events stream for the live request feed.

    Format: each event is `event: <type>\\ndata: <json>\\n\\n`. Reconnects
    receive the full event backlog so the UI can rebuild state. Heartbeats
    keep idle connections alive past proxy timeouts.
    """
    with JOBS_LOCK:
        job = JOBS.get(eval_id)
    if job is None:
        return jsonify({"error": "not found"}), 404

    q = job.subscribe()
    # Snapshot terminal status BEFORE we drain the queue — if the job already
    # finished, the queue ends with a done/error event and we should close
    # after delivering them. Otherwise we keep streaming live.
    with job.lock:
        terminal = job.status != "running"

    def stream():
        # Drain everything currently queued (backlog + any concurrent emits).
        # Then loop on the queue with a timeout for heartbeats. We exit on a
        # done/error event.
        try:
            while True:
                try:
                    ev = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    # If the job finished while we were idle, exit.
                    with job.lock:
                        if job.status != "running":
                            return
                    continue
                event_type = ev.get("type", "call")
                yield f"event: {event_type}\ndata: {json.dumps(ev)}\n\n"
                if event_type in ("done", "error"):
                    return
        finally:
            job.unsubscribe(q)

    # If the job was already terminal before we subscribed, our backlog still
    # includes the done/error event, so the stream() loop exits naturally.
    _ = terminal  # noted; nothing extra to do

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable proxy buffering if any
            "Connection": "keep-alive",
        },
    )


# --------------------------------------------------------------------------- #
# Admin API (PROD, read-only): login with a fresh 2FA, then pull player data.
# --------------------------------------------------------------------------- #


@app.get("/api/admin/status")
def api_admin_status():
    return jsonify(admin.status())


@app.post("/api/admin/login")
def api_admin_login():
    payload = (request.get_json(silent=True) if request.is_json else None) or {}
    username = (payload.get("username") or request.form.get("username") or "").strip()
    password = payload.get("password") or request.form.get("password") or ""
    code = (payload.get("code") or request.form.get("code") or "").strip()
    try:
        return jsonify(admin.login(username, password, code))
    except AdminError as e:
        return jsonify({"error": str(e), "need_login": e.need_login}), (e.status or 400)


@app.get("/api/admin/player")
def api_admin_player():
    """Read-only sweep for one player: identity + per-round history + stats."""
    nickname = (request.args.get("nickname") or "").strip()
    if not nickname:
        return jsonify({"error": "nickname required"}), 400
    search_type = (request.args.get("type") or "NICKNAME").strip().upper()
    default_from, default_to = _admin_default_range()
    date_from = (request.args.get("date_from") or "").strip() or default_from
    date_to = (request.args.get("date_to") or "").strip() or default_to
    try:
        return jsonify(admin.build_player_profile(nickname, date_from, date_to, search_type))
    except AdminError as e:
        return jsonify({"error": str(e), "need_login": e.need_login}), (e.status or 400)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
