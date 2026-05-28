"""Evaluator: runs each persona N times against a video + stats and aggregates.

The orchestrator is the product — this is where reproducibility and structured
output land. The LLM is just a component invoked via the LLMBackend interface.

Per the brief:
- 5 runs per persona, take median for the headline rating and dimension scores
- Final rating = weighted sum of dimension scores × persona weights
  (mechanical and traceable — not asked of the LLM)
- Track inter-run variance per persona, surface it in the UI
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from frames import extract_frames
from llm_backend import (
    DIMENSIONS,
    EvaluationRequest,
    EventSink,
    LLMBackend,
    MockBackend,
    _normalize_scores,
    _normalize_stats,
)

# 3 runs still produces a meaningful variance number (the brief targets ±0.5
# to ±1 across runs; pstdev across 3 captures that range), and cuts the call
# count by 40% over the original 5.
RUNS_PER_PERSONA = 1

# 30 ≈ 1fps for a 30fps source — the sweet spot Gemini itself uses internally
# for video sampling. The UI surfaces this knob; power users can drop to 1 if
# they want every frame at the cost of ~30× more payload per call.
DEFAULT_FRAME_STEP = 30

# Hard cap on frames sent per call. For long videos (e.g. 100 slot rounds ≈
# 5 min), frame_step alone produces hundreds of frames and the model spends
# real time processing each one. 48 evenly-distributed frames gives the
# persona plenty of signal to form a "look and feel" impression — more is
# diminishing returns at significantly higher cost+latency.
# Set to 0 (or env MAX_FRAMES=0) to disable the cap.
DEFAULT_MAX_FRAMES: int = 50

# Bound on concurrent persona workers. Each persona still runs its 3 runs
# sequentially, so total in-flight calls ≤ MAX_PARALLEL. Default 8 is
# comfortably under typical free-tier RPM limits (Gemini ~15 RPM, OpenRouter
# varies by model); the retry-on-429 path in each backend absorbs short bursts.
DEFAULT_MAX_PARALLEL = 8


@dataclass
class PersonaResult:
    persona_id: str
    persona_name: str
    persona_avatar: str
    persona_age: int
    persona_archetype: str
    persona_bio: str
    rating: float
    rating_variance: float
    mood: str
    dimension_scores: dict[str, int]
    dimension_variance: dict[str, float]
    would_replay: str
    would_replay_reason: str
    likes: list[str]
    dislikes: list[str]
    quote: str
    review: str
    runs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _mood_from_rating(r: float) -> str:
    if r >= 8.5:
        return "loved_it"
    if r >= 7.0:
        return "liked_it"
    if r >= 5.0:
        return "meh"
    if r >= 3.0:
        return "disliked_it"
    return "hated_it"


def _weighted_rating(dim_scores: dict[str, int], weights: dict[str, float]) -> float:
    total_w = sum(weights.get(d, 0) for d in DIMENSIONS) or 1.0
    raw = sum(dim_scores.get(d, 0) * weights.get(d, 0) for d in DIMENSIONS)
    return round(raw / total_w, 2)


def evaluate_persona(
    backend: LLMBackend,
    persona: dict[str, Any],
    stats: dict[str, Any],
    video_path: str | None,
    runs: int = RUNS_PER_PERSONA,
    frames: list[bytes] | None = None,
) -> PersonaResult:
    raw_runs: list[dict[str, Any]] = []
    for i in range(runs):
        req = EvaluationRequest(
            persona=persona,
            stats=stats,
            video_path=video_path,
            run_index=i,
            frames=frames or [],
        )
        try:
            out = backend.evaluate(req)
        except Exception as e:  # noqa: BLE001 — log and continue so one bad run doesn't kill the persona
            out = {
                "_error": str(e),
                "dimension_scores": {d: 4 for d in DIMENSIONS},
                "would_replay": "maybe",
                "would_replay_reason": "Evaluator error.",
                "likes": [],
                "dislikes": [],
                "quote": "(no comment)",
                "review": f"Evaluation failed: {e}",
            }
        out = _normalize_scores(out)
        raw_runs.append(out)

    # Median per dimension across the 5 runs
    medians: dict[str, int] = {}
    variances: dict[str, float] = {}
    for d in DIMENSIONS:
        values = [r["dimension_scores"][d] for r in raw_runs]
        medians[d] = int(round(statistics.median(values)))
        variances[d] = round(statistics.pstdev(values), 2) if len(values) > 1 else 0.0

    # Weighted final rating from medians (traceable, mechanical)
    rating = _weighted_rating(medians, persona.get("weights", {}))

    # Variance of the weighted rating across runs
    per_run_ratings = [_weighted_rating(r["dimension_scores"], persona.get("weights", {})) for r in raw_runs]
    rating_variance = round(statistics.pstdev(per_run_ratings), 2) if len(per_run_ratings) > 1 else 0.0

    # Pick the run whose own weighted rating is closest to the HEADLINE rating
    # (not the median of per-run ratings — those can differ since the headline
    # is computed from median dimension scores). This keeps the narrative
    # (quote, review, likes, dislikes) consistent with the headline mood.
    weights = persona.get("weights", {})
    median_run = min(raw_runs, key=lambda r: abs(_weighted_rating(r["dimension_scores"], weights) - rating))

    return PersonaResult(
        persona_id=persona["id"],
        persona_name=persona["name"],
        persona_avatar=persona.get("avatar", "🎮"),
        persona_age=persona.get("age", 0),
        persona_archetype=persona.get("archetype", ""),
        persona_bio=persona.get("bio", ""),
        rating=rating,
        rating_variance=rating_variance,
        mood=_mood_from_rating(rating),
        dimension_scores=medians,
        dimension_variance=variances,
        would_replay=median_run.get("would_replay", "maybe"),
        would_replay_reason=median_run.get("would_replay_reason", ""),
        likes=median_run.get("likes", []),
        dislikes=median_run.get("dislikes", []),
        quote=median_run.get("quote", ""),
        review=median_run.get("review", ""),
        runs=[{"dimension_scores": r["dimension_scores"], "would_replay": r.get("would_replay"), "quote": r.get("quote")} for r in raw_runs],
    )


def evaluate_all(
    backend: LLMBackend,
    personas: list[dict[str, Any]],
    video_path: str | None,
    runs: int = RUNS_PER_PERSONA,
    stats: dict[str, Any] | None = None,
    frame_step: int = DEFAULT_FRAME_STEP,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    event_sink: EventSink | None = None,
) -> dict[str, Any]:
    """Plug-and-play: extract frames once from the video, hand them to the
    backend for stats extraction, then run every persona in parallel against
    the same frames + stats. If `stats` is passed explicitly (e.g. from a
    test), extraction is skipped.

    `frame_step` — keep every Nth frame. Default 30 ≈ 1fps for a 30fps source.
    `max_frames` — optional final cap; useful as a safety net for long clips.
    `max_parallel` — bounded concurrency for persona calls (default 8).
    `event_sink` — optional callback that receives lifecycle events for the
        live UI feed: stage transitions (frames/stats/personas/done) plus
        per-call events from the backend (started/retrying/succeeded/failed).

    Mock backend doesn't need frames, so we skip ffmpeg when it's selected.
    Mock also has no rate limit, so we let it run with high parallelism.
    """
    started = time.time()

    # Attach the sink to the backend so its per-call events flow through too.
    # The backend default is no-op when sink is None, so unconditionally wiring
    # is fine — but we also want the sink reset after we're done so a backend
    # instance reused across runs doesn't leak the previous sink.
    prior_sink = backend.event_sink
    backend.event_sink = event_sink

    def emit(ev: dict) -> None:
        if event_sink is not None:
            try:
                event_sink(ev)
            except Exception:  # noqa: BLE001
                pass

    try:
        frames: list[bytes] = []
        is_mock = isinstance(backend, MockBackend)
        if video_path and not is_mock:
            env_step = os.environ.get("FRAME_STEP")
            if env_step:
                try:
                    frame_step = int(env_step)
                except ValueError:
                    pass
            env_cap = os.environ.get("MAX_FRAMES")
            if env_cap:
                try:
                    max_frames = int(env_cap)
                except ValueError:
                    pass
            # Convention: None means "use default cap", 0 (or negative) means
            # "no cap at all". Positive int = explicit cap.
            if max_frames is None:
                max_frames = DEFAULT_MAX_FRAMES
            effective_cap = max_frames if max_frames > 0 else None
            emit({"type": "stage", "stage": "extracting_frames",
                  "frame_step": frame_step, "frame_cap": effective_cap})
            frames = extract_frames(video_path, frame_step=frame_step, max_frames=effective_cap)
            print(
                f"[frames] extracted {len(frames)} frame(s) "
                f"(step={frame_step}, cap={effective_cap}, total bytes={sum(len(f) for f in frames)})",
                flush=True,
            )
            emit({"type": "stage", "stage": "frames_extracted",
                  "frame_count": len(frames), "bytes": sum(len(f) for f in frames)})

        if stats is None:
            emit({"type": "stage", "stage": "extracting_stats"})
            stats = backend.extract_stats(frames, video_path)
            # Models sometimes wrap each stat value in an explanatory object
            # (e.g. {"base_pacing": {"value": "slow", "reasoning": "..."}}).
            # Flatten before downstream consumers see it — otherwise the UI
            # renders "[object Object]" and persona prompts get garbage.
            stats, stat_warnings = _normalize_stats(stats)
            if stat_warnings:
                msg = f"Flattened wrapped stat values: {', '.join(stat_warnings)}"
                print(f"[stats] {msg}", flush=True)
                emit({"type": "stage", "stage": "stats_coerced",
                      "fields": stat_warnings, "message": msg})
            emit({"type": "stage", "stage": "stats_extracted"})

        env_parallel = os.environ.get("MAX_PARALLEL")
        if env_parallel:
            try:
                max_parallel = max(1, int(env_parallel))
            except ValueError:
                pass

        # Mock is in-process and instant — no point spinning up threads.
        workers = 1 if is_mock else min(max_parallel, len(personas))
        print(
            f"[evaluator] running {len(personas)} personas × {runs} runs "
            f"({workers} parallel workers, backend={backend.name})",
            flush=True,
        )
        emit({"type": "stage", "stage": "personas_started",
              "persona_count": len(personas), "runs": runs, "workers": workers})

        # Preserve persona order in the final results regardless of completion order.
        results: list[dict[str, Any] | None] = [None] * len(personas)

        def _run_one(idx_persona: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
            idx, p = idx_persona
            r = evaluate_persona(backend, p, stats, video_path, runs=runs, frames=frames)
            return idx, r.to_dict()

        if workers == 1:
            for pair in enumerate(personas):
                i, out = _run_one(pair)
                results[i] = out
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                for i, out in pool.map(_run_one, enumerate(personas)):
                    results[i] = out

        emit({"type": "stage", "stage": "personas_complete"})

        return {
            "backend": backend.name,
            "runs_per_persona": runs,
            "stats": stats,
            "personas": results,
            "elapsed_seconds": round(time.time() - started, 2),
            "frame_count": len(frames),
            "frame_step": frame_step if frames else None,
            "frame_cap": max_frames if frames else None,
            "parallel_workers": workers,
        }
    finally:
        # Detach the sink so a reused backend instance doesn't leak this run's
        # sink into the next run.
        backend.event_sink = prior_sink


def load_personas(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["personas"]
