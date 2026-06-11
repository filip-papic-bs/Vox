# Vox

A panel of AI players that watches a game and votes — each persona scores it through
their own preferences, and the dashboard shows where they agree and where they split.

16 AI "players", each with a distinct personality, watch a game video and
produce honest, in-character feedback across 12 dimensions. The dashboard
renders one card per persona.

**Plug-and-play:** drop a video, get persona reactions. The backend extracts
stats (pacing axes, brightness, theme, win rate, bonus rate, etc.) from the
video itself — you don't need to type anything into a stats form.

See `PROJECT_BRIEF.md` for the original spec.

## What's in here

```
personas.json         16 personas with traits, preferences, dimension weights
llm_backend.py        LLMBackend abstraction; OpenRouterBackend
                      The backend implements:
                        - extract_stats(video)  → stats dict
                        - evaluate(persona, stats, video, run) → feedback dict
evaluator.py          Calls extract_stats once, then runs 5 evaluations per
                      persona, takes median, computes the weighted final
                      rating mechanically
server.py             Flask server: upload, persona list, evaluation results
static/               Plain HTML + JS + CSS dashboard (no React, no build step)
storage/              JSON evaluation results + uploaded videos (gitignored)
```

## Dimensions (12)

Each persona scores a game on these axes; the headline rating is a weighted
sum of the 12 medians using per-persona weights from `personas.json`.

- **visual** — graphical style / brightness fit
- **theme** — theme keyword match against persona likes/dislikes
- **win_frequency** — hit-rate satisfaction
- **win_size** — typical payout magnitude
- **volatility_feel** — variance/swing feel
- **bonus_frequency** — how often the bonus triggers
- **bonus_quality** — bonus design (shallow → iconic), not just rate
- **base_pacing** — spin-to-spin tempo (slow vs fast)
- **anticipation** — buildup, near-misses, tension before reveals
- **session_flow** — arc over a session (flat / steady_build /
  peaks_and_valleys / front_loaded / back_loaded)
- **mechanic_depth** — feature complexity and layering
- **feature_variety** — count of distinct mechanics

Three of the dimensions are pacing-focused, so a fast-but-flat game and a
slow-but-suspenseful game score very differently — different personas pick
up on each axis.

Audio is intentionally out of scope.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Open <http://127.0.0.1:5000>. Evaluations run through **OpenRouter** — one
API key, many vision-capable models (Google, Anthropic, OpenAI, Meta, …).
Paste your OpenRouter key into the field on the page (it persists in browser
`localStorage`, never written to disk on the server) and pick a model. The
model watches the actual video and extracts the stats itself.

There's no stats form — that's the whole point. Drop the video and go.

## API key

You can provide the OpenRouter API key in either of these ways:

1. **In the UI field** (recommended) — typed into the dashboard, persisted in
   browser localStorage only. The server never stores it, never logs it to disk.
   Lives on your machine.
2. **Env var** — `export OPENROUTER_API_KEY=...` before starting the server.

The form field overrides the env var if both are set.

## Configuration

- `OPENROUTER_API_KEY` — fallback OpenRouter key
- `OPENROUTER_MODEL` — default model when none is selected in the form
  (defaults to `google/gemini-2.5-flash`). Pin to a specific snapshot in
  production so persona behavior doesn't drift between model updates.
- `OPENROUTER_MAX_RETRIES` — retries on 503/429/502/504 (default 6)
- `OPENROUTER_INITIAL_BACKOFF` — seconds to wait before the first retry (default 2.0)
- `OPENROUTER_MAX_BACKOFF` — backoff cap per retry (default 30.0)
- `OPENROUTER_TIMEOUT` — per-request timeout in seconds (default 120)
- `OPENROUTER_MAX_TOKENS` — output token cap per call (default 2000)
- `PORT` — server port, default 5000

## How reproducibility works

- `temperature = 0` on the backend
- Strict JSON output (`response_format: json_object`, with schema enforced in
  the prompt and normalized on parse)
- 5 runs per persona, median taken for headline numbers and dimension scores
- Per-run variance surfaced in the detail view (`±N` next to each score)
- Final rating is a **weighted sum** of dimension scores × persona weights —
  computed mechanically in the orchestrator, not asked of the LLM, so it's
  fully traceable

## How "no training-data recall" works

- The video is the primary input but the stats step decouples objective facts
  (win rate, bonus rate, themes) from the per-persona reaction step
- Personas score 5 dimensions, not a holistic vibe
- Scores 5 and 6 are forbidden — personas must pick a side
- The prompt explicitly tells the model not to mention the game's brand

## Trade-off the brief warned about

The original brief said "the LLM is the critic, not the statistician" and
preferred external structured stats over the LLM counting events in the video.
We do the latter here for the plug-and-play UX. For production accuracy you'd
ideally still pass structured stats (real win counts from the game backend)
— `evaluate_all(backend, personas, video_path, stats=...)` already supports
that path; the UI just doesn't expose it.
