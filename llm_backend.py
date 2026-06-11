"""LLM backend abstraction.

The evaluator only depends on this interface, so vendors can be swapped without
touching the rest of the system. One implementation ships:

  - OpenRouterBackend — vision-capable (OpenAI-compatible API). Single key,
                        many providers (Google, Anthropic, OpenAI, DeepSeek,
                        Meta, ...). Frames sent as base64 image_url parts.

The backend consumes frames (a list of JPEG bytes pre-extracted from the video
by the evaluator), not the raw video file.
"""

from __future__ import annotations

import base64
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import count as _id_counter
from typing import Any, Callable

# A backend's `event_sink`, when set, is called with one dict per call lifecycle
# event: started / succeeded / failed / retrying. The dict shape is documented
# in OpenRouterBackend._chat_completion. Backends remain usable with no sink
# attached (defaults to a no-op).
EventSink = Callable[[dict], None]


# Scores live in 1..10 but 5 and 6 are forbidden — personas must pick a side.
VALID_SCORES = [1, 2, 3, 4, 7, 8, 9, 10]


SYSTEM_INSTRUCTIONS = """\
You are evaluating a game video as a specific persona. Stay strictly in
character. Your job is NOT to count events in the video — the structured stats
provided alongside it are authoritative. Your job is to react to those facts
in character.

Rate the game on FIFTEEN dimensions:
  visual, theme, win_frequency, win_size, volatility_feel,
  bonus_frequency, bonus_quality,
  base_pacing, anticipation, session_flow,
  mechanic_depth, feature_variety,
  anticipation_honesty, celebration_calibration, clarity.

Each score is an integer from 1 to 10, but you MUST NOT use 5 or 6 — pick a
side. Use the persona's preferences to decide what a high or low score means
for THIS persona (e.g. a persona who hates fast pacing should score a fast
game LOW on base_pacing, not high; a persona who lives for big slow buildups
should score a snappy game LOW on anticipation).

CRITICAL — avoid mid-rating drift: if the persona's preference clearly
matches the stat, score 8–10. If it clearly conflicts, score 1–4. Do NOT
default to 7 for everything; that's lazy. Personas with strong opinions on a
dimension MUST commit. Reserve 7s only for dimensions the persona genuinely
has no opinion on (preference = "medium" with no strong context).

Three of the dimensions are about pacing:
  - base_pacing: spin-to-spin tempo (slow vs fast)
  - anticipation: tension, near-misses, buildup before reveals
  - session_flow: shape of the arc over a session (flat / steady_build /
    peaks_and_valleys / front_loaded / back_loaded)
Score these independently — a game can have fast base pacing but flat
session_flow, or slow base pacing but high anticipation.

The three NEW dimensions are about QUALITY, not intensity. Read carefully:

  - anticipation_honesty (different from anticipation!): does anticipation
    feel earned, or fake/manipulative? A teaser spin on reel 5 after reel 4
    has already killed the combination is DISHONEST anticipation. Unusual
    reel-stop order (e.g. reel 5 stopping before reel 4) reads as a bug.
    Match against persona's `fake_anticipation_sensitivity` and
    `reel_rhythm_strictness`. A persona with HIGH sensitivity and a
    convention-breaking game = score LOW even if the anticipation amount
    matches their taste.

  - celebration_calibration: are win celebrations scaled appropriately to
    payout size? A flashy 5-second animation for a 0.5x win is BAD
    calibration. Tiered animation (small wins quiet, big wins celebratory,
    mega wins cinematic) is GOOD calibration. Match against persona's
    `small_win_celebration_tolerance`. Personas with LOW tolerance want
    proportional celebration; HIGH-tolerance personas (Karen, Yuki, Margaret)
    don't care and score ~7.

  - clarity: at any given moment of play, can the player tell what's
    happening and why? Too many overlapping anticipation animations,
    indistinguishable teaser types, or opaque bonus rules all hurt clarity.
    Match against persona's `feature_learnability_threshold` and
    `visual_readability_preference`. A persona with `simple` threshold and a
    cluttered game = LOW score. A `complex`-pref persona is fine with noise.

Bias your likes/dislikes and review toward GAMEPLAY observations — mechanics,
volatility feel, feature variety, bonus pacing, win pattern, anticipation
moments, session arc, animation cadence, on-screen readability — rather than
only visuals. Visuals get one or two mentions at most; most of the review is
about how the game PLAYS for this persona. Do NOT mention audio or sound —
that is out of scope. Do NOT comment on autoplay behavior — the video only
shows what the player chose to capture.

Output strict JSON matching the requested schema. No prose outside JSON.
"""


DIMENSIONS = [
    "visual",
    "theme",
    "win_frequency",
    "win_size",
    "volatility_feel",
    "bonus_frequency",
    "bonus_quality",
    "base_pacing",
    "anticipation",
    "session_flow",
    "mechanic_depth",
    "feature_variety",
    # Three axes added after a round of real-player feedback exposed gaps:
    # personas were scoring intensity (how MUCH anticipation/pacing) but not
    # quality (is the anticipation HONEST, are wins celebrated PROPORTIONATELY,
    # is what's happening on screen LEGIBLE).
    "anticipation_honesty",
    "celebration_calibration",
    "clarity",
]


RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "dimension_scores": {
            "type": "object",
            "properties": {d: {"type": "integer"} for d in DIMENSIONS},
            "required": list(DIMENSIONS),
        },
        "would_replay": {"type": "string", "enum": ["yes", "no", "maybe"]},
        "would_replay_reason": {"type": "string"},
        "likes": {"type": "array", "items": {"type": "string"}},
        "dislikes": {"type": "array", "items": {"type": "string"}},
        "quote": {"type": "string"},
        "review": {"type": "string"},
    },
    "required": [
        "dimension_scores",
        "would_replay",
        "would_replay_reason",
        "likes",
        "dislikes",
        "quote",
        "review",
    ],
}


@dataclass
class EvaluationRequest:
    persona: dict[str, Any]
    stats: dict[str, Any]
    video_path: str | None
    run_index: int
    # Pre-extracted JPEG frames; the backend consumes these instead of the
    # raw video.
    frames: list[bytes] = field(default_factory=list)


# Stats schema the backend must produce from the video itself.
# Pacing is split into three independent axes — base_pacing, anticipation,
# session_arc — to give personas real surface area to disagree on.
STATS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rounds_played": {"type": "integer"},
        "win_rate": {"type": "number"},
        "bonus_rate": {"type": "number"},
        "avg_win_size_multiplier": {"type": "number"},
        "biggest_win_multiplier": {"type": "number"},
        "base_pacing": {"type": "string", "enum": ["slow", "medium", "fast"]},
        "anticipation_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "session_arc": {
            "type": "string",
            "enum": ["flat", "steady_build", "peaks_and_valleys", "front_loaded", "back_loaded"],
        },
        "near_miss_frequency": {"type": "string", "enum": ["low", "medium", "high"]},
        "dry_streak_feel": {"type": "string", "enum": ["low", "medium", "high"]},
        "visual_brightness": {"type": "string", "enum": ["low", "medium", "high"]},
        "theme": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "volatility": {"type": "string", "enum": ["low", "medium", "high"]},
        "mechanic_complexity": {"type": "string", "enum": ["simple", "medium", "complex"]},
        "feature_variety": {"type": "string", "enum": ["few", "balanced", "many"]},
        "bonus_quality": {
            "type": "string",
            "enum": ["shallow", "decent", "memorable", "iconic"],
        },
        # Quality-of-experience signals added after real-player feedback
        # exposed that "how much" stats weren't enough — players reacted
        # strongly to whether anticipation felt earned, whether small wins
        # were over-celebrated, and whether what was happening on screen
        # was legible at a glance.
        "small_win_celebration_intensity": {"type": "string", "enum": ["low", "medium", "high"]},
        "win_tier_distinction": {"type": "string", "enum": ["weak", "clear", "strong"]},
        "anticipation_honesty": {"type": "string", "enum": ["low", "medium", "high"]},
        "reel_stop_order": {"type": "string", "enum": ["conventional", "unusual"]},
        "visual_readability": {"type": "string", "enum": ["low", "medium", "high"]},
        "bonus_rule_clarity": {"type": "string", "enum": ["confusing", "mixed", "clear"]},
        "win_display_format": {"type": "string", "enum": ["multiplier", "currency", "both"]},
    },
    "required": [
        "rounds_played",
        "win_rate",
        "bonus_rate",
        "base_pacing",
        "anticipation_level",
        "session_arc",
        "visual_brightness",
        "theme",
        "themes",
        "volatility",
        "mechanic_complexity",
        "feature_variety",
        "bonus_quality",
        "small_win_celebration_intensity",
        "win_tier_distinction",
        "anticipation_honesty",
        "reel_stop_order",
        "visual_readability",
        "bonus_rule_clarity",
        "win_display_format",
    ],
}


STATS_EXTRACTION_PROMPT = """\
Watch this game video and return a JSON object describing it in OBJECTIVE
terms. You are NOT the critic here — just a careful observer. Do not mention
the game's brand or title. Do not describe audio — leave it out entirely.

OUTPUT SHAPE — read carefully:
Return a SINGLE flat JSON object whose VALUES are primitives (string, number,
boolean) or arrays of primitives. Do NOT wrap any value in an explanatory
object like {"value": "slow", "reasoning": "..."} — just return "slow".
Do NOT nest the entire response under a top-level key like {"stats": {...}}.
Each field listed below maps directly to one key in the returned object
with a primitive value.

Extract:
- rounds_played: integer estimate of how many spins / rounds you can count
- win_rate: fraction 0..1 — what fraction of rounds resulted in a win
- bonus_rate: fraction 0..1 — fraction of rounds that triggered a bonus / feature
- avg_win_size_multiplier: number — typical win size as multiple of bet
- biggest_win_multiplier: number — biggest single-round win as multiple of bet

PACING (three independent axes — judge each separately):
- base_pacing: "slow" | "medium" | "fast" — raw spin-to-spin tempo. How long
  is one round from press to settled result? Slow = >3s, medium = 1.5-3s,
  fast = <1.5s including any auto-spin behaviour.
- anticipation_level: "low" | "medium" | "high" — how often the game creates
  tension BEFORE the result lands (near-miss reels, slow drop-ins, lingering
  reveals, building music swells visible on screen, suspended frames before
  payouts). Low = result lands instantly. High = the game makes you wait for it.
- session_arc: "flat" (steady throughout) | "steady_build" (slow ramp toward
  something) | "peaks_and_valleys" (big swings, droughts then bursts) |
  "front_loaded" (best stuff happens early) | "back_loaded" (boring base, all
  payoff in the bonus). Judge from the overall shape of the video.
- near_miss_frequency: "low" | "medium" | "high" — how often you see clear
  near-miss patterns (e.g. two scatters where three would trigger, almost-
  aligned reels).
- dry_streak_feel: "low" (rarely goes more than ~5 spins without a win) |
  "medium" | "high" (long droughts visible). Look at the longest losing run
  in the video.

VISUAL + THEME:
- visual_brightness: "low", "medium", or "high" — overall brightness and saturation
- theme: ONE short keyword for the dominant theme (e.g. "candy", "egypt",
  "cyberpunk", "fruit", "horror", "fantasy", "nature", "viking")
- themes: 3-5 short keyword tags describing aesthetics + content

WIN PROFILE + MECHANICS:
- volatility: "low" (many small wins, rare losses) | "medium" | "high" (long
  droughts, rare large wins) — judge from the win pattern AND the size
  distribution, not just hit rate.
- mechanic_complexity: "simple" (one paytable, no modifiers) | "medium" (some
  modifiers, scatters, wilds) | "complex" (multiple mechanics, layered features)
- feature_variety: "few" (one bonus type) | "balanced" (2-3 features) |
  "many" (multiple distinct features: free spins, multipliers, sticky wilds,
  cascades, expanding reels, etc.)
- bonus_quality: "shallow" (a few free spins, nothing else) | "decent" |
  "memorable" (clear identity, escalation, visible math) | "iconic" (a feature
  someone would recommend the game for on its own merit).

QUALITY-OF-EXPERIENCE (these are what experienced players actually complain
about — judge carefully, don't default to "medium"):

- small_win_celebration_intensity: "low" (sub-1x wins are quiet — quick
  highlight, no full animation) | "medium" | "high" (every win, even 0.5x,
  triggers a big animation, sound burst, or screen-shake). Watch what happens
  on the SMALLEST wins specifically.

- win_tier_distinction: "weak" (small, big, and mega wins look the same — all
  loud, or all quiet) | "clear" (you can tell at a glance which tier you hit)
  | "strong" (clear three-tier system: subtle for small, enhanced for big,
  cinematic for mega). This is about VISUAL/animation tiering, not payout
  amount.

- anticipation_honesty: "low" (teaser spins fire even when no win is possible
  — e.g. anticipation on reel 5 after reel 4 has already killed the
  combination; teaser frames on early reels for combinations that can't
  exist) | "medium" | "high" (teasers only fire when a meaningful win is
  still mathematically possible). If you can't tell from the video, default
  to "medium".

- reel_stop_order: "conventional" (strict left-to-right: reel 1 → 2 → 3 → 4
  → 5, no skipping, no out-of-order stops) | "unusual" (skipped reels,
  middle-reel delays, reel 5 stopping before reel 4). Conventional is the
  expectation for 99% of slots.

- visual_readability: "low" (multiple overlapping animations make it hard to
  tell WHAT just paid and WHY; teaser types are indistinguishable; effects
  pile up) | "medium" | "high" (clean, separated cues — at any moment you
  can tell what's happening on screen). This is moment-to-moment legibility,
  NOT rule complexity.

- bonus_rule_clarity: "confusing" (e.g. 9 different color combinations that
  award different spin counts and the player needs to consult a paytable)
  | "mixed" (some quirks but generally understandable) | "clear" (one quick
  on-screen explanation is enough to play the bonus).

- win_display_format: "multiplier" (shows only ×N) | "currency" (shows only
  $/€ value) | "both" (shows ×N AND money — saves the player from mental
  math, widely preferred).

If an exact count is impossible, estimate honestly. Output strict JSON only.
"""


class LLMBackend(ABC):
    name: str = "abstract"
    # Optional live-progress hook. When set, the backend emits per-call events
    # so the UI can render a live request feed.
    event_sink: EventSink | None = None

    def _emit(self, ev: dict) -> None:
        sink = self.event_sink
        if sink is not None:
            try:
                sink(ev)
            except Exception:  # noqa: BLE001 — never let the UI hook break a call
                pass

    @abstractmethod
    def evaluate(self, req: EvaluationRequest) -> dict[str, Any]:
        """Return a dict matching RESPONSE_SCHEMA."""

    @abstractmethod
    def extract_stats(
        self,
        frames: list[bytes],
        video_path: str | None = None,
    ) -> dict[str, Any]:
        """Inspect the frames, return a stats dict matching STATS_SCHEMA."""


def build_user_prompt(persona: dict[str, Any], stats: dict[str, Any]) -> str:
    """Persona profile + anonymized stats — never includes the game's name."""
    sanitized_stats = {
        k: v for k, v in stats.items()
        if k.lower() != "game_name" and not k.lower().startswith("audio")
    }
    persona_keys = (
        "name", "age", "archetype", "bio",
        "preferences", "betting_style", "patience", "session_length", "voice",
    )
    return (
        "PERSONA PROFILE:\n"
        f"{json.dumps({k: persona[k] for k in persona_keys if k in persona}, indent=2)}\n\n"
        "OBJECTIVE STATS (authoritative — do not recount from video):\n"
        f"{json.dumps(sanitized_stats, indent=2)}\n\n"
        "Watch the video and produce a JSON evaluation as this persona on FIFTEEN\n"
        "dimensions (visual, theme, win_frequency, win_size, volatility_feel,\n"
        "bonus_frequency, bonus_quality, base_pacing, anticipation, session_flow,\n"
        "mechanic_depth, feature_variety, anticipation_honesty,\n"
        "celebration_calibration, clarity).\n\n"
        "Use the stats above as ground truth for win rate, win size, volatility,\n"
        "bonus rate, bonus_quality, mechanic_complexity, feature_variety,\n"
        "base_pacing, anticipation_level, session_arc, near_miss_frequency,\n"
        "dry_streak_feel, small_win_celebration_intensity, win_tier_distinction,\n"
        "anticipation_honesty, reel_stop_order, visual_readability,\n"
        "bonus_rule_clarity, and win_display_format. Use the video only for theme\n"
        "and visual impressions. Do NOT mention audio or sound — it is out of\n"
        "scope. Do NOT comment on autoplay behavior — out of scope.\n\n"
        "Compare the persona's preferences (visual_brightness, theme_likes/dislikes,\n"
        "win_frequency, win_size, volatility, bonus_frequency, bonus_quality,\n"
        "pacing, anticipation, session_flow, mechanic_complexity, feature_variety,\n"
        "dry_streak_tolerance, near_miss_appetite, small_win_celebration_tolerance,\n"
        "fake_anticipation_sensitivity, slot_literacy, reel_rhythm_strictness,\n"
        "bonus_ceremony_preference, feature_learnability_threshold,\n"
        "visual_readability_preference, win_display_preference, session_length,\n"
        "patience, betting_style) against the stats and react accordingly.\n\n"
        "CRITICAL — the persona is a REAL PLAYER reacting to the LOOK AND FEEL\n"
        "of the game, not a spreadsheet jockey. Real players don't count win\n"
        "percentages or compute volatility — they REACT to what they see and\n"
        "feel. Use the frames as primary evidence; the stats above are\n"
        "scaffolding. Especially for the three quality dimensions:\n"
        "- anticipation_honesty: WATCH the teaser/anticipation moments. Did\n"
        "  the game make you wait when nothing could pay? Did reel 5 stop\n"
        "  before reel 4? React to what you saw, not to the stat label.\n"
        "  Penalize hard if the persona is sensitive (high\n"
        "  fake_anticipation_sensitivity / strict reel_rhythm_strictness) AND\n"
        "  the video shows dishonest behavior.\n"
        "- celebration_calibration: WATCH how the game celebrates wins of\n"
        "  different sizes. Is a 0.5x win celebrated like a 50x win? Does the\n"
        "  game have visibly distinct animations for small/big/mega? React to\n"
        "  the actual animations; low-tolerance personas reward disciplined\n"
        "  calibration, high-tolerance personas (Karen, Yuki, Margaret, Pat)\n"
        "  don't care and score ~7.\n"
        "- clarity: WATCH a few moments and ask: can you tell what's happening\n"
        "  on screen and why? Are multiple animations overlapping in a way\n"
        "  that obscures what just paid? Are anticipation cues distinguishable\n"
        "  from each other? Simple-threshold personas need it readable;\n"
        "  complex-threshold personas (Lena, Sofia, Elena, Derek) tolerate\n"
        "  noise. The stat labels (visual_readability, bonus_rule_clarity)\n"
        "  are hints — your eyes are the judge.\n\n"
        "Score sharply: if a preference clearly conflicts with reality, the\n"
        "score for that dimension is 1-4; if it clearly aligns, 8-10. Score 7\n"
        "ONLY for dimensions this persona genuinely has no opinion on. Most\n"
        "personas will end up with at least one 1-3 score and at least one 8-10\n"
        "score on the same game.\n\n"
        "Reminder: scores are 1-10, but 5 and 6 are FORBIDDEN. The likes/dislikes\n"
        "and review should be mostly about gameplay — mechanics, win pattern,\n"
        "volatility feel, bonus design, pacing, anticipation honesty, celebration\n"
        "cadence, on-screen clarity — and only briefly about visuals/theme. Keep\n"
        "the quote to one sentence written in this persona's voice.\n\n"
        "OUTPUT SHAPE — CRITICAL:\n"
        "`dimension_scores` MUST be an object whose values are PLAIN INTEGERS\n"
        "between 1 and 10, with no 5 or 6. Do NOT wrap scores in objects like\n"
        '{"score": 7, "reason": "..."}. Do NOT add extra keys. Example of the\n'
        "correct shape:\n"
        '  "dimension_scores": {"visual": 8, "theme": 7, "win_frequency": 4, ...}\n'
        "Every dimension listed above must appear as a flat integer.\n\n"
        "NARRATIVE FIELDS — these are MANDATORY, NEVER leave them empty:\n"
        "- quote: ONE complete sentence in the persona's voice. Not empty.\n"
        "  Example: \"Now this is more my speed — lovely game!\" (Karen)\n"
        "- likes: an array of 2-4 short specific things this persona enjoyed.\n"
        '  Example: ["the bonus design", "win sizes lined up with my taste"].\n'
        "  If genuinely nothing was likeable, use [\"nothing in particular\"] —\n"
        "  but find SOMETHING; you're playing a real person, not a critic.\n"
        "- dislikes: an array of 2-4 short specific things this persona\n"
        "  disliked. If genuinely nothing was bad, [\"no real complaints\"].\n"
        '  Example: ["over-celebration of small wins", "rule clarity"].\n'
        "- review: 2-4 sentences in the persona's voice giving the full\n"
        "  reaction. Not a summary — a REACTION. Reference specific moments\n"
        "  from the video, not just scores. NEVER an empty string.\n"
        "- would_replay: one of \"yes\" | \"no\" | \"maybe\". Pick decisively.\n"
        "- would_replay_reason: one short sentence justifying the replay\n"
        "  answer in the persona's voice.\n\n"
        "If you produce empty strings, [] arrays, or null for ANY of these,\n"
        "the evaluation is unusable. Always write actual content."
    )


# --------------------------------------------------------------------------- #
# OpenRouter backend
# --------------------------------------------------------------------------- #


# OpenRouter is an OpenAI-compatible gateway proxying many providers. One API
# key, many models. This app sends frames as image_url parts, so the curated
# list is VISION-ONLY — text-only models would fail the moment images arrive.
# Users can still type any provider/model id manually if they want something
# outside this list.
# Curated fallback. The UI populates the dropdown from OpenRouter's live
# catalog (/api/openrouter/models) on load; this list is only used if that
# fetch fails. All IDs below are verified against the live catalog.
OPENROUTER_MODELS = [
    "google/gemini-2.5-flash",                  # cheap, fast, multimodal — best default for this app
    "google/gemini-2.5-flash-lite",             # cheaper, still multimodal
    "anthropic/claude-haiku-4.5",               # reliable, multimodal (dots in version, not dashes)
    "openai/gpt-4o-mini",                       # OpenAI's cheap vision model
    "meta-llama/llama-3.2-11b-vision-instruct", # Llama 3.2 Vision (11B is what OpenRouter actually hosts)
    "qwen/qwen2.5-vl-72b-instruct",             # Qwen 2.5 VL 72B (strong open-weights VLM)
    "mistralai/pixtral-large-2411",             # Pixtral Large (Mistral's flagship VLM)
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional headers OpenRouter uses for analytics/leaderboard ranking. Harmless
# and not auth-related — they let your usage show up under a friendly name on
# the OpenRouter dashboard.
OPENROUTER_REFERER = "https://github.com/local/ai-personas-game-test"
OPENROUTER_TITLE = "Vox"


def _build_openrouter_chain(start: str | None) -> list[str]:
    chain: list[str] = []
    if start:
        chain.append(start)
    for m in OPENROUTER_MODELS:
        if m not in chain:
            chain.append(m)
    return chain or list(OPENROUTER_MODELS)


class OpenRouterBackend(LLMBackend):
    """OpenAI-compatible vision backend with retry + model fallback.

    OpenRouter proxies many providers behind a single OpenAI-shape API and a
    single key. Frames (JPEG bytes) are sent as base64-encoded `image_url`
    content parts; same path handles both stats extraction and persona
    evaluation.

    JSON output: uses `response_format={"type":"json_object"}` when the model
    supports it. Reasoner-style models (DeepSeek R1, etc.) and any model in
    NO_JSON_MODE_MODELS opt out — we fall back to extracting the first JSON
    object from the response text via _extract_json_object.
    """

    name = "openrouter"

    RETRY_STATUS = {429, 500, 502, 503, 504}
    # Connection-layer hiccups (SSL EOF, reset, broken pipe, DNS blips) are
    # transient and worth retrying — they used to crash the whole call. The
    # specific failure that prompted this list: `ReadError: [SSL:
    # UNEXPECTED_EOF_WHILE_READING] unexpected eof while reading`.
    RETRY_MESSAGE_HINTS = (
        "rate limit", "overloaded", "timeout", "temporarily",
        "ssl", "unexpected_eof", "unexpected eof", "connection reset",
        "broken pipe", "connection aborted", "remote disconnected",
        "incompleteread", "eof occurred", "bad gateway",
    )
    NOT_FOUND_HINTS = ("404", "not_found", "not found", "model_not_found", "no allowed providers")
    # Models that don't support response_format=json_object. Reasoning models
    # (R1, o1-style) typically reject it; we ask for JSON in the prompt and
    # parse it out of the response text instead. Currently empty because the
    # curated default list is all JSON-capable VLMs — but the mechanism stays
    # so users can plug in a reasoning model via the custom model field.
    NO_JSON_MODE_MODELS: set[str] = set()

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Either export it or pass it in the UI."
            )
        self._api_key = key

        start_model = model or os.environ.get("OPENROUTER_MODEL") or OPENROUTER_MODELS[0]
        self.models = _build_openrouter_chain(start_model)
        self._model_idx = 0
        self.requested_model = start_model
        self.fallback_log: list[dict[str, str]] = []

        self.max_retries = int(os.environ.get("OPENROUTER_MAX_RETRIES", "6"))
        self.initial_backoff = float(os.environ.get("OPENROUTER_INITIAL_BACKOFF", "2.0"))
        self.max_backoff = float(os.environ.get("OPENROUTER_MAX_BACKOFF", "30.0"))
        self.timeout_seconds = float(os.environ.get("OPENROUTER_TIMEOUT", "120"))
        # `detail` is an OpenAI-specific image_url hint. OpenRouter proxies it
        # through to upstream providers, some of which return 500s when they
        # see unrecognized fields. Default OFF — set OPENROUTER_IMAGE_DETAIL=low
        # to opt back in if you're routing exclusively to OpenAI.
        self.image_detail = os.environ.get("OPENROUTER_IMAGE_DETAIL", "")
        # Generous output cap: a persona evaluation includes 15 scores + a
        # full review + likes/dislikes/quote, which can be 1500+ tokens in
        # JSON. Without an explicit budget some providers truncate around
        # 1024–2048 and we lose the trailing narrative fields. Bump if you
        # see review/likes still getting cut.
        self.max_tokens = int(os.environ.get("OPENROUTER_MAX_TOKENS", "2000"))
        # Guards fallback bookkeeping under parallel persona evaluation.
        self._fallback_lock = threading.Lock()
        # Monotonic per-instance call-id source; lets the UI update rows in place.
        self._call_id_source = _id_counter(1)

    @property
    def current_model(self) -> str:
        return self.models[self._model_idx]

    def _is_404(self, status: int | None, msg: str) -> bool:
        if status == 404:
            return True
        m = msg.lower()
        return any(h in m for h in self.NOT_FOUND_HINTS)

    def _is_retriable(self, status: int | None, msg: str) -> bool:
        if status in self.RETRY_STATUS:
            return True
        m = msg.lower()
        # Lowercase hints so a hint like "Bad Gateway" still matches.
        return any(h.lower() in m for h in self.RETRY_MESSAGE_HINTS)

    def _advance_model(self, reason: str) -> bool:
        with self._fallback_lock:
            if self._model_idx + 1 >= len(self.models):
                return False
            old = self.current_model
            self._model_idx += 1
            new = self.current_model
            self.fallback_log.append({"from": old, "to": new, "reason": reason})
            print(f"[openrouter] falling back: {old} → {new} ({reason})", flush=True)
            return True

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Single POST to /chat/completions. Raises with (status, message)
        embedded in the exception so the retry layer can categorize it."""
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                # Optional OpenRouter ranking/analytics headers — harmless,
                # surface this app on the OpenRouter dashboard.
                "HTTP-Referer": OPENROUTER_REFERER,
                "X-Title": OPENROUTER_TITLE,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass
            # OpenRouter wraps upstream provider errors as
            # {"error": {"message": "...", "code": ..., "metadata": {"provider_name": "...", "raw": "..."}}}
            # Surface the message + provider so users see WHICH provider failed,
            # not just "Internal server error". Falls back to the raw body if
            # the envelope doesn't match.
            extracted = _extract_openrouter_error(err_body)
            raise RuntimeError(f"HTTP {e.code}: {extracted or err_body or e.reason}") from e
        except TimeoutError as e:
            # Explicit catch so the surfaced message contains "timeout" — the
            # retry layer's hint list looks for that substring. Without this
            # the message can be just "_ssl.c:..." style noise that doesn't
            # match the hints.
            raise RuntimeError(f"Connection timed out after {self.timeout_seconds:.0f}s: {e}") from e
        except urllib.error.URLError as e:
            # socket-level errors (DNS failure, connection refused, SSL issues
            # mid-read) often arrive wrapped in URLError. Include the type so
            # the retry layer can pattern-match it.
            raise RuntimeError(f"URLError: {type(e.reason).__name__ if hasattr(e, 'reason') else 'unknown'}: {e.reason if hasattr(e, 'reason') else e}") from e

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"OpenRouter returned non-JSON envelope: {raw[:200]!r}") from e
        # OpenRouter sometimes returns HTTP 200 with the error embedded in the
        # body (e.g. when an upstream provider returns 5xx mid-stream). Detect
        # and raise so the retry/fallback layer can act on it.
        if isinstance(data, dict) and "error" in data and "choices" not in data:
            extracted = _extract_openrouter_error(raw)
            raise RuntimeError(f"OpenRouter error: {extracted or raw[:200]}")
        return data

    def _chat_completion(
        self,
        label: str,
        messages: list[dict[str, str]],
        want_json: bool = True,
    ) -> str:
        """Run chat completion with retry-on-transient + fallback-on-model-fatal.

        Returns the assistant's `content` string. Caller is responsible for
        parsing JSON out of it.

        Emits these events through `self._emit` for the live UI feed:
          {"type":"started", id, label, backend:"openrouter", model, attempt:1}
          {"type":"retrying", id, label, model, attempt, error}
          {"type":"succeeded", id, label, model, duration_ms, attempts}
          {"type":"failed", id, label, model, duration_ms, attempts, error}
        """
        call_id = next(self._call_id_source)
        call_started = time.time()
        total_attempts = 0
        last_err: Exception | None = None

        self._emit({
            "type": "started", "id": call_id, "label": label,
            "backend": "openrouter", "model": self.current_model, "attempt": 1,
        })

        while True:
            model = self.current_model
            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "stream": False,
                "max_tokens": self.max_tokens,
            }
            if want_json and model not in self.NO_JSON_MODE_MODELS:
                payload["response_format"] = {"type": "json_object"}

            delay = self.initial_backoff
            attempt = 0
            should_fallback = False
            fallback_reason = ""
            while attempt < self.max_retries:
                attempt += 1
                total_attempts += 1
                try:
                    data = self._post_chat(payload)
                    choices = data.get("choices") or []
                    if not choices:
                        raise RuntimeError(f"OpenRouter returned no choices: {data!r}")
                    content = choices[0].get("message", {}).get("content")
                    if not isinstance(content, str):
                        raise RuntimeError(f"OpenRouter message.content not a string: {choices[0]!r}")
                    self._emit({
                        "type": "succeeded", "id": call_id, "label": label,
                        "model": model, "attempts": total_attempts,
                        "duration_ms": round((time.time() - call_started) * 1000),
                    })
                    return content
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    # Pull HTTP status out of "HTTP <code>:" prefix if present
                    status: int | None = None
                    if msg.startswith("HTTP "):
                        try:
                            status = int(msg.split(" ", 1)[1].split(":", 1)[0])
                        except (ValueError, IndexError):
                            status = None

                    if self._is_404(status, msg):
                        should_fallback = True
                        fallback_reason = f"404 on {model}"
                        print(f"[openrouter] {label}: {model} → 404. Falling back.", flush=True)
                        break
                    if self._is_retriable(status, msg):
                        if attempt < self.max_retries:
                            wait = delay + random.uniform(0, delay * 0.25)
                            print(
                                f"[openrouter] {label} on {model} attempt {attempt}/{self.max_retries}: "
                                f"{msg[:120]}. Retry in {wait:.1f}s…",
                                flush=True,
                            )
                            self._emit({
                                "type": "retrying", "id": call_id, "label": label,
                                "model": model, "attempt": total_attempts,
                                "error": msg[:200], "wait_ms": round(wait * 1000),
                            })
                            time.sleep(wait)
                            delay = min(delay * 2, self.max_backoff)
                            continue
                        should_fallback = True
                        fallback_reason = f"exhausted retries on {model}"
                        print(f"[openrouter] {label}: {fallback_reason}. Falling back.", flush=True)
                        break
                    # Non-retriable, non-404 → real error
                    self._emit({
                        "type": "failed", "id": call_id, "label": label,
                        "model": model, "attempts": total_attempts,
                        "duration_ms": round((time.time() - call_started) * 1000),
                        "error": msg[:400],
                    })
                    raise
            if not should_fallback:
                if last_err:
                    raise last_err
                raise RuntimeError("unreachable")
            if not self._advance_model(fallback_reason):
                self._emit({
                    "type": "failed", "id": call_id, "label": label,
                    "model": model, "attempts": total_attempts,
                    "duration_ms": round((time.time() - call_started) * 1000),
                    "error": f"All models exhausted: {self.models}. Last: {str(last_err)[:200]}",
                })
                if last_err:
                    raise last_err
                raise RuntimeError(f"All OpenRouter models exhausted: {self.models}")
            # Loop continues on the next model — emit a retrying event so the
            # UI knows the row is still alive on a different model.
            self._emit({
                "type": "retrying", "id": call_id, "label": label,
                "model": self.current_model, "attempt": total_attempts,
                "error": f"fallback: {fallback_reason}",
            })

    def _user_content_with_frames(self, frames: list[bytes], text: str) -> list[dict[str, Any]]:
        """Build OpenAI-vision-style content list: frames first (so the model
        sees them in order before reading the instructions), then the prompt."""
        parts: list[dict[str, Any]] = []
        for frame in frames:
            b64 = base64.b64encode(frame).decode("ascii")
            image_url: dict[str, Any] = {"url": f"data:image/jpeg;base64,{b64}"}
            # `detail` is OpenAI-specific; some OpenRouter-proxied providers
            # have returned 500s on unrecognized image_url fields. Only attach
            # it when the user explicitly configures one via env.
            if self.image_detail:
                image_url["detail"] = self.image_detail
            parts.append({"type": "image_url", "image_url": image_url})
        parts.append({"type": "text", "text": text})
        return parts

    def extract_stats(
        self,
        frames: list[bytes],
        video_path: str | None = None,
    ) -> dict[str, Any]:
        del video_path  # frames are authoritative
        if not frames:
            raise RuntimeError("OpenRouter backend requires frames for stats extraction.")
        messages = [
            {"role": "user", "content": self._user_content_with_frames(frames, STATS_EXTRACTION_PROMPT)},
        ]
        content = self._chat_completion("extract_stats", messages, want_json=True)
        data = _extract_json_object(content)
        if data is None:
            raise RuntimeError(f"OpenRouter stats extraction returned non-JSON: {content[:300]!r}")
        return data

    def evaluate(self, req: EvaluationRequest) -> dict[str, Any]:
        user_text = build_user_prompt(req.persona, req.stats)
        label = f"evaluate[{req.persona.get('id','?')}#{req.run_index}]"
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": self._user_content_with_frames(req.frames, user_text)},
        ]
        content = self._chat_completion(label, messages, want_json=True)
        data = _extract_json_object(content)
        if data is None:
            raise RuntimeError(f"OpenRouter returned non-JSON content: {content[:300]!r}")
        return _normalize_narrative(_normalize_scores(data))


def _extract_openrouter_error(body: str) -> str | None:
    """Pull a human-readable message out of an OpenRouter error response body.

    OpenRouter normalizes upstream failures into:
        {"error": {"message": "...", "code": ..., "metadata": {"provider_name": "...", "raw": "..."}}}
    Returns a one-line summary like
        "Provider returned error [provider: OpenAI] raw: ..." or None if the
    body doesn't parse / doesn't have the expected shape.
    """
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    err = data.get("error")
    if not isinstance(err, dict):
        return None
    msg = str(err.get("message", "")).strip()
    parts = [msg] if msg else []
    meta = err.get("metadata") or {}
    if isinstance(meta, dict):
        provider = meta.get("provider_name")
        if provider:
            parts.append(f"[provider: {provider}]")
        raw = meta.get("raw")
        if raw:
            raw_str = str(raw)
            parts.append(f"raw: {raw_str[:300]}")
    code = err.get("code")
    if code is not None and not msg:
        parts.append(f"code: {code}")
    return " ".join(parts) if parts else None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON extraction. With response_format=json_object the whole
    string is JSON; with deepseek-reasoner we may get reasoning + a JSON block,
    so we fall back to finding the first {...} balanced object."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences if present
    if text.startswith("```"):
        fenced = text.strip("`")
        if fenced.startswith("json"):
            fenced = fenced[4:]
        try:
            return json.loads(fenced.strip())
        except json.JSONDecodeError:
            pass
    # Balanced-brace scan for the first JSON object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    break

    # Truncation recovery: the response was cut off mid-string or mid-array.
    # Walk the original text again, tracking the open-bracket stack, and at
    # each safe boundary (after a `,` or `}` at the OUTERMOST level) try to
    # close the structure and parse. This salvages dimension_scores even
    # when `review`/`likes` got truncated by max_tokens.
    salvaged = _try_close_truncated(text[start:])
    if salvaged is not None:
        return salvaged
    return None


def _try_close_truncated(text: str) -> dict[str, Any] | None:
    """Heuristic recovery from JSON that was cut off mid-stream.

    Walks the text, tracking bracket/quote state. Whenever we're outside a
    string AND at depth 1 (i.e. between fields of the outermost object),
    record that position as a safe truncation boundary. After the walk, try
    to parse the longest prefix ending at a safe boundary, padded with the
    needed closing braces.

    Real-world example this fixes: a 4000-token response that ends mid-review
    like `..."review": "Karen rated this game with`. We trim back to before
    the unfinished review field and parse what we have.
    """
    if not text or text[0] != "{":
        return None
    stack: list[str] = []
    in_string = False
    esc = False
    safe_boundaries: list[int] = []  # indices after a comma at outermost level

    for i, c in enumerate(text):
        if in_string:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
            continue
        if c in "[{":
            stack.append(c)
        elif c == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif c == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif c == "," and len(stack) == 1:
            # We're between top-level fields — a safe truncation point.
            safe_boundaries.append(i)

    # Try the longest safe-prefix that re-balances.
    for cut in reversed(safe_boundaries):
        candidate = text[:cut]  # drops the trailing comma
        # Count remaining open structures to add closing braces.
        rebuilt = _rebalance(candidate)
        if rebuilt is None:
            continue
        try:
            return json.loads(rebuilt)
        except json.JSONDecodeError:
            continue
    return None


def _rebalance(text: str) -> str | None:
    """Append closing brackets/braces to balance `text`. Returns None if the
    text ends inside a string (we can't know where the string was supposed to
    close)."""
    stack: list[str] = []
    in_string = False
    esc = False
    for c in text:
        if in_string:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c in "[{":
            stack.append(c)
        elif c == "]" and stack and stack[-1] == "[":
            stack.pop()
        elif c == "}" and stack and stack[-1] == "{":
            stack.pop()
    if in_string:
        return None
    # Strip a trailing comma; JSON doesn't allow it before the close.
    text = text.rstrip(" \t\n\r,")
    closing = "".join("]" if c == "[" else "}" for c in reversed(stack))
    return text + closing


# Score wrapper keys we look for when a model nests an integer inside an
# object instead of returning it flat. Order matters — the most likely keys
# (verbatim "score") are checked first.
_SCORE_KEYS = ("score", "value", "rating", "Score", "Value", "Rating", "n", "number")
# Stat-value wrapper keys. Same idea, for strings.
_STAT_VALUE_KEYS = ("value", "category", "label", "result", "answer", "Value", "Category")


def _coerce_score(v: Any) -> int | None:
    """Extract an integer score from int / float / numeric str / dict.

    Some models (notably Gemini in `json_object` mode) wrap each dimension
    score in an explanatory object like {"score": 7, "reason": "..."} instead
    of returning a flat integer. Without this coercion every score would
    collapse to the fallback 4 because `int(float({"score": 7}))` raises.

    Returns None if nothing salvageable was found; the caller picks a default.
    """
    if isinstance(v, bool):
        # bool is an int subclass in Python — explicitly reject so True doesn't
        # become 1 and silently pollute every score.
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(round(v))
    if isinstance(v, str):
        try:
            return int(round(float(v.strip())))
        except (ValueError, TypeError):
            return None
    if isinstance(v, dict):
        for key in _SCORE_KEYS:
            if key in v:
                inner = _coerce_score(v[key])
                if inner is not None:
                    return inner
        # Last-resort: any numeric value buried in the dict.
        for inner in v.values():
            n = _coerce_score(inner)
            if n is not None:
                return n
        return None
    return None


def _coerce_stat_value(v: Any) -> Any:
    """Flatten a single stats field. Strings/numbers pass through; nested
    objects unwrap to the inner value when a recognized wrapper key is found.

    This fixes the `[object Object]` rendering bug — when a model returns
    `{"base_pacing": {"value": "slow", "reasoning": "..."}}` we want
    `"slow"` in the output, not the whole object.
    """
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        # For list-valued stats (e.g. `themes`), recursively flatten entries.
        return [_coerce_stat_value(x) for x in v]
    if isinstance(v, dict):
        for key in _STAT_VALUE_KEYS:
            if key in v:
                inner = v[key]
                # Only unwrap if the inner is a primitive — don't accidentally
                # unwrap into another dict that loses the original structure.
                if isinstance(inner, (str, int, float, bool)) or inner is None:
                    return inner
                # Inner is itself a dict — recurse once more.
                if isinstance(inner, dict):
                    return _coerce_stat_value(inner)
        # No recognized wrapper — leave as-is so the caller can decide. Some
        # legitimate stats are dict-valued (e.g. `dimension_scores`).
        return v
    return v


def _normalize_stats(stats: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Flatten any per-field wrapping. Returns (cleaned_stats, list_of_fields_that_needed_flattening).

    The returned warning list lets the caller surface "model didn't return
    flat values" in the UI feed so the user knows the result was salvaged
    rather than coming back clean.
    """
    if not isinstance(stats, dict):
        return ({}, ["stats was not a dict — discarded"])
    # If the model wrapped everything in a top-level "stats" / "result" key,
    # unwrap once.
    if len(stats) == 1:
        only_key = next(iter(stats))
        if only_key.lower() in ("stats", "result", "output", "extracted", "data"):
            inner = stats[only_key]
            if isinstance(inner, dict):
                stats = inner

    warnings: list[str] = []
    cleaned: dict[str, Any] = {}
    for k, v in stats.items():
        new_v = _coerce_stat_value(v)
        if isinstance(v, dict) and not isinstance(new_v, dict):
            warnings.append(k)
        cleaned[k] = new_v
    return cleaned, warnings


def _normalize_scores(data: dict[str, Any]) -> dict[str, Any]:
    """Defensive: clamp scores into 1..10, bump 5/6 to nearest forbidden-safe
    value, and unwrap object-wrapped scores via `_coerce_score`. Sets
    `data["_score_coercions"]` to a list of dimension names that needed
    unwrapping so the caller can warn the user when a model misbehaves."""
    ds = data.get("dimension_scores") or {}
    if not isinstance(ds, dict):
        ds = {}
    # Backwards-compat aliases: older runs / older LLM responses may use the
    # pre-refactor dimension names. Map them in before normalization so we
    # don't silently drop scores on those.
    aliases = {"bonus": "bonus_frequency", "pacing": "base_pacing"}
    for old, new in aliases.items():
        if old in ds and new not in ds:
            ds[new] = ds[old]
    fixed: dict[str, int] = {}
    coercions: list[str] = []
    missing: list[str] = []
    for d in DIMENSIONS:
        raw = ds.get(d)
        v = _coerce_score(raw)
        if v is None:
            if raw is None:
                missing.append(d)
            else:
                # Something was there but unparseable — record as coerced.
                coercions.append(d)
            v = 4
        elif isinstance(raw, dict):
            coercions.append(d)
        v = max(1, min(10, v))
        if v == 5:
            v = 4
        elif v == 6:
            v = 7
        fixed[d] = v
    data["dimension_scores"] = fixed
    if coercions:
        data["_score_coercions"] = coercions
    if missing and len(missing) == len(DIMENSIONS):
        # Nothing usable came back — flag the whole response.
        data["_score_missing"] = True
    return data


# Common key aliases models use instead of our canonical narrative keys.
# Listed canonical-first then alternates; we pick the first non-empty match.
_QUOTE_ALIASES = ("quote", "comment", "caption", "oneliner", "one_liner", "tagline", "headline", "first_impression")
_LIKES_ALIASES = ("likes", "pros", "positives", "liked", "highlights", "loves", "good", "what_i_liked")
_DISLIKES_ALIASES = ("dislikes", "cons", "negatives", "disliked", "complaints", "hates", "bad", "what_i_didnt_like", "what_i_disliked")
_REVIEW_ALIASES = ("review", "commentary", "reaction", "summary", "analysis", "full_review", "full_reaction", "verdict")
_REPLAY_REASON_ALIASES = ("would_replay_reason", "replay_reason", "reason", "replay_explanation", "replay_rationale")
_REPLAY_ALIASES = ("would_replay", "replay", "play_again", "would_play_again")


def _string_from_any(v: Any) -> str | None:
    """Coerce a value to a string. Unwraps dicts with text/value/content keys.
    Returns None if the result is empty or nothing salvageable."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, dict):
        for k in ("text", "value", "content", "string", "comment", "review"):
            if k in v:
                inner = _string_from_any(v[k])
                if inner:
                    return inner
        return None
    if isinstance(v, list):
        # Some models return a list of strings for a single-string field —
        # join them.
        joined = " ".join(_string_from_any(x) or "" for x in v).strip()
        return joined if joined else None
    return None


def _list_from_any(v: Any) -> list[str]:
    """Coerce a value to a list of non-empty strings."""
    if v is None:
        return []
    if isinstance(v, list):
        out: list[str] = []
        for x in v:
            s = _string_from_any(x)
            if s:
                out.append(s)
        return out
    if isinstance(v, str):
        # Comma/semicolon-split a single-string answer.
        parts = [p.strip() for p in v.replace(";", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(v, dict):
        # Some models return {"item1": "...", "item2": "..."}; take values.
        return [s for s in (_string_from_any(x) for x in v.values()) if s]
    return []


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first non-empty value from `data` matching any of `keys`."""
    for k in keys:
        if k in data:
            v = data[k]
            if v is not None and v != "" and v != []:
                return v
    return None


def _normalize_narrative(data: dict[str, Any]) -> dict[str, Any]:
    """Fill canonical narrative keys (quote/likes/dislikes/review/would_replay
    /would_replay_reason) from common aliases, unwrap dict-wrapped strings,
    and normalize list/string types. Records which fields were synthesized in
    `data["_narrative_coercions"]` so the UI feed can warn the user.

    Does NOT invent content — if the model genuinely returned nothing for a
    field, we leave it empty for `evaluate_persona`'s fallback to handle.
    """
    if not isinstance(data, dict):
        return data
    coercions: list[str] = []

    # quote
    if not data.get("quote") or isinstance(data.get("quote"), (dict, list)):
        original_present = "quote" in data and data["quote"] not in (None, "")
        salvaged = _string_from_any(_first_present(data, _QUOTE_ALIASES))
        if salvaged:
            data["quote"] = salvaged
            if original_present is False or "quote" not in data:
                coercions.append("quote")

    # likes / dislikes
    if not isinstance(data.get("likes"), list) or not data["likes"]:
        salvaged_list = _list_from_any(_first_present(data, _LIKES_ALIASES))
        if salvaged_list:
            data["likes"] = salvaged_list
            coercions.append("likes")
    else:
        data["likes"] = _list_from_any(data["likes"])

    if not isinstance(data.get("dislikes"), list) or not data["dislikes"]:
        salvaged_list = _list_from_any(_first_present(data, _DISLIKES_ALIASES))
        if salvaged_list:
            data["dislikes"] = salvaged_list
            coercions.append("dislikes")
    else:
        data["dislikes"] = _list_from_any(data["dislikes"])

    # review
    if not data.get("review") or isinstance(data.get("review"), (dict, list)):
        salvaged = _string_from_any(_first_present(data, _REVIEW_ALIASES))
        if salvaged:
            data["review"] = salvaged
            coercions.append("review")

    # would_replay_reason
    if not data.get("would_replay_reason") or isinstance(data.get("would_replay_reason"), (dict, list)):
        salvaged = _string_from_any(_first_present(data, _REPLAY_REASON_ALIASES))
        if salvaged:
            data["would_replay_reason"] = salvaged
            coercions.append("would_replay_reason")

    # would_replay — must be in the {yes,no,maybe} enum
    wr = data.get("would_replay")
    if not isinstance(wr, str) or wr.lower() not in ("yes", "no", "maybe"):
        salvaged = _string_from_any(_first_present(data, _REPLAY_ALIASES))
        if isinstance(salvaged, str):
            low = salvaged.lower().strip(".!?, ")
            if low in ("yes", "no", "maybe", "y", "n"):
                data["would_replay"] = {"y": "yes", "n": "no"}.get(low, low)
                coercions.append("would_replay")
            elif "yes" in low and "no" not in low:
                data["would_replay"] = "yes"
                coercions.append("would_replay")
            elif "no" in low and "yes" not in low:
                data["would_replay"] = "no"
                coercions.append("would_replay")

    if coercions:
        data["_narrative_coercions"] = coercions
    return data


def get_backend(
    name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> LLMBackend:
    name = (name or "openrouter").lower()
    if name == "openrouter":
        return OpenRouterBackend(api_key=api_key, model=model)
    raise ValueError(f"Unknown LLM backend: {name!r}")
