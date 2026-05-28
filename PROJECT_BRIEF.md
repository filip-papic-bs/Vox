# Vox Machina — Project Brief

## What This Is

A system where a defined set of AI "players" (personas), each with distinct personality traits, watch a game video and produce honest, in-character feedback. The user uploads a video of a game; the system produces a dashboard showing every persona's reaction.

The goal is **consistent, differentiated, persona-faithful feedback** — the same video given to the same persona should produce nearly the same feedback every time, and personas with different preferences should produce visibly different opinions.

## How It Works (End-to-End Flow)

1. **Personas are pre-defined.** A fixed roster of 10 (or so) players, each with a name, avatar/icon, age, archetype, and a profile of preferences (visual taste, win-frequency preference, bonus appetite, pacing tolerance, theme likes/dislikes, betting style, etc.). Defined once in a config file.
2. **The user uploads a video** of a game session (plus, optionally, a structured stats summary: rounds played, win/loss counts, bonus triggers, average win size, biggest win, etc.).
3. **The system runs each persona against the video.** One LLM call per persona, producing structured feedback (rating, dimension scores, likes, dislikes, would-replay, in-character quote).
4. **The dashboard renders the results** as a grid of persona cards — name, icon, headline rating, short summary. Clicking a card opens the full detailed feedback for that persona.

## Architecture

```
┌──────────────────────┐
│  Persona Registry    │  Config file — 10 personas with traits, icons, priorities
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  User Input          │  Video upload + optional stats summary
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Evaluator           │  For each persona:
│                      │   - Persona system prompt
│                      │   - Video + stats as input
│                      │   - temperature=0, pinned model version
│                      │   - Strict JSON schema output
│                      │   - 5 runs → median rating
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  Dashboard (web UI)  │  Grid of persona cards
│                      │   - Click card → full feedback view
│                      │   - Export to CSV/JSON if needed
└──────────────────────┘
```

## Key Architectural Principles

1. **The LLM is the critic, not the statistician.** Objective game stats (win frequency, bonus rate, etc.) come from structured data provided alongside the video — NOT from the LLM counting events in video. The LLM's job is to react to those facts in character.
2. **Vendor-swappable LLM layer.** Build an `LLMBackend` abstraction so the underlying model can be swapped (Gemini Flash ↔ Claude Haiku ↔ local Qwen2.5-VL via Ollama) without touching the rest of the system.
3. **The orchestrator is the product.** The LLM is a component. Personas, prompts, structured outputs, dashboard rendering — that's the actual codebase.

## Tech Stack

| Layer | Choice | Reasoning |
|---|---|---|
| Language | Python | Standard for AI/automation |
| LLM API (primary) | **Gemini Flash** | Cheapest multimodal, native video input |
| LLM API (alt) | Claude Haiku, or local Qwen2.5-VL via Ollama | Backup options |
| Frame extraction (if needed) | ffmpeg / OpenCV | If sampling frames instead of native video input |
| Dashboard | Simple web app (HTML + JS, or Next.js / Streamlit / Gradio) | Whatever ships fastest |
| Storage | SQLite or JSON files | No infra needed |

**Budget target:** €100/month total API spend. Expected actual cost: ~€10–30/month.

## Reproducibility (same video → same feedback)

LLMs are stochastic by default; the following gets it close to deterministic:
- `temperature = 0`
- **Pin to a specific model snapshot version**, not just a model family (model behavior drifts otherwise)
- Strict JSON schema output (constrains the output space)
- Run each persona evaluation **5 times, take median** for headline numbers
- Track inter-run variance per persona and report it in the dashboard

Realistic expectation: ratings vary by ±0.5 to ±1 on a 10-point scale across runs. Tight enough to be useful.

## Validity (real judgment, not training-data recall)

LLMs have seen tons of game reviews. To prevent them from pattern-matching popular game names instead of evaluating features:

- **Anonymize the input** — crop/blur game logos, never mention the game's name in the prompt
- **Feature-anchored prompts** — personas rate specific dimensions (visual appeal, win frequency satisfaction, bonus excitement, pacing, theme fit) rather than holistic vibe
- **Final score is a weighted sum** of dimension ratings × persona priorities — mechanical and traceable
- **Force commitment** — forbid ratings of 5 or 6; personas must pick a side

## Persona Schema (proposed)

```yaml
- id: karen_001
  name: Karen
  age: 45
  archetype: "Cautious suburban casual"
  avatar: "assets/karen.png"
  bio: "Plays slots once a week to unwind. Hates losing streaks."
  preferences:
    visual_brightness: high
    win_frequency: high
    win_size: low                # prefers many small wins over rare big ones
    bonus_frequency: medium
    pacing: slow
    theme_likes: ["nature", "candy", "animals"]
    theme_dislikes: ["horror", "violence", "war"]
  betting_style: small_safe
  patience: low_for_losses
  voice: "warm but quick to complain, uses '!' a lot"
```

The `voice` field shapes how the in-character quote sounds — keeps personas feeling distinct in tone, not just in numerical ratings.

## Dashboard Spec

**Main view:** grid of persona cards (one per persona). Each card shows:
- Avatar / icon
- Name + age
- Headline rating (e.g., 7.5/10)
- One-line in-character reaction quote
- Color-coded mood indicator (loved it / liked it / meh / disliked it / hated it)

**Detail view (click a card):**
- Full persona profile reminder
- Per-dimension scores (visual, win frequency, bonus, pacing, theme — bar chart)
- "What I liked" / "What I didn't" lists
- Would I play again? (Yes / No / Maybe + reason)
- Full in-character review paragraph
- Run-to-run variance indicator (so you can see consistency)

**Optional:** comparison view that shows two uploaded videos side by side with how each persona reacted to both.

## Build Checklist

- [ ] Define 10 distinct personas in YAML/JSON, with avatars
- [ ] Define the JSON output schema for evaluations (ratings, dimension scores, quote, etc.)
- [ ] Build `LLMBackend` abstraction; first implementation = Gemini Flash
- [ ] Build the evaluator: takes (persona, video, stats) → runs 5 times → returns median + variance
- [ ] Build the dashboard: persona grid + detail view
- [ ] Smoke test: upload one video, verify all personas produce differentiated and run-consistent outputs

## Open Questions

1. Are stats (win count, bonus triggers, etc.) entered manually when uploading a video, or extracted automatically (OCR from frames)?
2. What's the right number of evaluation dimensions per persona? (Too few = bland; too many = noisy. Start with ~5.)
3. Dashboard tech: lightweight HTML/JS, Streamlit (fastest to prototype), or a real React frontend? (dont use react, simple js dashboard grid)
4. Where do persona avatars come from? Hand-picked stock images, AI-generated, or simple emoji/initials at first? (simple svg art or emojis)

## Notes

- This project's novel contribution is the **persona-driven evaluation system**. Everything else is plumbing.
- A "good" output is not necessarily one that agrees with popular taste — it's one that's *internally consistent* (same persona, same video, same opinion) and *externally differentiated* (different personas, same video, visibly different opinions).
