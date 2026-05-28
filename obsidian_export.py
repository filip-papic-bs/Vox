"""Export the persona evaluator's state to a folder of cross-linked markdown
notes that Obsidian can open as a vault. The point is graph view: the more
the notes wiki-link each other, the richer the web becomes.

Layout:
  vault/
    personas/      one note per persona, linking to weighted dimensions,
                   liked/disliked themes, and every evaluation they reacted to
    dimensions/    one note per scoring dimension, with backlinks to personas
    themes/        one note per theme keyword, with likers/dislikers + evals
    evaluations/   one note per uploaded video, linking the theme tags and
                   each persona's reaction
    README.md      instructions to open in Obsidian

The exporter is *tolerant of schema drift* — it reads whatever fields exist in
the persona dict and evaluation JSON and links them up. Old 5-dim evaluations
and new 12-dim ones coexist in the same vault without issue.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Filename / display helpers
# --------------------------------------------------------------------------- #


_BAD_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|]')


def _safe_filename(s: str) -> str:
    return _BAD_FILENAME_CHARS.sub("", str(s)).strip() or "untitled"


def _title_theme(t: Any) -> str:
    """Theme keywords like 'cyberpunk' / 'KAWAII' / 'lucky7' come from many
    sources; normalize them so '[[Cyberpunk]]' from one note links to '[[
    cyberpunk]]' from another. Title-case but keep readable."""
    s = str(t).strip()
    return s.title() if s else ""


def _dim_label(d: str) -> str:
    return d.replace("_", " ").title()


def _eval_filename(e: dict[str, Any]) -> str:
    ts = e.get("created_at") or 0
    date = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H-%M") if ts else "eval"
    video = e.get("video_filename") or ""
    stem = video.rsplit(".", 1)[0] if "." in video else video
    stem = _safe_filename(stem) or (e.get("id") or "untitled")[:8]
    return f"{date} — {stem}"


# --------------------------------------------------------------------------- #
# Exporter
# --------------------------------------------------------------------------- #


_MOOD_ORDER = ["loved_it", "liked_it", "meh", "disliked_it", "hated_it"]
_MOOD_HEADING = {
    "loved_it": "Loved it",
    "liked_it": "Liked it",
    "meh": "Meh",
    "disliked_it": "Disliked it",
    "hated_it": "Hated it",
}

# Stat keys we know about + how to format them in the evaluation note. New
# keys not in this list still appear via the catch-all loop below.
_STATS_STRING_KEYS = (
    "pacing", "base_pacing", "visual_brightness", "volatility",
    "mechanic_complexity", "feature_variety", "audio_intensity",
    "bonus_quality", "session_arc", "session_flow", "anticipation_level",
    "near_miss_frequency", "dry_streak_feel",
)
_STATS_PCT_KEYS = ("win_rate", "bonus_rate")
_STATS_NUMERIC_KEYS = ("rounds_played", "avg_win_size_multiplier", "biggest_win_multiplier")


def export_vault(
    vault_dir: Path,
    personas: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Wipe-and-rewrite the vault. Returns a small summary dict."""
    vault_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("personas", "dimensions", "themes", "evaluations"):
        d = vault_dir / sub
        d.mkdir(exist_ok=True)
        # Wipe stale notes so renames / removals propagate
        for f in d.glob("*.md"):
            f.unlink()

    # --- Discover dimensions across personas + evaluations ---
    dimension_keys: set[str] = set()
    for p in personas:
        dimension_keys.update((p.get("weights") or {}).keys())
    for e in evaluations:
        for r in e.get("personas", []) or []:
            dimension_keys.update((r.get("dimension_scores") or {}).keys())

    # --- Discover themes ---
    themes: set[str] = set()
    for p in personas:
        prefs = p.get("preferences") or {}
        for t in prefs.get("theme_likes") or []:
            themes.add(_title_theme(t))
        for t in prefs.get("theme_dislikes") or []:
            themes.add(_title_theme(t))
    for e in evaluations:
        stats = e.get("stats") or {}
        if stats.get("theme"):
            themes.add(_title_theme(stats["theme"]))
        for t in stats.get("themes") or []:
            themes.add(_title_theme(t))
    themes.discard("")

    # --- Backlink maps ---
    theme_to_likers: dict[str, list[str]] = {}
    theme_to_dislikers: dict[str, list[str]] = {}
    for p in personas:
        prefs = p.get("preferences") or {}
        for t in prefs.get("theme_likes") or []:
            theme_to_likers.setdefault(_title_theme(t), []).append(p["name"])
        for t in prefs.get("theme_dislikes") or []:
            theme_to_dislikers.setdefault(_title_theme(t), []).append(p["name"])

    theme_to_evals: dict[str, list[dict]] = {}
    for e in evaluations:
        stats = e.get("stats") or {}
        tags = set()
        if stats.get("theme"):
            tags.add(_title_theme(stats["theme"]))
        for t in stats.get("themes") or []:
            tags.add(_title_theme(t))
        for tt in tags:
            theme_to_evals.setdefault(tt, []).append(e)

    pid_to_reactions: dict[str, list[tuple[dict, dict]]] = {}
    for e in evaluations:
        for r in e.get("personas") or []:
            pid_to_reactions.setdefault(r.get("persona_id", ""), []).append((e, r))

    dim_to_personas: dict[str, list[tuple[dict, float]]] = {}
    for p in personas:
        for d, w in (p.get("weights") or {}).items():
            try:
                w = float(w)
            except (TypeError, ValueError):
                continue
            if w > 0:
                dim_to_personas.setdefault(d, []).append((p, w))
    for d in dim_to_personas:
        dim_to_personas[d].sort(key=lambda x: -x[1])

    # --- Write dimension notes ---
    for d in sorted(dimension_keys):
        label = _dim_label(d)
        lines = [
            "---",
            "type: dimension",
            f"key: {d}",
            "---",
            "",
            f"# {label}",
            "",
            f"Scoring dimension *(key: `{d}`)*. Personas rate every game on this and",
            "their personal weight here controls how much it moves their overall rating.",
            "",
            "## Personas who weight this",
            "",
        ]
        for p, w in dim_to_personas.get(d, []):
            lines.append(f"- [[{p['name']}]] — **{int(round(w * 100))}%**")
        if not dim_to_personas.get(d):
            lines.append("_(No persona currently weights this dimension.)_")
        (vault_dir / "dimensions" / f"{_safe_filename(label)}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # --- Write theme notes ---
    for theme in sorted(themes):
        if not theme:
            continue
        lines = ["---", "type: theme", "---", "", f"# {theme}", ""]
        likers = theme_to_likers.get(theme) or []
        dislikers = theme_to_dislikers.get(theme) or []
        if likers:
            lines += ["## Liked by", ""]
            lines += [f"- [[{n}]]" for n in likers]
            lines.append("")
        if dislikers:
            lines += ["## Disliked by", ""]
            lines += [f"- [[{n}]]" for n in dislikers]
            lines.append("")
        evs = theme_to_evals.get(theme) or []
        if evs:
            lines += ["## Featured in", ""]
            for e in evs:
                lines.append(f"- [[{_eval_filename(e)}]]")
            lines.append("")
        if not (likers or dislikers or evs):
            lines.append("_(No backlinks yet — this theme is mentioned but unused.)_")
        (vault_dir / "themes" / f"{_safe_filename(theme)}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # --- Write persona notes ---
    for p in personas:
        name = p["name"]
        prefs = p.get("preferences") or {}
        weights = p.get("weights") or {}
        lines = [
            "---",
            "type: persona",
            f"id: {p.get('id', '')}",
            f"age: {p.get('age', '')}",
            f"archetype: \"{p.get('archetype', '')}\"",
            "---",
            "",
            f"# {name} {p.get('avatar', '')}".rstrip(),
            "",
            f"**Age {p.get('age', '?')} · {p.get('archetype', '')}**",
            "",
            f"> {p.get('bio', '')}",
            "",
            "## Dimension priorities",
            "",
        ]
        sorted_weights = sorted(weights.items(), key=lambda x: -float(x[1] or 0))
        for d, w in sorted_weights:
            try:
                pct = int(round(float(w) * 100))
            except (TypeError, ValueError):
                continue
            if pct > 0:
                lines.append(f"- [[{_dim_label(d)}]] — **{pct}%**")

        # Gameplay preferences = everything in prefs that isn't a theme list
        gameplay = [(k, v) for k, v in prefs.items() if k not in ("theme_likes", "theme_dislikes")]
        if gameplay:
            lines += ["", "## Gameplay preferences", ""]
            for k, v in gameplay:
                lines.append(f"- **{k.replace('_', ' ')}**: `{v}`")
        for k in ("betting_style", "patience", "session_length"):
            if k in p:
                lines.append(f"- **{k.replace('_', ' ')}**: `{p[k]}`")

        likes = prefs.get("theme_likes") or []
        dislikes = prefs.get("theme_dislikes") or []
        if likes:
            lines += ["", "## Themes liked", ""]
            lines.append(" · ".join(f"[[{_title_theme(t)}]]" for t in likes))
        if dislikes:
            lines += ["", "## Themes disliked", ""]
            lines.append(" · ".join(f"[[{_title_theme(t)}]]" for t in dislikes))

        if p.get("voice"):
            lines += ["", "## Voice", "", f"> {p['voice']}"]

        rxns = pid_to_reactions.get(p.get("id", "")) or []
        if rxns:
            lines += ["", "## Reactions", ""]
            for e, r in sorted(rxns, key=lambda x: -(x[0].get("created_at") or 0)):
                mood = (r.get("mood") or "").replace("_", " ")
                try:
                    rating = float(r.get("rating") or 0)
                except (TypeError, ValueError):
                    rating = 0.0
                lines.append(f"- [[{_eval_filename(e)}]] — **{rating:.1f}/10** · {mood}")
                quote = r.get("quote") or ""
                if quote:
                    lines.append(f"  > {quote}")

        (vault_dir / "personas" / f"{_safe_filename(name)}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # --- Write evaluation notes ---
    for e in evaluations:
        fname = _eval_filename(e)
        stats = e.get("stats") or {}
        lines = ["---", "type: evaluation", f"id: {e.get('id', '')}",
                 f"backend: {e.get('backend', '')}"]
        if e.get("model_used"):
            lines.append(f"model: {e['model_used']}")
        lines += ["---", "", f"# {fname}", ""]
        if e.get("backend"):
            mp = f" · model `{e.get('model_used')}`" if e.get("model_used") else ""
            lines.append(f"*Backend `{e['backend']}`{mp}*")
            lines.append("")

        lines += ["## Extracted from the video", ""]
        if stats.get("theme"):
            lines.append(f"- **Theme**: [[{_title_theme(stats['theme'])}]]")
        ttags = stats.get("themes") or []
        if ttags:
            tagged = " · ".join(f"[[{_title_theme(t)}]]" for t in ttags)
            lines.append(f"- **Theme tags**: {tagged}")
        for k in _STATS_STRING_KEYS:
            if stats.get(k) is not None:
                lines.append(f"- **{k.replace('_', ' ')}**: `{stats[k]}`")
        for k in _STATS_PCT_KEYS:
            v = stats.get(k)
            if isinstance(v, (int, float)):
                lines.append(f"- **{k.replace('_', ' ')}**: {v * 100:.1f}%")
        for k in _STATS_NUMERIC_KEYS:
            v = stats.get(k)
            if v is not None:
                suffix = "×" if "multiplier" in k else ""
                lines.append(f"- **{k.replace('_', ' ')}**: {v}{suffix}")
        # Surface any unknown stat keys as well, so new schema fields aren't hidden
        known = {"theme", "themes"} | set(_STATS_STRING_KEYS) | set(_STATS_PCT_KEYS) | set(_STATS_NUMERIC_KEYS)
        for k, v in stats.items():
            if k in known or v is None:
                continue
            lines.append(f"- **{k.replace('_', ' ')}**: `{v}`")
        lines.append("")

        # Persona reactions grouped by mood
        groups: dict[str, list[dict]] = {}
        for r in e.get("personas") or []:
            groups.setdefault(r.get("mood") or "meh", []).append(r)
        lines += ["## Persona reactions", ""]
        for m in _MOOD_ORDER:
            rs = groups.get(m) or []
            if not rs:
                continue
            lines.append(f"### {_MOOD_HEADING[m]}")
            lines.append("")
            for r in sorted(rs, key=lambda x: -float(x.get("rating") or 0)):
                try:
                    rating = float(r.get("rating") or 0)
                except (TypeError, ValueError):
                    rating = 0.0
                lines.append(f"- [[{r.get('persona_name', '?')}]] — **{rating:.1f}/10**")
                quote = r.get("quote") or ""
                if quote:
                    lines.append(f"  > {quote}")
            lines.append("")

        (vault_dir / "evaluations" / f"{_safe_filename(fname)}.md").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    # --- Index / README ---
    readme = (
        "# Vox Machina — Obsidian Vault\n\n"
        "Auto-generated by the persona evaluator. Contents:\n\n"
        f"- **{len(personas)} personas** — each with their weights, theme likes/dislikes, "
        "and every evaluation they reacted to\n"
        f"- **{len(dimension_keys)} scoring dimensions** with backlinks to the personas who weight them\n"
        f"- **{len(themes)} themes** — taxonomy of likes/dislikes + which uploads featured them\n"
        f"- **{len(evaluations)} evaluations** — one note per upload, grouped by mood\n\n"
        "## See the web\n\n"
        "1. Install Obsidian: <https://obsidian.md>\n"
        "2. Obsidian → *Open folder as vault* → select **this folder**.\n"
        "3. Press **Ctrl/Cmd + G** to open the graph view.\n"
        "4. In the graph settings sidebar → *Groups*, color-code by frontmatter:\n"
        "   `type: persona`, `type: dimension`, `type: theme`, `type: evaluation`.\n"
        "5. Crank up *Force → link force* and drop *center force* a touch — the web spreads "
        "into clusters: dimensions in the middle, personas around them, themes as outer satellites, "
        "evaluations bridging everything they touched.\n\n"
        "## Refresh\n\n"
        "The vault re-renders every time the web app finishes a new evaluation. You can also "
        "POST to `/api/export-obsidian` to force a refresh.\n"
    )
    (vault_dir / "README.md").write_text(readme, encoding="utf-8")

    return {
        "vault_path": str(vault_dir.resolve()),
        "persona_count": len(personas),
        "evaluation_count": len(evaluations),
        "theme_count": len(themes),
        "dimension_count": len(dimension_keys),
    }


def export_from_storage(
    vault_dir: Path,
    eval_dir: Path,
    personas_path: Path,
) -> dict[str, Any]:
    """Convenience wrapper: read everything off disk and rebuild the vault."""
    import json

    with open(personas_path, encoding="utf-8") as f:
        personas = json.load(f).get("personas", [])

    evaluations = []
    for p in sorted(eval_dir.glob("*.json"), reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                evaluations.append(json.load(f))
        except (OSError, ValueError):
            continue
    return export_vault(vault_dir, personas, evaluations)
