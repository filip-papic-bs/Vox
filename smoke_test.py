"""End-to-end smoke test for the mock backend.

Verifies:
  1. All personas produce a valid result (no exceptions, schema-conforming).
  2. Ratings span at least 5 points across personas (i.e. they actually disagree).
  3. Per-persona run variance stays ≤ 1.5 (the brief expects ±0.5 to ±1).
  4. No dimension score equals 5 or 6 (forbidden by the spec).
  5. At least one persona scores ≥ 8 AND at least one scores ≤ 4 on the same
     game (rejects the "everything ends up mid" failure mode).
"""

from __future__ import annotations

import sys

from evaluator import evaluate_all, load_personas
from llm_backend import MockBackend

EXPECTED_PERSONAS = 18

# Cyberpunk-degen preset — should split the roster cleanly. Includes the new
# quality-of-experience fields so the test covers the post-feedback dimensions.
SAMPLE_STATS = {
    "rounds_played": 200,
    "win_rate": 0.22,
    "bonus_rate": 0.04,
    "avg_win_size_multiplier": 8.0,
    "biggest_win_multiplier": 2500,
    "base_pacing": "fast",
    "anticipation_level": "high",
    "session_arc": "peaks_and_valleys",
    "near_miss_frequency": "high",
    "dry_streak_feel": "high",
    "visual_brightness": "high",
    "theme": "cyberpunk",
    "themes": ["cyberpunk", "neon", "edgy", "futuristic", "money"],
    "volatility": "high",
    "mechanic_complexity": "complex",
    "feature_variety": "many",
    "bonus_quality": "iconic",
    # Quality-of-experience axes: this preset is "loud and a bit dishonest" —
    # over-celebrates small wins, unusual reel stops, low readability.
    # Should make Diana / Sofia / Marco unhappy on those axes specifically.
    "small_win_celebration_intensity": "high",
    "win_tier_distinction": "weak",
    "anticipation_honesty": "low",
    "reel_stop_order": "unusual",
    "visual_readability": "low",
    "bonus_rule_clarity": "mixed",
    "win_display_format": "multiplier",
}


def main() -> int:
    personas = load_personas("personas.json")
    # Explicit stats path: bypass extract_stats so this test stays deterministic
    # and exercises the persona-scoring logic specifically.
    result = evaluate_all(MockBackend(), personas, video_path=None, stats=SAMPLE_STATS)
    results = result["personas"]

    print(f"Ran {len(results)} personas in {result['elapsed_seconds']}s\n")
    print(f"{'persona':<14}{'rating':>8}{'mood':>16}{'replay':>10}   quote")
    print("-" * 100)
    failed = []
    for r in sorted(results, key=lambda x: -x["rating"]):
        print(f"{r['persona_name']:<14}{r['rating']:>8.2f}{r['mood']:>16}{r['would_replay']:>10}   {r['quote'][:60]}")
        # Forbidden scores
        for d, v in r["dimension_scores"].items():
            if v in (5, 6):
                failed.append(f"{r['persona_id']} {d} = {v}")
        # Variance bound
        if r["rating_variance"] > 1.5:
            failed.append(f"{r['persona_id']} variance {r['rating_variance']} > 1.5")

    ratings = [r["rating"] for r in results]
    spread = max(ratings) - min(ratings)
    print(f"\nRating spread across personas: {spread:.2f}")

    ok = True
    if len(results) != EXPECTED_PERSONAS:
        print(f"FAIL: expected {EXPECTED_PERSONAS} personas, got {len(results)}")
        ok = False
    if spread < 5.0:
        print(f"FAIL: rating spread {spread:.2f} < 5.0 — personas not differentiated enough")
        ok = False
    if max(ratings) < 8.0:
        print(f"FAIL: top rating {max(ratings):.2f} < 8.0 — nobody loved it (mid-syndrome)")
        ok = False
    if min(ratings) > 4.0:
        print(f"FAIL: bottom rating {min(ratings):.2f} > 4.0 — nobody hated it (mid-syndrome)")
        ok = False
    if failed:
        print("FAIL:")
        for f in failed:
            print(f"  - {f}")
        ok = False

    if ok:
        print("\nOK — smoke test passed.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
