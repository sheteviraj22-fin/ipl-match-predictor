"""
engine.py
=========
Public API facade for the IPL prediction backend.

Exposes exactly the six required backend functions:

    compute_player_stats(data)
    compute_player_ratings(player_stats)
    compute_team_strength(squad, player_ratings)
    predict_match(teamA, teamB, context)
    simulate_season(fixtures, squads, N)

All inputs/outputs conform to the strict JSON data contract.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ipl_engine.pipeline   import (
    load_ball_by_ball,
    load_matches,
    compute_player_match_stats,
    compute_player_aggregate_stats,
)
from ipl_engine.ratings    import compute_player_ratings as _compute_ratings
from ipl_engine.team_model import (
    compute_team_strength as _compute_team_strength,
    compute_venue_bias,
    compute_toss_adjustment,
)
from ipl_engine.predictor  import (
    predict_match   as _predict_match,
    simulate_season as _simulate_season,
    K_DEFAULT,
)


# ---------------------------------------------------------------------------
# 1. compute_player_stats
# ---------------------------------------------------------------------------

def compute_player_stats(
    bbb_path: str,
    matches_path: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stage 1: Raw data → feature tables.

    Parameters
    ----------
    bbb_path      : path to ball-by-ball CSV
    matches_path  : path to matches CSV (optional; used for venue/toss tables)

    Returns
    -------
    (player_match_stats, player_aggregate_stats)
    Both are DataFrames matching the table contracts defined in pipeline.py.
    """
    bbb = load_ball_by_ball(bbb_path)
    pms = compute_player_match_stats(bbb)
    agg = compute_player_aggregate_stats(pms)
    return pms, agg


# ---------------------------------------------------------------------------
# 2. compute_player_ratings
# ---------------------------------------------------------------------------

def compute_player_ratings(player_agg_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 2: Aggregate stats → normalized 0–1 player scores.

    Input:  player_aggregate_stats DataFrame
    Output: PlayerRating DataFrame

    Schema: [player, batting_score, bowling_score, allrounder_score,
              is_batting_qualified, is_bowling_qualified, is_allrounder]
    """
    return _compute_ratings(player_agg_stats)


# ---------------------------------------------------------------------------
# 3. compute_team_strength
# ---------------------------------------------------------------------------

def compute_team_strength(
    squad: list[str],
    player_ratings: pd.DataFrame,
    team_name: str = "UNKNOWN",
) -> dict:
    """
    Stage 3: Squad + ratings → TeamStrength dict.

    Parameters
    ----------
    squad          : list of player name strings
    player_ratings : output of compute_player_ratings()
    team_name      : label injected into the output dict

    Returns
    -------
    TeamStrength dict:
    {
        "team":               str,
        "batting_unit":       float,
        "bowling_unit":       float,
        "allrounder_balance": float,
        "total_strength":     float,
        "squad_size":         int,
        "squad_matched":      int,
    }
    """
    result = _compute_team_strength(squad, player_ratings)
    result["team"] = team_name
    return result


# ---------------------------------------------------------------------------
# 4. predict_match
# ---------------------------------------------------------------------------

def predict_match(
    teamA: str,
    teamB: str,
    squads: dict[str, list[str]],
    player_ratings: pd.DataFrame,
    context: dict | None = None,
    venue_bias_table: pd.DataFrame | None = None,
    toss_adjustment: float = 0.0,
    k: float = K_DEFAULT,
) -> dict:
    """
    Stage 4: Predict win probabilities for a single match.

    Parameters
    ----------
    teamA / teamB    : team name strings
    squads           : { team_name: [player, ...] }
    player_ratings   : output of compute_player_ratings()
    context          : optional dict with keys:
                          "venue"        : str
                          "toss_winner"  : str
                          "toss_decision": str  ("bat" | "field")
    venue_bias_table : output of compute_venue_bias() (optional)
    toss_adjustment  : output of compute_toss_adjustment() (optional)
    k                : logistic scaling constant

    Returns
    -------
    MatchPrediction dict:
    {
        "teamA":             str,
        "teamB":             str,
        "win_probability_A": float,
        "win_probability_B": float,
        "base_diff":         float,
        "venue_adjustment":  float,
        "toss_adjustment":   float,
        "final_score":       float,
        "venue":             str | None,
        "toss_winner":       str | None,
    }
    """
    ctx = context or {}

    sa = _compute_team_strength(squads.get(teamA, []), player_ratings)["total_strength"]
    sb = _compute_team_strength(squads.get(teamB, []), player_ratings)["total_strength"]

    return _predict_match(
        team_a           = teamA,
        team_b           = teamB,
        strength_a       = sa,
        strength_b       = sb,
        venue            = ctx.get("venue"),
        toss_winner      = ctx.get("toss_winner"),
        toss_decision    = ctx.get("toss_decision"),
        venue_bias_table = venue_bias_table,
        toss_adjustment  = toss_adjustment,
        k                = k,
    )


# ---------------------------------------------------------------------------
# 5. simulate_season
# ---------------------------------------------------------------------------

def simulate_season(
    fixtures: list[dict],
    squads: dict[str, list[str]],
    player_ratings: pd.DataFrame,
    venue_bias_table: pd.DataFrame | None = None,
    toss_adjustment: float = 0.0,
    N: int = 10_000,
    k: float = K_DEFAULT,
    random_seed: int | None = 42,
    season: str | None = None,
    squads_by_season: dict[str, dict[str, list[str]]] | None = None,
) -> list[dict]:
    """
    Stage 5: Monte Carlo simulation over a season fixture list.

    Parameters
    ----------
    fixtures : list of fixture dicts:
        [{ "match_id": str, "teamA": str, "teamB": str,
           "venue": str, "toss_winner": str | None }, ...]
    squads          : { team_name: [player, ...] }
    player_ratings  : output of compute_player_ratings()
    N               : number of Monte Carlo iterations
    k               : logistic scaling constant
    random_seed     : for reproducibility

    Returns
    -------
    List of SimulationResult dicts, sorted by title_prob desc:
    [
        {
            "team":         str,
            "title_prob":   float,
            "playoff_prob": float,
            "avg_points":   float,
            "avg_wins":     float,
        },
        ...
    ]
    """
    # ADD: resolve season-specific squads if provided
    if season and squads_by_season and season in squads_by_season:
        from backtest import _canonical
        season_map = squads_by_season[season]
        resolved: dict[str, list[str]] = {}
        for team in squads:
            resolved[team] = (
                season_map.get(team)
                or season_map.get(_canonical(team))
                or squads.get(team, [])
            )
        squads = resolved

    
    return _simulate_season(
        fixtures         = fixtures,
        squads           = squads,
        player_ratings   = player_ratings,
        venue_bias_table = venue_bias_table,
        toss_adjustment  = toss_adjustment,
        n_simulations    = N,
        k                = k,
        random_seed      = random_seed,
    )


# ---------------------------------------------------------------------------
# CONTEXT BUILDERS  (helpers for frontend / orchestration layer)
# ---------------------------------------------------------------------------

def build_context_tables(matches_path: str) -> dict:
    """
    Computes venue_bias_table and toss_adjustment from match history.
    Returns dict with keys: "venue_bias_table", "toss_adjustment".
    """
    matches = load_matches(matches_path)
    vbt     = compute_venue_bias(matches)
    ta      = compute_toss_adjustment(matches)
    return {
        "venue_bias_table": vbt,
        "toss_adjustment":  ta,
    }

# engine.py — append at bottom

import hashlib, pickle, os

def _cache_key(bbb_path: str, matches_path: str) -> str:
    """Stable key based on file sizes + mtimes (no content hash needed)."""
    def _sig(p: str) -> str:
        s = os.stat(p)
        return f"{s.st_size}:{s.st_mtime}"
    raw = f"{bbb_path}|{_sig(bbb_path)}|{matches_path}|{_sig(matches_path)}"
    return hashlib.md5(raw.encode()).hexdigest()

def load_or_compute(
    bbb_path: str,
    matches_path: str,
    cache_dir: str = ".ipl_cache",
) -> dict:
    """
    Returns { player_ratings, venue_bias_table, toss_adjustment }.
    Reads from pickle if inputs haven't changed; recomputes otherwise.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key  = _cache_key(bbb_path, matches_path)
    path = os.path.join(cache_dir, f"{key}.pkl")

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # Cold path
    _, agg   = compute_player_stats(bbb_path, matches_path)
    ratings  = compute_player_ratings(agg)
    ctx      = build_context_tables(matches_path)

    # Inside load_or_compute cold path — convert to parquet if not done
    def _to_parquet(csv_path: str) -> str:
        pq = csv_path.replace(".csv", ".parquet")
        if not os.path.exists(pq):
            pd.read_csv(csv_path).to_parquet(pq, index=False)
        return pq

    bbb_path     = _to_parquet(bbb_path)     if bbb_path.endswith(".csv")     else bbb_path
    matches_path = _to_parquet(matches_path) if matches_path.endswith(".csv") else matches_path
    
    result = {
        "player_ratings":   ratings,
        "venue_bias_table": ctx["venue_bias_table"],
        "toss_adjustment":  ctx["toss_adjustment"],
    }
    with open(path, "wb") as f:
        pickle.dump(result, f, protocol=5)

    return result