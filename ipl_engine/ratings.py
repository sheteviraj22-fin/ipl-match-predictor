"""
ratings.py
==========
Stage 2 of the IPL prediction backend.

Converts player_aggregate_stats into normalized 0–1 scores:

    BattingScore
    BowlingScore
    AllRounderScore

All normalization is min-max over the qualified player pool.
No hardcoded values. Fully data-driven.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# QUALIFICATION THRESHOLDS
# ---------------------------------------------------------------------------

MIN_BALLS_FACED   = 30
MIN_BALLS_BOWLED  = 60

# ---------------------------------------------------------------------------
# FORMULA WEIGHTS
# ---------------------------------------------------------------------------

BATTING_WEIGHTS = {
    "strike_rate":              0.30,
    "avg_runs":                 0.25,
    "boundary_rate":            0.15,
    "consistency_score_inv":    0.15,
    "pressure_strike_rate":     0.10,
    "experience_score":         0.05,
}

BOWLING_WEIGHTS = {
    "wicket_rate":              0.35,
    "economy_inv":              0.25,
    "dot_ball_rate":            0.25,
    "pressure_wickets_rate":    0.10,
    "experience_score":         0.05,
}

ALLROUNDER_BATTING_WEIGHT  = 0.50
ALLROUNDER_BOWLING_WEIGHT  = 0.50

ALLROUNDER_MIN_BATTING_QUAL = MIN_BALLS_FACED
ALLROUNDER_MIN_BOWLING_QUAL = MIN_BALLS_BOWLED

MISSING_BATTING_SCORE = 0.5
MISSING_BOWLING_SCORE = 0.5

# ---------------------------------------------------------------------------
# UTILITY
# ---------------------------------------------------------------------------

def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def _inverse(series: pd.Series) -> pd.Series:
    return series.max() - series


def _clip_outliers(series: pd.Series, q_lo: float = 0.01, q_hi: float = 0.99) -> pd.Series:
    lo = series.quantile(q_lo)
    hi = series.quantile(q_hi)
    return series.clip(lo, hi)


# ---------------------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------------------

def compute_player_ratings(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  player_aggregate_stats  (from pipeline.compute_player_aggregate_stats)
    Output: PlayerRating table

    Schema
    ------
    player, batting_score, bowling_score, allrounder_score,
    is_batting_qualified, is_bowling_qualified, is_allrounder
    """
    df = agg.copy()

    # Ensure optional columns exist with safe defaults
    for col in ["pressure_strike_rate", "experience_score", "pressure_wickets",
                "total_balls_bowled", "innings_bowled"]:
        if col not in df.columns:
            df[col] = 0.0

    # ---- BATTING -----------------------------------------------------------
    bat_pool = df[df["total_balls"] >= MIN_BALLS_FACED].copy()

    if len(bat_pool) > 0:
        bat_pool["_n_strike_rate"]     = _minmax(_clip_outliers(bat_pool["strike_rate"]))
        bat_pool["_n_avg_runs"]        = _minmax(_clip_outliers(bat_pool["avg_runs"]))
        bat_pool["_n_boundary_rate"]   = _minmax(_clip_outliers(bat_pool["boundary_rate"]))
        bat_pool["_n_consistency_inv"] = _minmax(_inverse(_clip_outliers(bat_pool["consistency_score"])))
        bat_pool["_n_pressure_sr"]     = _minmax(_clip_outliers(bat_pool["pressure_strike_rate"]))
        bat_pool["_n_experience"]      = _minmax(bat_pool["experience_score"])

        bat_pool["batting_score"] = (
            BATTING_WEIGHTS["strike_rate"]           * bat_pool["_n_strike_rate"]
            + BATTING_WEIGHTS["avg_runs"]            * bat_pool["_n_avg_runs"]
            + BATTING_WEIGHTS["boundary_rate"]       * bat_pool["_n_boundary_rate"]
            + BATTING_WEIGHTS["consistency_score_inv"] * bat_pool["_n_consistency_inv"]
            + BATTING_WEIGHTS["pressure_strike_rate"] * bat_pool["_n_pressure_sr"]
            + BATTING_WEIGHTS["experience_score"]    * bat_pool["_n_experience"]
        )
        bat_pool["is_batting_qualified"] = True
    else:
        bat_pool["batting_score"]        = MISSING_BATTING_SCORE
        bat_pool["is_batting_qualified"] = True

    # ---- BOWLING -----------------------------------------------------------
    bowl_pool = df[df["total_balls_bowled"] >= MIN_BALLS_BOWLED].copy()

    if len(bowl_pool) > 0:
        # pressure_wickets_rate: pressure_wickets / innings_bowled (normalised)
        bowl_pool["_pressure_wick_rate"] = (
            bowl_pool["pressure_wickets"] / bowl_pool["innings_bowled"].replace(0, np.nan)
        ).fillna(0)

        bowl_pool["_n_wicket_rate"]        = _minmax(_clip_outliers(bowl_pool["wicket_rate"]))
        bowl_pool["_n_economy_inv"]        = _minmax(_inverse(_clip_outliers(bowl_pool["economy"])))
        bowl_pool["_n_dot_ball_rate"]      = _minmax(_clip_outliers(bowl_pool["dot_ball_rate"]))
        bowl_pool["_n_pressure_wick_rate"] = _minmax(_clip_outliers(bowl_pool["_pressure_wick_rate"]))
        bowl_pool["_n_experience"]         = _minmax(bowl_pool["experience_score"])

        bowl_pool["bowling_score"] = (
            BOWLING_WEIGHTS["wicket_rate"]          * bowl_pool["_n_wicket_rate"]
            + BOWLING_WEIGHTS["economy_inv"]        * bowl_pool["_n_economy_inv"]
            + BOWLING_WEIGHTS["dot_ball_rate"]      * bowl_pool["_n_dot_ball_rate"]
            + BOWLING_WEIGHTS["pressure_wickets_rate"] * bowl_pool["_n_pressure_wick_rate"]
            + BOWLING_WEIGHTS["experience_score"]   * bowl_pool["_n_experience"]
        )
        bowl_pool["is_bowling_qualified"] = True
    else:
        bowl_pool["bowling_score"]        = MISSING_BOWLING_SCORE
        bowl_pool["is_bowling_qualified"] = True

    # ---- MERGE -------------------------------------------------------------
    ratings = df[["player"]].copy()
    ratings = ratings.merge(
        bat_pool[["player", "batting_score", "is_batting_qualified"]],
        on="player", how="left"
    ).merge(
        bowl_pool[["player", "bowling_score", "is_bowling_qualified"]],
        on="player", how="left"
    )

    ratings["batting_score"]        = ratings["batting_score"].fillna(MISSING_BATTING_SCORE)
    ratings["bowling_score"]        = ratings["bowling_score"].fillna(MISSING_BOWLING_SCORE)
    ratings["is_batting_qualified"] = ratings["is_batting_qualified"].fillna(False)
    ratings["is_bowling_qualified"] = ratings["is_bowling_qualified"].fillna(False)

    # ---- ALL-ROUNDER -------------------------------------------------------
    is_ar = ratings["is_batting_qualified"] & ratings["is_bowling_qualified"]
    ratings["is_allrounder"] = is_ar

    ratings["allrounder_score"] = np.where(
        is_ar,
        ALLROUNDER_BATTING_WEIGHT * ratings["batting_score"]
        + ALLROUNDER_BOWLING_WEIGHT * ratings["bowling_score"],
        0.0
    )

    output_cols = [
        "player",
        "batting_score", "bowling_score", "allrounder_score",
        "is_batting_qualified", "is_bowling_qualified", "is_allrounder",
    ]
    return ratings[output_cols].reset_index(drop=True)