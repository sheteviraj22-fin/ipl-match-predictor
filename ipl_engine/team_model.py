"""
team_model.py
=============
Stage 3 of the IPL prediction backend.

Computes TeamStrength from:
  - current squad (dynamic input)
  - player_ratings (from ratings.py)

Also computes:
  - venue_bias_factor from historical match data
  - toss_adjustment from historical toss→win correlation
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# FORMULA WEIGHTS  (TeamStrength)
# ---------------------------------------------------------------------------

TEAM_BATTING_WEIGHT    = 0.45
TEAM_BOWLING_WEIGHT    = 0.40
TEAM_ALLROUNDER_WEIGHT = 0.15

TOP_BATTERS  = 7   # pick top N by BattingScore
TOP_BOWLERS  = 5   # pick top N by BowlingScore


# ---------------------------------------------------------------------------
# TEAM STRENGTH
# ---------------------------------------------------------------------------

def compute_team_strength(
    squad: list[str],
    player_ratings: pd.DataFrame,
) -> dict:
    """
    Compute TeamStrength for a single team.

    Parameters
    ----------
    squad          : list of player names (current squad)
    player_ratings : output of ratings.compute_player_ratings()

    Returns
    -------
    TeamStrength dict (matches JSON contract)
    {
        team        : str,          # caller must inject
        batting_unit        : float,
        bowling_unit        : float,
        allrounder_balance  : float,
        total_strength      : float,
        squad_size          : int,
        squad_matched       : int,  # players found in historical data
    }
    """
    pr = player_ratings.copy()

    # Filter to squad members only
    squad_ratings = pr[pr["player"].isin(squad)]
    matched = len(squad_ratings)

    if matched == 0:
        return {
            "batting_unit": 0.0,
            "bowling_unit": 0.0,
            "allrounder_balance": 0.0,
            "total_strength": 0.0,
            "squad_size": len(squad),
            "squad_matched": 0,
        }

    # BattingUnit: mean of top-N batters in squad
    top_bat = (squad_ratings
               .nlargest(TOP_BATTERS, "batting_score")["batting_score"]
               .mean())

    # BowlingUnit: mean of top-N bowlers in squad
    top_bowl = (squad_ratings
                .nlargest(TOP_BOWLERS, "bowling_score")["bowling_score"]
                .mean())

    # AllRounderBalance: mean allrounder_score of qualified all-rounders
    ars = squad_ratings[squad_ratings["is_allrounder"]]["allrounder_score"]
    ar_balance = ars.mean() if len(ars) > 0 else 0.0

    total = (
        TEAM_BATTING_WEIGHT    * top_bat
        + TEAM_BOWLING_WEIGHT  * top_bowl
        + TEAM_ALLROUNDER_WEIGHT * ar_balance
    )

    return {
        "batting_unit":       round(float(top_bat),    6),
        "bowling_unit":       round(float(top_bowl),   6),
        "allrounder_balance": round(float(ar_balance), 6),
        "total_strength":     round(float(total),      6),
        "squad_size":         len(squad),
        "squad_matched":      matched,
    }


# ---------------------------------------------------------------------------
# VENUE BIAS
# ---------------------------------------------------------------------------

def compute_venue_bias(matches: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a venue_bias_factor per venue.

    Definition:
        venue_bias_factor = (home_win_rate − 0.5)
        where home_win_rate = fraction of matches where toss_winner == winning_team
        at that venue.

        Range: [−0.5, +0.5].
        Positive → batting-first side historically advantages.
        Negative → chasing side historically advantages.

    Returns a DataFrame with columns: [venue, venue_bias_factor, sample_size]
    """
    df = matches.copy()

    # Flag: did toss winner win the match?
    df["toss_winner_won"] = (df["toss_winner"] == df["winning_team"]).astype(int)

    venue_stats = (df.groupby("venue")
                   .agg(
                       toss_wins   = ("toss_winner_won", "sum"),
                       sample_size = ("toss_winner_won", "count"),
                   )
                   .reset_index())

    venue_stats["toss_win_rate"]     = venue_stats["toss_wins"] / venue_stats["sample_size"]
    venue_stats["venue_bias_factor"] = venue_stats["toss_win_rate"] - 0.5

    # Shrink estimates with low sample size toward zero (Bayesian-style)
    MIN_SAMPLE = 5
    venue_stats.loc[venue_stats["sample_size"] < MIN_SAMPLE, "venue_bias_factor"] = 0.0

    return venue_stats[["venue", "venue_bias_factor", "sample_size"]]


# ---------------------------------------------------------------------------
# TOSS ADJUSTMENT
# ---------------------------------------------------------------------------

def compute_toss_adjustment(matches: pd.DataFrame) -> float:
    """
    Computes a global toss_adjustment scalar.

    Definition:
        P(toss_winner_wins) across entire dataset − 0.5
        Represents average advantage from winning toss.

    Returns float in [−0.5, 0.5].
    """
    df = matches.copy()
    df["toss_winner_won"] = (df["toss_winner"] == df["winning_team"]).astype(int)
    p = df["toss_winner_won"].mean()
    return round(float(p - 0.5), 6)
