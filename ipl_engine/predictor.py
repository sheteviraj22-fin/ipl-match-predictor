"""
predictor_updated.py  (regression edition)
==========================================
Drop-in replacement for ipl_engine/predictor.py

Model change
------------
OLD: final_score = base_diff + venue_adj + toss_adj
     prob = sigmoid(k × final_score)

NEW: Z  = INTERCEPT + Σ(coef_i × feature_i)    [logistic regression]
     P(teamA wins) = sigmoid(Z)

Eighteen features are derived from squad composition, player ratings, venue
characteristics, and match context.  When optional squad / player-rating data
is absent the function degrades gracefully to strength-based proxies, so all
existing callers that pass only strength_a / strength_b continue to work.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Model parameters
# ─────────────────────────────────────────────────────────────────────────────

K_DEFAULT: float = 10.0          # kept for interface compatibility; not used in regression

INTERCEPT: float = 0.1627

# Logistic-regression coefficients — all features are teamA-perspective
# (positive value → favours team A winning)
COEF: dict[str, float] = {
    "star_player_diff":        0.3060,
    "chasing_advantage":       0.2572,
    "batting_strength_diff":  -0.2386,
    "experience_diff":         0.2371,
    "bat_x_chasing":          -0.2164,
    "venue_type":             -0.1720,
    "bowling_strength_diff":  -0.1579,
    "home_ground_advantage":   0.1373,
    "balance_index_diff":     -0.1016,
    "allrounder_diff":        -0.0842,
    "bat_x_venue":            -0.0839,
    "venue_winrate_diff":      0.0542,
    "bowling_depth_diff":      0.0414,
    "toss_advantage":         -0.0372,
    "batting_depth_diff":      0.0301,
    "venue_wickets_avg":       0.0204,
    "recent_form_diff":       -0.0106,
    "venue_avg_runs":          0.0104,
}

# Top-N players by overall rating are counted as "stars"
STAR_N: int = 3

# Player is counted toward batting/bowling "depth" if rating exceeds this
DEPTH_THRESHOLD: float = 0.50

# Home-ground keyword table — substring matched against venue name (lower-cased)
HOME_GROUNDS: dict[str, list[str]] = {
    "Mumbai Indians":               ["wankhede", "mumbai"],
    "Chennai Super Kings":          ["chepauk", "chennai", "ma chidambaram"],
    "Royal Challengers Bengaluru":  ["chinnaswamy", "bengaluru", "bangalore"],
    "Royal Challengers Bangalore":  ["chinnaswamy", "bengaluru", "bangalore"],
    "Delhi Capitals":               ["arun jaitley", "feroz shah", "kotla", "delhi"],
    "Kolkata Knight Riders":        ["eden gardens", "kolkata"],
    "Rajasthan Royals":             ["sawai mansingh", "jaipur"],
    "Sunrisers Hyderabad":          ["rajiv gandhi", "hyderabad", "uppal"],
    "Punjab Kings":                 ["pca", "mohali", "mullanpur", "dharamsala"],
    "Lucknow Super Giants":         ["ekana", "lucknow"],
    "Gujarat Titans":               ["narendra modi", "ahmedabad", "motera"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, *aliases: str, default: float = 0.0) -> pd.Series:
    """
    Return the first column in *aliases* that exists in *df*.
    Falls back to a constant Series of *default* if none are found.
    """
    for alias in aliases:
        if alias in df.columns:
            return df[alias].fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


def _safe(row: Any, *keys: str, fallback: float = 0.0) -> float:
    """
    Try each key against a pandas Series / dict row in order.
    Returns the first non-null numeric value found, else *fallback*.
    """
    for key in keys:
        try:
            v = row[key]
            if pd.notna(v):
                return float(v)
        except (KeyError, TypeError, IndexError):
            pass
    return fallback


def _is_home(team: str, venue: str | None) -> float:
    """Return 1.0 if *venue* is a recognised home ground for *team*, else 0.0."""
    if not venue:
        return 0.0
    vl = venue.lower()
    return 1.0 if any(kw in vl for kw in HOME_GROUNDS.get(team, [])) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _squad_features(
    team: str,
    squads: dict[str, list[str]] | None,
    player_ratings: pd.DataFrame | None,
) -> dict[str, float]:
    """
    Derive per-squad features from player-rating data.

    Accepted column aliases (first match wins)
    ------------------------------------------
    batting  : batting_rating | batting | bat_rating | bat
    bowling  : bowling_rating | bowling | bowl_rating | bowl
    allround : allrounder_rating | allrounder | ar_rating
    experience: experience | experience_rating | exp_rating | exp
    form     : recent_form | form | recent_form_rating
    overall  : overall_rating | total_strength | overall | rating

    Returns neutral values (0.5) when squad / rating data is absent.
    """
    _neutral: dict[str, float] = {
        "batting_strength":    0.5,
        "bowling_strength":    0.5,
        "allrounder_strength": 0.0,
        "experience":          0.5,
        "star_player":         0.5,
        "batting_depth":       0.5,
        "bowling_depth":       0.5,
        "balance_index":       0.5,
        "recent_form":         0.5,
    }

    if player_ratings is None or player_ratings.empty:
        return _neutral

    # Detect player-name column
    name_col = next(
        (c for c in ("player", "name", "player_name", "Player") if c in player_ratings.columns),
        None,
    )

    squad_list = (squads or {}).get(team, [])
    if name_col and squad_list:
        df = player_ratings[player_ratings[name_col].isin(squad_list)].copy()
    else:
        df = player_ratings.copy()

    if df.empty:
        return _neutral

    bat  = _col(df, "batting_rating",    "batting",   "bat_rating",     "bat",       default=0.5)
    bowl = _col(df, "bowling_rating",    "bowling",   "bowl_rating",    "bowl",      default=0.5)
    ar   = _col(df, "allrounder_rating", "allrounder","ar_rating",                   default=0.0)
    exp  = _col(df, "experience",        "experience_rating", "exp_rating", "exp",   default=0.5)
    form = _col(df, "recent_form",       "form",      "recent_form_rating",          default=0.5)
    ovr  = _col(df, "overall_rating",    "total_strength", "overall", "rating",      default=0.0)

    # If no overall column found, synthesise from batting + bowling average
    if (ovr == 0.0).all():
        ovr = (bat + bowl) / 2.0

    # Star players: mean rating of top-STAR_N players
    top_n      = ovr.nlargest(STAR_N)
    star       = float(top_n.mean()) if not top_n.empty else 0.5

    bat_depth  = float((bat  > DEPTH_THRESHOLD).mean())
    bowl_depth = float((bowl > DEPTH_THRESHOLD).mean())

    b_mean = float(bat.mean())
    w_mean = float(bowl.mean())
    balance = b_mean / (b_mean + w_mean) if (b_mean + w_mean) > 0 else 0.5

    return {
        "batting_strength":    b_mean,
        "bowling_strength":    w_mean,
        "allrounder_strength": float(ar.mean()),
        "experience":          float(exp.mean()),
        "star_player":         star,
        "batting_depth":       bat_depth,
        "bowling_depth":       bowl_depth,
        "balance_index":       balance,
        "recent_form":         float(form.mean()),
    }


def _strength_proxy(strength: float) -> dict[str, float]:
    """
    Generate proxy squad features when only a scalar team-strength is available.
    Applies sigmoid normalisation so any real-valued strength maps to (0, 1).
    Used as fallback when squads / player_ratings are not supplied.
    """
    s = 1.0 / (1.0 + math.exp(-strength))   # sigmoid normalisation → (0, 1)
    return {
        "batting_strength":    s * 0.60,
        "bowling_strength":    s * 0.40,
        "allrounder_strength": s * 0.30,
        "experience":          s,
        "star_player":         s,
        "batting_depth":       s,
        "bowling_depth":       s,
        "balance_index":       0.5,
        "recent_form":         s,
    }


def _venue_features(
    venue: str | None,
    venue_bias_table: pd.DataFrame | None,
    team_a: str,
    team_b: str,
) -> dict[str, float]:
    """
    Extract venue-level scalar features.

    Accepted column aliases in venue_bias_table
    -------------------------------------------
    venue_type        : venue_type | venue_bias_factor
    venue_avg_runs    : venue_avg_runs | avg_runs
    venue_wickets_avg : venue_wickets_avg | avg_wickets
    team win-rates    : {team_slug}_winrate | winrate_{team_slug}
    """
    defaults: dict[str, float] = {
        "venue_type":        0.0,
        "venue_avg_runs":    0.0,
        "venue_wickets_avg": 0.0,
        "winrate_a":         0.5,
        "winrate_b":         0.5,
    }

    if not venue or venue_bias_table is None or venue_bias_table.empty:
        return defaults

    rows = venue_bias_table[venue_bias_table["venue"] == venue]
    if rows.empty:
        return defaults

    r = rows.iloc[0]

    vtype  = _safe(r, "venue_type",        "venue_bias_factor", fallback=0.0)
    avgr   = _safe(r, "venue_avg_runs",    "avg_runs",          fallback=0.0)
    avgw   = _safe(r, "venue_wickets_avg", "avg_wickets",       fallback=0.0)

    def _winrate(team: str) -> float:
        slug = team.lower().replace(" ", "_")
        return _safe(r,
                     f"{slug}_winrate",
                     f"winrate_{slug}",
                     fallback=0.5)

    return {
        "venue_type":        vtype,
        "venue_avg_runs":    avgr,
        "venue_wickets_avg": avgw,
        "winrate_a":         _winrate(team_a),
        "winrate_b":         _winrate(team_b),
    }


def _chasing_flag(
    team_a: str,
    toss_winner: str | None,
    toss_decision: str | None,
) -> float:
    """
    Return 1.0 if team_a bats second (is chasing), else 0.0.

    team_a chases when:
      • toss_winner == team_a  AND  toss_decision in {field, bowl}
      • toss_winner != team_a  AND  toss_decision == bat
        (team_b chose to bat first → team_a chases)
    """
    if not toss_winner or not toss_decision:
        return 0.0
    dec = toss_decision.strip().lower()
    if toss_winner == team_a and dec in ("field", "bowl"):
        return 1.0
    if toss_winner != team_a and dec == "bat":
        return 1.0
    return 0.0


def _compute_z(fv: dict[str, float]) -> float:
    """Dot-product of feature vector with COEF + INTERCEPT → regression score Z."""
    return INTERCEPT + sum(COEF[f] * fv.get(f, 0.0) for f in COEF)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  predict_match  (public API — backward-compatible signature)
# ─────────────────────────────────────────────────────────────────────────────

def predict_match(
    team_a: str,
    team_b: str,
    strength_a: float,
    strength_b: float,
    venue: str | None = None,
    toss_winner: str | None = None,
    toss_decision: str | None = None,
    venue_bias_table: pd.DataFrame | None = None,
    toss_adjustment: float = 0.0,
    k: float = K_DEFAULT,
    # ── NEW optional params (ignored by old callers) ─────────────────────────
    squads: dict[str, list[str]] | None = None,
    player_ratings: pd.DataFrame | None = None,
) -> dict:
    """
    Predict win probabilities for a single match using logistic regression.

    Parameters (unchanged)
    -----------------------
    team_a, team_b        : franchise names
    strength_a, strength_b: pre-computed team strength scalars (any scale)
    venue                 : ground name (optional)
    toss_winner           : name of team that won the toss (optional)
    toss_decision         : 'bat' or 'field' (optional)
    venue_bias_table      : DataFrame with venue stats (optional)
    toss_adjustment       : legacy additive offset — preserved for compatibility
    k                     : kept in signature; not used by regression model

    New optional parameters
    -----------------------
    squads         : { team_name: [player_name, ...] }
    player_ratings : DataFrame from compute_player_ratings()
                     When absent, strength_a / strength_b proxy is used.

    Returns
    -------
    JSON-serialisable dict with the same keys as before:
        teamA, teamB, win_probability_A, win_probability_B,
        base_diff, venue_adjustment, toss_adjustment, final_score,
        venue, toss_winner
    Plus regression diagnostics:
        regression_z, feature_vector
    """
    # ── Squad features ────────────────────────────────────────────────────────
    use_squad_data = (
        squads is not None
        and player_ratings is not None
        and not player_ratings.empty
    )
    if use_squad_data:
        fa = _squad_features(team_a, squads, player_ratings)
        fb = _squad_features(team_b, squads, player_ratings)
    else:
        fa = _strength_proxy(strength_a)
        fb = _strength_proxy(strength_b)

    # ── Venue features ────────────────────────────────────────────────────────
    vf = _venue_features(venue, venue_bias_table, team_a, team_b)

    # ── Match-context scalars ─────────────────────────────────────────────────
    chasing   = _chasing_flag(team_a, toss_winner, toss_decision)
    bat_diff  = fa["batting_strength"] - fb["batting_strength"]
    home_adv  = _is_home(team_a, venue) - _is_home(team_b, venue)
    toss_adv  = 1.0 if toss_winner == team_a else 0.0

    # ── Assemble feature vector ───────────────────────────────────────────────
    fv: dict[str, float] = {
        "star_player_diff":        fa["star_player"]         - fb["star_player"],
        "chasing_advantage":       chasing,
        "batting_strength_diff":   bat_diff,
        "experience_diff":         fa["experience"]          - fb["experience"],
        "bat_x_chasing":           bat_diff * chasing,
        "venue_type":              vf["venue_type"],
        "bowling_strength_diff":   fa["bowling_strength"]    - fb["bowling_strength"],
        "home_ground_advantage":   home_adv,
        "balance_index_diff":      fa["balance_index"]       - fb["balance_index"],
        "allrounder_diff":         fa["allrounder_strength"] - fb["allrounder_strength"],
        "bat_x_venue":             bat_diff * vf["venue_type"],
        "venue_winrate_diff":      vf["winrate_a"]           - vf["winrate_b"],
        "bowling_depth_diff":      fa["bowling_depth"]       - fb["bowling_depth"],
        "toss_advantage":          toss_adv,
        "batting_depth_diff":      fa["batting_depth"]       - fb["batting_depth"],
        "venue_wickets_avg":       vf["venue_wickets_avg"],
        "recent_form_diff":        fa["recent_form"]         - fb["recent_form"],
        "venue_avg_runs":          vf["venue_avg_runs"],
    }

    # ── Regression score + probability ───────────────────────────────────────
    z = _compute_z(fv)

    # toss_adjustment: legacy parameter kept as a small additive offset on Z.
    # Default is 0.0 so it has no effect unless explicitly set by old callers.
    z += float(toss_adjustment) * 0.10

    prob_a = 1.0 / (1.0 + math.exp(-z))
    prob_b = 1.0 - prob_a

    return {
        # ── Core prediction (same contract as before) ─────────────────────────
        "teamA":             team_a,
        "teamB":             team_b,
        "win_probability_A": round(prob_a, 8),
        "win_probability_B": round(prob_b, 8),
        # ── Regression diagnostics ────────────────────────────────────────────
        "regression_z":      round(z, 6),
        "feature_vector":    {k: round(v, 6) for k, v in fv.items()},
        # ── Legacy keys preserved for downstream JSON consumers ───────────────
        # base_diff  → most informative single-feature proxy (star player gap)
        # final_score → aliased to regression Z
        "base_diff":         round(fv["star_player_diff"], 6),
        "venue_adjustment":  round(vf["venue_type"], 6),
        "toss_adjustment":   round(toss_adv, 6),
        "final_score":       round(z, 6),
        "venue":             venue,
        "toss_winner":       toss_winner,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  simulate_season  (True Monte Carlo — updated to use regression model)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_season(
    fixtures: list[dict],
    squads: dict[str, list[str]],
    player_ratings: pd.DataFrame,
    venue_bias_table: pd.DataFrame | None = None,
    toss_adjustment: float = 0.0,
    n_simulations: int = 10_000,
    k: float = K_DEFAULT,
    random_seed: int | None = 42,
) -> list[dict]:
    """
    True Monte Carlo season simulation using regression-based probabilities.

    Algorithm
    ---------
    1. Pre-compute regression win probability for every league fixture.
    2. Draw N × F uniform samples in one vectorised call.
    3. Accumulate 2-pt wins per simulation with NumPy.
    4. Apply Gaussian NRR proxy noise for tiebreaking.
    5. Rank teams and run IPL playoff tree per simulation.
    6. Aggregate title / playoff qualification counts → probabilities.

    Parameters
    ----------
    fixtures         : list of { match_id, teamA, teamB, venue,
                                  toss_winner, toss_decision }
    squads           : { team_name: [player_name, ...] }
    player_ratings   : output of compute_player_ratings()
    venue_bias_table : optional venue stats DataFrame
    toss_adjustment  : legacy scalar (passed through to predict_match)
    n_simulations    : Monte Carlo sample count
    k                : kept for interface compat; not used in regression
    random_seed      : for reproducibility (None → random)

    Returns
    -------
    List of SimulationResult dicts sorted by title_prob desc:
        { team, title_prob, playoff_prob, avg_points, avg_wins }
    """
    from ipl_engine.team_model import compute_team_strength as _cts

    # ── Discover teams ────────────────────────────────────────────────────────
    team_set: set[str] = set()
    for f in fixtures:
        team_set.add(f["teamA"])
        team_set.add(f["teamB"])

    all_teams = sorted(team_set)
    n_teams   = len(all_teams)
    team_idx  = {t: i for i, t in enumerate(all_teams)}

    # Cache scalar strength (only used for _pw_get fallback path)
    _strength_cache: dict[str, float] = {}

    def _strength(team: str) -> float:
        if team not in _strength_cache:
            squad = squads.get(team, [])
            ts    = _cts(squad, player_ratings)
            _strength_cache[team] = float(ts["total_strength"])
        return _strength_cache[team]

    # ── Pre-compute regression win probabilities per fixture ──────────────────
    F       = len(fixtures)
    probs_a = np.empty(F, dtype=np.float64)

    for fi, fix in enumerate(fixtures):
        pred = predict_match(
            team_a           = fix["teamA"],
            team_b           = fix["teamB"],
            strength_a       = _strength(fix["teamA"]),
            strength_b       = _strength(fix["teamB"]),
            venue            = fix.get("venue"),
            toss_winner      = fix.get("toss_winner"),
            toss_decision    = fix.get("toss_decision"),
            venue_bias_table = venue_bias_table,
            toss_adjustment  = toss_adjustment,
            k                = k,
            squads           = squads,
            player_ratings   = player_ratings,
        )
        probs_a[fi] = pred["win_probability_A"]

    # ── Vectorised sampling — shape (N, F) ────────────────────────────────────
    N   = n_simulations
    rng = np.random.default_rng(random_seed)

    samples = rng.random((N, F))            # uniform [0, 1)
    a_wins  = samples < probs_a[None, :]   # True → teamA wins this fixture

    # ── Accumulate points — shape (N, n_teams) ────────────────────────────────
    points = np.zeros((N, n_teams), dtype=np.float32)

    for fi, fix in enumerate(fixtures):
        ai = team_idx[fix["teamA"]]
        bi = team_idx[fix["teamB"]]
        points[:, ai] += a_wins[:, fi].astype(np.float32) * 2
        points[:, bi] += (~a_wins[:, fi]).astype(np.float32) * 2

    # ── NRR proxy tiebreaker — tiny Gaussian noise breaks point ties ──────────
    nrr_noise = rng.standard_normal((N, n_teams)).astype(np.float32) * 0.05
    sort_key  = points + nrr_noise

    # ── Rank teams per simulation (descending) ────────────────────────────────
    rankings = np.argsort(-sort_key, axis=1)   # shape (N, n_teams)

    # ── Playoff win-probability cache ─────────────────────────────────────────
    # Uses regression model (with squad data) for knockout-match probs.
    _pw: dict[tuple[str, str], float] = {}

    def _pw_get(t1: str, t2: str) -> float:
        if (t1, t2) not in _pw:
            p = predict_match(
                t1, t2,
                _strength(t1), _strength(t2),
                venue_bias_table = venue_bias_table,
                toss_adjustment  = toss_adjustment,
                k                = k,
                squads           = squads,
                player_ratings   = player_ratings,
            )
            _pw[(t1, t2)] = p["win_probability_A"]
            _pw[(t2, t1)] = p["win_probability_B"]
        return _pw[(t1, t2)]

    # ── Playoff tree ──────────────────────────────────────────────────────────
    #   Qualifier 1  : rank[0] vs rank[1]  → winner → Final
    #   Eliminator   : rank[2] vs rank[3]  → winner → Q2
    #   Qualifier 2  : Q1-loser vs Elim-winner → winner → Final
    #   Final        : Q1-winner vs Q2-winner

    title_count   = np.zeros(n_teams, dtype=np.int64)
    playoff_count = np.zeros(n_teams, dtype=np.int64)

    # Full IPL playoff tree requires at least 4 teams.
    # With fewer teams we fall back to a simple head-to-head final.
    run_full_playoffs = n_teams >= 4

    playoff_rng = rng.random((N, 3))   # 3 pre-final knockout draws per sim

    for sim_i in range(N):
        r  = rankings[sim_i]
        rand = playoff_rng[sim_i]

        if run_full_playoffs:
            t1 = all_teams[r[0]]   # league 1st
            t2 = all_teams[r[1]]   # league 2nd
            t3 = all_teams[r[2]]   # league 3rd
            t4 = all_teams[r[3]]   # league 4th

            playoff_count[r[0]] += 1
            playoff_count[r[1]] += 1
            playoff_count[r[2]] += 1
            playoff_count[r[3]] += 1

            # Q1 (1st vs 2nd)
            q1_winner = t1 if rand[0] < _pw_get(t1, t2) else t2
            q1_loser  = t2 if q1_winner == t1 else t1

            # Eliminator (3rd vs 4th)
            elim_winner = t3 if rand[1] < _pw_get(t3, t4) else t4

            # Q2 (Q1-loser vs Elim-winner)
            q2_winner = (
                q1_loser if rand[2] < _pw_get(q1_loser, elim_winner) else elim_winner
            )

            # Final — combine prior draws for a cheap independent-looking sample
            final_rand = (rand[0] + rand[1] + rand[2]) % 1.0
            champion   = (
                q1_winner if final_rand < _pw_get(q1_winner, q2_winner) else q2_winner
            )
        else:
            # Fallback: top-2 play a straight final
            t1 = all_teams[r[0]]
            t2 = all_teams[r[1]] if n_teams > 1 else t1
            for ri in range(min(n_teams, 2)):
                playoff_count[r[ri]] += 1
            champion = t1 if rand[0] < _pw_get(t1, t2) else t2

        title_count[team_idx[champion]] += 1

    # ── Build output ──────────────────────────────────────────────────────────
    results: list[dict] = []
    for i, team in enumerate(all_teams):
        avg_pts = float(points[:, i].mean())
        results.append({
            "team":         team,
            "title_prob":   round(float(title_count[i])   / N, 6),
            "playoff_prob": round(float(playoff_count[i]) / N, 6),
            "avg_points":   round(avg_pts, 4),
            "avg_wins":     round(avg_pts / 2.0, 4),
        })

    return sorted(results, key=lambda x: x["title_prob"], reverse=True)
