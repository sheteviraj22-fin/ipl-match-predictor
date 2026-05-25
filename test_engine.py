"""
test_engine.py
==============
End-to-end integration test using the real IPL datasets.
Validates pipeline → ratings → team strength → prediction → simulation.
Run from /home/claude/ipl_engine/:  python -m pytest tests/ -v
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from ipl_engine.engine import (
    compute_player_stats,
    compute_player_ratings,
    compute_team_strength,
    predict_match,
    simulate_season,
    build_context_tables,
)

BBB_PATH     = "/mnt/user-data/uploads/ipl_ball_by_ball_2008_2022.csv"
MATCHES_PATH = "/mnt/user-data/uploads/ipl_matches_2008_2022.csv"


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def pms_agg():
    pms, agg = compute_player_stats(BBB_PATH, MATCHES_PATH)
    return pms, agg

@pytest.fixture(scope="session")
def ratings(pms_agg):
    _, agg = pms_agg
    return compute_player_ratings(agg)

@pytest.fixture(scope="session")
def ctx():
    return build_context_tables(MATCHES_PATH)

# ---------------------------------------------------------------------------
# STAGE 1 — pipeline
# ---------------------------------------------------------------------------

def test_player_match_stats_schema(pms_agg):
    pms, _ = pms_agg
    required = {"player", "match_id", "role", "runs_scored", "balls_faced",
                "strike_rate", "wickets_taken", "balls_bowled", "economy"}
    assert required.issubset(set(pms.columns)), f"Missing: {required - set(pms.columns)}"
    assert len(pms) > 1000

def test_player_aggregate_stats_schema(pms_agg):
    _, agg = pms_agg
    required = {"player", "avg_runs", "strike_rate", "boundary_rate",
                "consistency_score", "wicket_rate", "economy", "dot_ball_rate"}
    assert required.issubset(set(agg.columns))
    assert len(agg) > 100
    assert agg["player"].nunique() == len(agg), "Duplicate players in aggregate"

def test_no_negative_rates(pms_agg):
    _, agg = pms_agg
    for col in ["avg_runs", "strike_rate", "boundary_rate",
                "wicket_rate", "economy", "dot_ball_rate"]:
        assert (agg[col] >= 0).all(), f"Negative values in {col}"

# ---------------------------------------------------------------------------
# STAGE 2 — ratings
# ---------------------------------------------------------------------------

def test_ratings_schema(ratings):
    required = {"player", "batting_score", "bowling_score", "allrounder_score"}
    assert required.issubset(set(ratings.columns))

def test_ratings_bounds(ratings):
    for col in ["batting_score", "bowling_score", "allrounder_score"]:
        assert (ratings[col] >= 0).all() and (ratings[col] <= 1).all(), \
            f"{col} out of [0,1]"

def test_allrounder_requires_both(ratings):
    ar = ratings[ratings["is_allrounder"]]
    assert (ar["is_batting_qualified"]).all()
    assert (ar["is_bowling_qualified"]).all()

# ---------------------------------------------------------------------------
# STAGE 3 — team strength
# ---------------------------------------------------------------------------

SAMPLE_SQUAD_A = [
    "V Kohli", "RG Sharma", "MS Dhoni", "SK Raina", "KA Pollard",
    "DJ Bravo", "R Jadeja", "PP Chawla", "B Kumar", "JJ Bumrah", "SL Malinga"
]
SAMPLE_SQUAD_B = [
    "DA Warner", "S Dhawan", "AB de Villiers", "CH Gayle", "KL Rahul",
    "Y Singh", "R Ashwin", "IK Pathan", "UT Yadav", "Harbhajan Singh", "A Mishra"
]

def test_team_strength_schema(ratings):
    ts = compute_team_strength(SAMPLE_SQUAD_A, ratings, team_name="TeamA")
    for key in ["team", "batting_unit", "bowling_unit", "total_strength"]:
        assert key in ts, f"Missing key: {key}"

def test_team_strength_bounds(ratings):
    ts = compute_team_strength(SAMPLE_SQUAD_A, ratings, team_name="TeamA")
    assert 0.0 <= ts["total_strength"] <= 1.0

def test_empty_squad_returns_zero(ratings):
    ts = compute_team_strength([], ratings, team_name="Empty")
    assert ts["total_strength"] == 0.0

# ---------------------------------------------------------------------------
# STAGE 4 — match prediction
# ---------------------------------------------------------------------------

def test_predict_match_schema(ratings, ctx):
    squads = {"TeamA": SAMPLE_SQUAD_A, "TeamB": SAMPLE_SQUAD_B}
    pred = predict_match(
        teamA          = "TeamA",
        teamB          = "TeamB",
        squads         = squads,
        player_ratings = ratings,
        context        = {"venue": "Eden Gardens", "toss_winner": "TeamA"},
        venue_bias_table = ctx["venue_bias_table"],
        toss_adjustment  = ctx["toss_adjustment"],
    )
    for key in ["teamA", "teamB", "win_probability_A", "win_probability_B"]:
        assert key in pred

def test_probabilities_sum_to_one(ratings, ctx):
    squads = {"TeamA": SAMPLE_SQUAD_A, "TeamB": SAMPLE_SQUAD_B}
    pred = predict_match("TeamA", "TeamB", squads, ratings,
                         venue_bias_table=ctx["venue_bias_table"],
                         toss_adjustment=ctx["toss_adjustment"])
    assert abs(pred["win_probability_A"] + pred["win_probability_B"] - 1.0) < 1e-9

def test_symmetric_prediction(ratings):
    squads = {"A": SAMPLE_SQUAD_A, "B": SAMPLE_SQUAD_B}
    pa = predict_match("A", "B", squads, ratings)
    pb = predict_match("B", "A", squads, ratings)
    assert abs(pa["win_probability_A"] - pb["win_probability_B"]) < 1e-9

# ---------------------------------------------------------------------------
# STAGE 5 — Monte Carlo
# ---------------------------------------------------------------------------

SAMPLE_FIXTURES = [
    {"match_id": "1", "teamA": "TeamA", "teamB": "TeamB", "venue": "Eden Gardens", "toss_winner": None},
    {"match_id": "2", "teamA": "TeamB", "teamB": "TeamA", "venue": "Wankhede Stadium", "toss_winner": None},
    {"match_id": "3", "teamA": "TeamA", "teamB": "TeamB", "venue": "M Chinnaswamy Stadium", "toss_winner": None},
]
SAMPLE_SQUADS = {"TeamA": SAMPLE_SQUAD_A, "TeamB": SAMPLE_SQUAD_B}

def test_simulation_schema(ratings, ctx):
    results = simulate_season(
        fixtures         = SAMPLE_FIXTURES,
        squads           = SAMPLE_SQUADS,
        player_ratings   = ratings,
        venue_bias_table = ctx["venue_bias_table"],
        toss_adjustment  = ctx["toss_adjustment"],
        N                = 500,
    )
    assert len(results) == 2
    for r in results:
        for key in ["team", "title_prob", "playoff_prob", "avg_points"]:
            assert key in r

def test_simulation_title_probs_sum_to_one(ratings, ctx):
    results = simulate_season(
        fixtures=SAMPLE_FIXTURES, squads=SAMPLE_SQUADS,
        player_ratings=ratings, N=500,
    )
    total = sum(r["title_prob"] for r in results)
    assert abs(total - 1.0) < 0.02   # small Monte Carlo variance tolerance

def test_simulation_reproducible(ratings, ctx):
    r1 = simulate_season(SAMPLE_FIXTURES, SAMPLE_SQUADS, ratings, N=500, random_seed=99)
    r2 = simulate_season(SAMPLE_FIXTURES, SAMPLE_SQUADS, ratings, N=500, random_seed=99)
    assert r1 == r2

# ---------------------------------------------------------------------------
# JSON CONTRACT SERIALISABILITY
# ---------------------------------------------------------------------------

def test_all_outputs_json_serialisable(ratings, ctx):
    squads = {"TeamA": SAMPLE_SQUAD_A, "TeamB": SAMPLE_SQUAD_B}

    ts   = compute_team_strength(SAMPLE_SQUAD_A, ratings, "TeamA")
    pred = predict_match("TeamA", "TeamB", squads, ratings)
    sim  = simulate_season(SAMPLE_FIXTURES, squads, ratings, N=200)

    # Must not raise
    json.dumps(ts)
    json.dumps(pred)
    json.dumps(sim)
