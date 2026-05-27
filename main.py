"""
main.py — IPL Prediction API
Thin orchestration layer over ipl_engine.engine. No logic lives here.
"""

from __future__ import annotations

from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from engine import (
    compute_player_stats,
    compute_player_ratings,
    compute_team_strength,
    predict_match,
    simulate_season,
    build_context_tables,
)
from backtest import backtest_model

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="IPL Prediction API", version="1.1.0")

# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_STATE: dict[str, Any] = {
    "player_ratings":    None,
    "venue_bias_table":  None,
    "toss_adjustment":   None,
    "loaded":            False,
}
# main.py — add after _STATE definition

import os, threading

_DEFAULT_BBB = os.getenv(
    "IPL_BBB_PATH",
    "dataset/IPL.csv"
)

_DEFAULT_MATCHES = os.getenv(
    "IPL_MATCHES_PATH",
    "dataset/matches.csv"
)

def _background_load() -> None:
    print("BBB:", _DEFAULT_BBB)
    print("MATCHES:", _DEFAULT_MATCHES)
    try:
        from engine import load_or_compute
        result = load_or_compute(_DEFAULT_BBB, _DEFAULT_MATCHES)
        _STATE.update({**result, "loaded": True})
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Background preload failed: %s", exc)

threading.Thread(target=_background_load, daemon=True).start()

def _require_data() -> None:
    if not _STATE["loaded"]:
        raise HTTPException(status_code=400, detail="Data not loaded. Call /load-data first.")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class LoadDataRequest(BaseModel):
    bbb_path: str
    matches_path: str


class TeamStrengthRequest(BaseModel):
    team: str
    squad: list[str]


class PredictMatchRequest(BaseModel):
    teamA: str
    teamB: str
    squads: dict[str, list[str]]
    venue: str | None = None
    toss_winner: str | None = None


class Fixture(BaseModel):
    match_id: str
    teamA: str
    teamB: str
    venue: str | None = None
    toss_winner: str | None = None


class SimulateSeasonRequest(BaseModel):
    fixtures: list[Fixture]
    squads: dict[str, list[str]]
    simulations: int = 10_000
    season: str | None = None
    squads_by_season: dict[str, dict[str, list[str]]] | None = None

class BacktestRequest(BaseModel):
    bbb_path: str
    matches_path: str
    k: float = 10.0
    min_train_matches: int = 50
    # Optional: { "2019": { "MI": [...], ... }, ... }
    squads_by_season: dict[str, dict[str, list[str]]] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "data_loaded": _STATE["loaded"],
    }


    # main.py — replace load_data() body

    @app.post("/load-data")
    def load_data(req: LoadDataRequest):
        try:
            from engine import load_or_compute
            result = load_or_compute(req.bbb_path, req.matches_path)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        _STATE.update({**result, "loaded": True})
        return {"status": "data_loaded"}

@app.post("/reload-data")
def reload_data(req: LoadDataRequest):
    _STATE["loaded"] = False
    return load_data(req)


@app.post("/team-strength")
def team_strength(req: TeamStrengthRequest):
    _require_data()
    try:
        result = compute_team_strength(
            squad          = req.squad,
            player_ratings = _STATE["player_ratings"],
            team_name      = req.team,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/predict-match")
def predict_match_endpoint(req: PredictMatchRequest):
    _require_data()
    context = {
        "venue":       req.venue,
        "toss_winner": req.toss_winner,
    }
    try:
        result = predict_match(
            teamA            = req.teamA,
            teamB            = req.teamB,
            squads           = req.squads,
            player_ratings   = _STATE["player_ratings"],
            context          = context,
            venue_bias_table = _STATE["venue_bias_table"],
            toss_adjustment  = _STATE["toss_adjustment"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.post("/simulate-season")
def simulate_season_endpoint(req: SimulateSeasonRequest):
    _require_data()
    fixtures = [f.model_dump() for f in req.fixtures]
    try:
        results = simulate_season(
            fixtures=req.fixtures,
            squads=req.squads,
            simulations=req.simulations,
            season=req.season,
            squads_by_season=req.squads_by_season,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return results


# ---------------------------------------------------------------------------
# NEW: Backtesting endpoint
# ---------------------------------------------------------------------------

@app.post("/backtest")
def backtest_endpoint(req: BacktestRequest):
    """
    Walk-forward backtesting with strict time split (no lookahead bias).

    Returns:
        accuracy, total_matches, correct, per_season breakdown,
        best/worst season, streak stats, and full match_log.

    This is compute-intensive — expect 5–60 seconds depending on dataset size.
    """
    try:
        result = backtest_model(
            matches_path     = req.matches_path,
            bbb_path         = req.bbb_path,
            squads_by_season = req.squads_by_season,
            k                = req.k,
            min_train_matches= req.min_train_matches,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)