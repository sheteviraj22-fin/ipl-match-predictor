"""
pipeline.py
===========
Stage 1 of the IPL prediction backend.

Transforms raw ball-by-ball and match CSVs into two structured
feature tables:

    player_match_stats   — per-player, per-match aggregates
    player_aggregate_stats — career-level normalized feature table

Column contracts are strict: downstream modules depend on exact names.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

POWERPLAY_OVERS   = range(0, 6)
MIDDLE_OVERS      = range(6, 15)
DEATH_OVERS       = range(15, 20)

MIN_BALLS_FACED_BATTING  = 30
MIN_BALLS_BOWLED_BOWLING = 60

RECENCY_LAMBDA    = 0.003   # time-decay rate; higher = faster decay
LAST_N_MATCHES    = 20      # rolling window per player

BATTING_POSITION_WEIGHTS = {
    "top":    1.20,   # positions 1–3
    "middle": 1.00,   # positions 4–6
    "lower":  0.85,   # positions 7+
}

FORM_CAREER_WEIGHT_RECENT  = 0.70
FORM_CAREER_WEIGHT_CAREER  = 0.30

MISSING_BATTING_SCORE  = 0.5
MISSING_BOWLING_SCORE  = 0.5


# ---------------------------------------------------------------------------
# LOADERS
# ---------------------------------------------------------------------------

def load_ball_by_ball(path: str) -> pd.DataFrame:
    """
    Load and normalise ball-by-ball dataset.

    Accepted schemas:
      - ipl_ball_by_ball_2008_2022.csv  (minimal schema)
      - IPL.csv                          (enriched schema)

    Returns a canonical DataFrame with all original + derived columns.
    """
    df = pd.read_csv(path, low_memory=False)
    col = set(df.columns)

    rename_map: dict[str, str] = {}

    if "overs" in col and "over" not in col:
        rename_map["overs"] = "over"
    if "over" in col and "ball_number" in col:
        rename_map["ball_number"] = "ball"
    if "ball" not in col and "ball_no" in col:
        rename_map["ball_no"] = "ball"

    if "id" in col and "match_id" not in col:
        rename_map["id"] = "match_id"
    if "match_id" not in col and "id" not in col:
        raise KeyError("Cannot find match_id / id column in ball-by-ball data")

    df = df.rename(columns=rename_map)

    # Wicket flag: prefer bowler_wicket (Module 5)
    if "bowler_wicket" in df.columns:
        df["iswicket_delivery"] = pd.to_numeric(df["bowler_wicket"], errors="coerce").fillna(0)
    elif "iswicket_delivery" not in df.columns:
        df["iswicket_delivery"] = 0
    else:
        df["iswicket_delivery"] = pd.to_numeric(df["iswicket_delivery"], errors="coerce").fillna(0)

    if "valid_ball" not in df.columns:
        df["valid_ball"] = 1

    required = ["match_id", "innings", "over", "ball", "batter", "bowler",
                 "runs_batter", "iswicket_delivery", "batting_team", "valid_ball"]
    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    df["runs_batter"]       = pd.to_numeric(df["runs_batter"],       errors="coerce").fillna(0)
    df["iswicket_delivery"] = pd.to_numeric(df["iswicket_delivery"], errors="coerce").fillna(0)
    df["over"]              = pd.to_numeric(df["over"],              errors="coerce").fillna(0).astype(int)
    df["valid_ball"]        = pd.to_numeric(df["valid_ball"],        errors="coerce").fillna(1).astype(int)
    df["match_id"]          = df["match_id"].astype(str)

    df["is_boundary"] = ((df["runs_batter"] == 4) | (df["runs_batter"] == 6)).astype(int)

    if "runs_total" not in df.columns:
        df["runs_total"] = df["runs_batter"]
    df["runs_total"] = pd.to_numeric(df["runs_total"], errors="coerce").fillna(df["runs_batter"])

    # Module 4: correct dot ball logic
    df["is_dot_ball"] = ((df["runs_total"] == 0) & (df["valid_ball"] == 1)).astype(int)

    # Date normalisation (Module 1)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT

    # Batting position (Module 6)
    if "bat_pos" in df.columns:
        df["bat_pos"] = pd.to_numeric(df["bat_pos"], errors="coerce").fillna(5)
    else:
        df["bat_pos"] = 5

    # Pressure / chase flag (Module 7)
    if "runs_target" in df.columns:
        df["runs_target"]   = pd.to_numeric(df["runs_target"], errors="coerce").fillna(0)
        df["pressure_flag"] = (df["runs_target"] > 0).astype(int)
    else:
        df["runs_target"]   = 0
        df["pressure_flag"] = 0

    # Venue columns (Module 13)
    for vc in ["venue", "city"]:
        if vc not in df.columns:
            df[vc] = "Unknown"

    return df


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # match_id fix
    if "id" in df.columns and "match_id" not in df.columns:
        df = df.rename(columns={"id": "match_id"})

    df["match_id"] = df["match_id"].astype(str)

    # 🔥 CRITICAL FIX (your error)
    if "winning_team" not in df.columns:
        if "match_won_by" in df.columns:
            df["winning_team"] = df["match_won_by"]
        else:
            df["winning_team"] = "Unknown"

    return df


# ---------------------------------------------------------------------------
# RECENCY HELPERS  (Modules 1 & 2)
# ---------------------------------------------------------------------------

def _compute_match_weights(match_dates: pd.Series, lam: float = RECENCY_LAMBDA) -> pd.Series:
    """Return exp(-λ × days_since_match) weights aligned to match_dates index."""
    max_date = match_dates.max()
    days_since = (max_date - match_dates).dt.days.fillna(0).clip(lower=0)
    return np.exp(-lam * days_since)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.reindex(values.index).fillna(0)
    total_w = w.sum()
    if total_w == 0:
        return 0.0
    return float((values * w).sum() / total_w)


def _get_last_n_matches(player_data: pd.DataFrame, date_col: str = "date", n: int = LAST_N_MATCHES) -> pd.DataFrame:
    """Keep only the last N distinct match dates for a player's rows."""
    if date_col not in player_data.columns or player_data[date_col].isna().all():
        return player_data
    sorted_dates = player_data[date_col].dropna().sort_values().unique()
    if len(sorted_dates) <= n:
        return player_data
    cutoff = sorted_dates[-n]
    return player_data[player_data[date_col] >= cutoff]


# ---------------------------------------------------------------------------
# BATTING POSITION WEIGHT  (Module 6)
# ---------------------------------------------------------------------------

def _position_weight(bat_pos: float) -> float:
    if bat_pos <= 3:
        return BATTING_POSITION_WEIGHTS["top"]
    if bat_pos <= 6:
        return BATTING_POSITION_WEIGHTS["middle"]
    return BATTING_POSITION_WEIGHTS["lower"]


# ---------------------------------------------------------------------------
# VENUE RUN FACTOR  (Module 13)
# ---------------------------------------------------------------------------

def compute_venue_run_factor(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average runs_total per over per venue as a venue adjustment factor.
    Returns DataFrame: venue, venue_run_factor
    """
    if "venue" not in bbb.columns:
        return pd.DataFrame(columns=["venue", "venue_run_factor"])

    per_match = (
        bbb.groupby(["venue", "match_id"])["runs_total"]
        .sum()
        .reset_index(name="match_runs")
    )
    venue_avg = (
        per_match.groupby("venue")["match_runs"]
        .mean()
        .reset_index(name="venue_avg_runs")
    )
    global_mean = venue_avg["venue_avg_runs"].mean()
    if global_mean == 0:
        venue_avg["venue_run_factor"] = 1.0
    else:
        venue_avg["venue_run_factor"] = venue_avg["venue_avg_runs"] / global_mean

    return venue_avg[["venue", "venue_run_factor"]]


# ---------------------------------------------------------------------------
# DATA QUALITY / COVERAGE TABLE  (Module 12)
# ---------------------------------------------------------------------------

def compute_player_coverage(
    squads: dict[str, list[str]],
    player_ratings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Optional diagnostic: returns player coverage per team.

    Parameters
    ----------
    squads          : { team_name: [player, ...] }
    player_ratings  : output of compute_player_ratings()

    Returns
    -------
    DataFrame: team, players_matched, total_players, coverage_ratio
    """
    known = set(player_ratings["player"].tolist())
    rows = []
    for team, squad in squads.items():
        total   = len(squad)
        matched = sum(1 for p in squad if p in known)
        rows.append({
            "team":            team,
            "players_matched": matched,
            "total_players":   total,
            "coverage_ratio":  matched / total if total > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# STAGE 1A — player_match_stats
# ---------------------------------------------------------------------------

def compute_player_match_stats(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    TABLE: player_match_stats
    One row per (player, match_id, role).

    Columns
    -------
    player, match_id, date,
    runs_scored, balls_faced, fours, sixes, strike_rate,
    wickets_taken, balls_bowled, runs_conceded, economy,
    dot_balls, boundary_balls,
    dismissals_involved,
    phase_pp_runs, phase_mid_runs, phase_death_runs,
    phase_pp_balls, phase_mid_balls, phase_death_balls,
    avg_bat_pos, position_weight,
    pressure_runs, pressure_balls, pressure_wickets,
    recency_weight
    """
    rows: list[dict] = []

    # Build per-match date map
    if "date" in bbb.columns:
        match_date_map = bbb.groupby("match_id")["date"].first()
        max_date = bbb["date"].max()
    else:
        match_date_map = pd.Series(dtype="datetime64[ns]")
        max_date = pd.NaT

    for match_id, grp in bbb.groupby("match_id"):

        match_date = match_date_map.get(match_id, pd.NaT)

        # Per-match recency weight (scalar, applied later in aggregate)
        if pd.notna(match_date) and pd.notna(max_date):
            days_since      = (max_date - match_date).days
            match_rec_weight = float(np.exp(-RECENCY_LAMBDA * max(0, days_since)))
        else:
            match_rec_weight = 1.0

        # ---- BATTING -------------------------------------------------------
        bat_grp = grp.groupby("batter")

        bat_stats = bat_grp.agg(
            runs_scored    = ("runs_batter",  "sum"),
            balls_faced    = ("valid_ball",   "sum"),
            boundary_balls = ("is_boundary",  "sum"),
            avg_bat_pos    = ("bat_pos",      "mean"),
        ).reset_index().rename(columns={"batter": "player"})

        # fours / sixes
        fours = grp[grp["runs_batter"] == 4].groupby("batter").size().rename("fours")
        sixes = grp[grp["runs_batter"] == 6].groupby("batter").size().rename("sixes")
        bat_stats = (bat_stats
                     .join(fours, on="player")
                     .join(sixes, on="player"))
        bat_stats["fours"] = bat_stats["fours"].fillna(0)
        bat_stats["sixes"] = bat_stats["sixes"].fillna(0)

        bat_stats["strike_rate"] = (bat_stats["runs_scored"] / bat_stats["balls_faced"].replace(0, np.nan)) * 100

        # Module 6: position weight
        bat_stats["position_weight"] = bat_stats["avg_bat_pos"].apply(_position_weight)
        bat_stats["adjusted_runs"]   = bat_stats["runs_scored"] * bat_stats["position_weight"]

        # Phase splits — batting
        def _phase_bat(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            return ph.groupby("batter").agg(
                runs  = ("runs_batter", "sum"),
                balls = ("valid_ball",  "sum"),
            )

        pp_bat  = _phase_bat(POWERPLAY_OVERS).rename(columns={"runs": "phase_pp_runs",    "balls": "phase_pp_balls"})
        mid_bat = _phase_bat(MIDDLE_OVERS).rename(   columns={"runs": "phase_mid_runs",   "balls": "phase_mid_balls"})
        dth_bat = _phase_bat(DEATH_OVERS).rename(    columns={"runs": "phase_death_runs", "balls": "phase_death_balls"})

        bat_stats = (bat_stats
                     .merge(pp_bat.reset_index().rename( columns={"batter": "player"}), on="player", how="left")
                     .merge(mid_bat.reset_index().rename(columns={"batter": "player"}), on="player", how="left")
                     .merge(dth_bat.reset_index().rename(columns={"batter": "player"}), on="player", how="left"))

        # Module 7: pressure batting stats
        pressure_bat = grp[grp["pressure_flag"] == 1].groupby("batter").agg(
            pressure_runs  = ("runs_batter", "sum"),
            pressure_balls = ("valid_ball",  "sum"),
        ).reset_index().rename(columns={"batter": "player"})

        bat_stats = bat_stats.merge(pressure_bat, on="player", how="left")

        bat_stats["match_id"]        = match_id
        bat_stats["date"]            = match_date
        bat_stats["role"]            = "bat"
        bat_stats["recency_weight"]  = match_rec_weight

        # ---- BOWLING -------------------------------------------------------
        bowl_grp = grp.groupby("bowler")

        bowl_stats = bowl_grp.agg(
            balls_bowled  = ("valid_ball",        "sum"),
            runs_conceded = ("runs_total",         "sum"),
            wickets_taken = ("iswicket_delivery", "sum"),
            dot_balls     = ("is_dot_ball",        "sum"),
        ).reset_index().rename(columns={"bowler": "player"})

        bowl_stats["economy"] = (
            bowl_stats["runs_conceded"] / (bowl_stats["balls_bowled"] / 6).replace(0, np.nan)
        )

        # Phase bowling economy
        def _phase_bowl_eco(phase_range):
            ph = grp[grp["over"].isin(phase_range)]
            g  = ph.groupby("bowler").agg(
                runs  = ("runs_total", "sum"),
                balls = ("valid_ball", "sum"),
            )
            g["eco"] = g["runs"] / (g["balls"] / 6).replace(0, np.nan)
            return g[["eco"]]

        pp_eco  = _phase_bowl_eco(POWERPLAY_OVERS).rename( columns={"eco": "phase_pp_eco"})
        mid_eco = _phase_bowl_eco(MIDDLE_OVERS).rename(    columns={"eco": "phase_mid_eco"})
        dth_eco = _phase_bowl_eco(DEATH_OVERS).rename(     columns={"eco": "phase_death_eco"})

        bowl_stats = (bowl_stats
                      .merge(pp_eco.reset_index().rename( columns={"bowler": "player"}), on="player", how="left")
                      .merge(mid_eco.reset_index().rename(columns={"bowler": "player"}), on="player", how="left")
                      .merge(dth_eco.reset_index().rename(columns={"bowler": "player"}), on="player", how="left"))

        # Module 7: pressure bowling
        pressure_bowl = grp[grp["pressure_flag"] == 1].groupby("bowler").agg(
            pressure_wickets = ("iswicket_delivery", "sum"),
        ).reset_index().rename(columns={"bowler": "player"})
        bowl_stats = bowl_stats.merge(pressure_bowl, on="player", how="left")

        # Dismissals involved
        dismissals = (grp[grp["iswicket_delivery"] == 1]
                      .groupby("bowler").size()
                      .reset_index(name="dismissals_involved")
                      .rename(columns={"bowler": "player"}))
        bowl_stats = bowl_stats.merge(dismissals, on="player", how="left")
        bowl_stats["dismissals_involved"] = bowl_stats["dismissals_involved"].fillna(0)

        bowl_stats["match_id"]       = match_id
        bowl_stats["date"]           = match_date
        bowl_stats["role"]           = "bowl"
        bowl_stats["recency_weight"] = match_rec_weight

        rows.append(bat_stats)
        rows.append(bowl_stats)

    out = pd.concat(rows, ignore_index=True)

    fill_zero_cols = [
        "phase_pp_runs",  "phase_pp_balls",
        "phase_mid_runs", "phase_mid_balls",
        "phase_death_runs","phase_death_balls",
        "phase_pp_eco",   "phase_mid_eco",   "phase_death_eco",
        "pressure_runs",  "pressure_balls",  "pressure_wickets",
        "adjusted_runs",  "position_weight", "avg_bat_pos",
    ]
    for c in fill_zero_cols:
        if c not in out.columns:
            out[c] = 0
        else:
            out[c] = out[c].fillna(0)

    return out


# ---------------------------------------------------------------------------
# STAGE 1B — player_aggregate_stats
# ---------------------------------------------------------------------------

def _weighted_agg_batting(bat_recent: pd.DataFrame) -> dict:
    """
    Compute weighted batting aggregate for a single player's recent matches.
    Returns a dict of aggregated metrics.
    """
    w = bat_recent["recency_weight"].values
    w_sum = w.sum()
    if w_sum == 0:
        w_sum = 1.0

    def wm(col):
        return float((bat_recent[col].values * w).sum() / w_sum)

    total_balls  = float(bat_recent["balls_faced"].sum())
    total_runs   = float(bat_recent["runs_scored"].sum())
    total_bounds = float(bat_recent["boundary_balls"].sum())
    innings      = len(bat_recent)

    avg_runs     = wm("runs_scored")
    # Weighted strike rate
    w_sr         = wm("strike_rate") if "strike_rate" in bat_recent.columns else (total_runs / max(total_balls, 1) * 100)

    # Coefficient of variation for consistency (Module 14)
    sr_vals = bat_recent["strike_rate"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_sr = sr_vals.mean()
    std_sr  = sr_vals.std()
    cv_sr   = (std_sr / mean_sr) if (mean_sr and mean_sr != 0) else 0.0

    return {
        "avg_runs":          avg_runs,
        "strike_rate":       w_sr,
        "boundary_rate":     total_bounds / max(total_balls, 1),
        "consistency_score": float(cv_sr),
        "total_balls":       total_balls,
        "innings_batted":    innings,
        "pp_runs":           float(bat_recent["phase_pp_runs"].sum()),
        "pp_balls":          float(bat_recent["phase_pp_balls"].sum()),
        "mid_runs":          float(bat_recent["phase_mid_runs"].sum()),
        "mid_balls":         float(bat_recent["phase_mid_balls"].sum()),
        "dth_runs":          float(bat_recent["phase_death_runs"].sum()),
        "dth_balls":         float(bat_recent["phase_death_balls"].sum()),
        "pressure_runs":     float(bat_recent["pressure_runs"].sum()),
        "pressure_balls":    float(bat_recent["pressure_balls"].sum()),
        "avg_position_weight": float(bat_recent["position_weight"].mean()) if "position_weight" in bat_recent.columns else 1.0,
        "experience_score_bat": innings,  # raw; normalized later
    }


def _weighted_agg_bowling(bowl_recent: pd.DataFrame) -> dict:
    w = bowl_recent["recency_weight"].values
    w_sum = w.sum()
    if w_sum == 0:
        w_sum = 1.0

    def wm(col):
        return float((bowl_recent[col].values * w).sum() / w_sum)

    total_balls   = float(bowl_recent["balls_bowled"].sum())
    total_runs    = float(bowl_recent["runs_conceded"].sum())
    total_wickets = float(bowl_recent["wickets_taken"].sum())
    total_dots    = float(bowl_recent["dot_balls"].sum())
    innings       = len(bowl_recent)

    overs = total_balls / 6 if total_balls > 0 else 1.0
    eco   = total_runs / overs

    # Weighted economy
    w_eco = wm("economy") if "economy" in bowl_recent.columns else eco

    # CV for bowling consistency (Module 14)
    eco_vals = bowl_recent["economy"].replace([np.inf, -np.inf], np.nan).dropna()
    mean_eco = eco_vals.mean()
    std_eco  = eco_vals.std()
    cv_eco   = (std_eco / mean_eco) if (mean_eco and mean_eco != 0) else 0.0

    return {
        "wicket_rate":       total_wickets / max(total_balls, 1),
        "economy":           w_eco,
        "dot_ball_rate":     total_dots / max(total_balls, 1),
        "consistency_bowl":  float(cv_eco),
        "total_balls_bowled": total_balls,
        "innings_bowled":    innings,
        "phase_pp_eco":      float(bowl_recent["phase_pp_eco"].replace(0, np.nan).mean()) if "phase_pp_eco" in bowl_recent.columns else 0.0,
        "phase_mid_eco":     float(bowl_recent["phase_mid_eco"].replace(0, np.nan).mean()) if "phase_mid_eco" in bowl_recent.columns else 0.0,
        "phase_death_eco":   float(bowl_recent["phase_death_eco"].replace(0, np.nan).mean()) if "phase_death_eco" in bowl_recent.columns else 0.0,
        "pressure_wickets":  float(bowl_recent["pressure_wickets"].sum()) if "pressure_wickets" in bowl_recent.columns else 0.0,
        "experience_score_bowl": innings,
    }


def compute_player_aggregate_stats(pms: pd.DataFrame) -> pd.DataFrame:
    """
    TABLE: player_aggregate_stats
    One row per player. Recency-weighted form + career hybrid.

    Columns (all original + new)
    ----------------------------
    player,
    avg_runs, strike_rate, boundary_rate, consistency_score,
    wicket_rate, economy, dot_ball_rate,
    phase_pp_sr, phase_mid_sr, phase_death_sr,
    phase_pp_eco, phase_mid_eco, phase_death_eco,
    innings_batted, innings_bowled,
    total_balls, total_balls_bowled,
    pressure_strike_rate, pressure_consistency,
    experience_score,
    avg_position_weight,
    form_batting_score_raw, form_bowling_score_raw,
    career_batting_score_raw, career_bowling_score_raw,
    """

    bat_all  = pms[pms["role"] == "bat"].copy()
    bowl_all = pms[pms["role"] == "bowl"].copy()

    # Ensure date column exists
    if "date" not in bat_all.columns:
        bat_all["date"] = pd.NaT
    if "date" not in bowl_all.columns:
        bowl_all["date"] = pd.NaT

    players = pms["player"].unique()
    records = []

    for player in players:
        pb = bat_all[bat_all["player"] == player].copy()
        pw = bowl_all[bowl_all["player"] == player].copy()

        # Last-N filter (Module 2)
        pb_recent = _get_last_n_matches(pb) if len(pb) > 0 else pb
        pw_recent = _get_last_n_matches(pw) if len(pw) > 0 else pw

        # ---------- BATTING ------------------------------------------------
        if len(pb) >= 1:
            # Recent (form) weighted agg
            form_bat   = _weighted_agg_batting(pb_recent) if len(pb_recent) >= 1 else _weighted_agg_batting(pb)
            # Career agg (no recency filter)
            career_bat = _weighted_agg_batting(pb)

            # Module 10: hybrid score
            avg_runs_final       = FORM_CAREER_WEIGHT_RECENT * form_bat["avg_runs"]       + FORM_CAREER_WEIGHT_CAREER * career_bat["avg_runs"]
            sr_final             = FORM_CAREER_WEIGHT_RECENT * form_bat["strike_rate"]    + FORM_CAREER_WEIGHT_CAREER * career_bat["strike_rate"]
            br_final             = FORM_CAREER_WEIGHT_RECENT * form_bat["boundary_rate"]  + FORM_CAREER_WEIGHT_CAREER * career_bat["boundary_rate"]
            cs_final             = FORM_CAREER_WEIGHT_RECENT * form_bat["consistency_score"] + FORM_CAREER_WEIGHT_CAREER * career_bat["consistency_score"]

            # Phase SRs from career totals (more stable)
            pp_sr  = (career_bat["pp_runs"]  / max(career_bat["pp_balls"],  1)) * 100
            mid_sr = (career_bat["mid_runs"] / max(career_bat["mid_balls"], 1)) * 100
            dth_sr = (career_bat["dth_runs"] / max(career_bat["dth_balls"], 1)) * 100

            # Pressure (Module 7)
            p_balls = career_bat["pressure_balls"]
            p_runs  = career_bat["pressure_runs"]
            pressure_sr = (p_runs / p_balls * 100) if p_balls > 0 else sr_final
            # Pressure consistency: compare pressure SR to overall SR
            pressure_cons = abs(pressure_sr - sr_final) / max(sr_final, 1.0)

            bat_rec = {
                "avg_runs":               avg_runs_final,
                "strike_rate":            sr_final,
                "boundary_rate":          br_final,
                "consistency_score":      cs_final,
                "total_balls":            career_bat["total_balls"],
                "innings_batted":         career_bat["innings_batted"],
                "phase_pp_sr":            pp_sr,
                "phase_mid_sr":           mid_sr,
                "phase_death_sr":         dth_sr,
                "pressure_strike_rate":   pressure_sr,
                "pressure_consistency":   pressure_cons,
                "avg_position_weight":    career_bat["avg_position_weight"],
                "experience_score_bat":   career_bat["experience_score_bat"],
            }
        else:
            bat_rec = {k: 0.0 for k in [
                "avg_runs","strike_rate","boundary_rate","consistency_score",
                "total_balls","innings_batted","phase_pp_sr","phase_mid_sr","phase_death_sr",
                "pressure_strike_rate","pressure_consistency","avg_position_weight","experience_score_bat",
            ]}

        # ---------- BOWLING ------------------------------------------------
        if len(pw) >= 1:
            form_bowl   = _weighted_agg_bowling(pw_recent) if len(pw_recent) >= 1 else _weighted_agg_bowling(pw)
            career_bowl = _weighted_agg_bowling(pw)

            wr_final   = FORM_CAREER_WEIGHT_RECENT * form_bowl["wicket_rate"]   + FORM_CAREER_WEIGHT_CAREER * career_bowl["wicket_rate"]
            eco_final  = FORM_CAREER_WEIGHT_RECENT * form_bowl["economy"]       + FORM_CAREER_WEIGHT_CAREER * career_bowl["economy"]
            dbr_final  = FORM_CAREER_WEIGHT_RECENT * form_bowl["dot_ball_rate"] + FORM_CAREER_WEIGHT_CAREER * career_bowl["dot_ball_rate"]

            bowl_rec = {
                "wicket_rate":          wr_final,
                "economy":              eco_final,
                "dot_ball_rate":        dbr_final,
                "total_balls_bowled":   career_bowl["total_balls_bowled"],
                "innings_bowled":       career_bowl["innings_bowled"],
                "phase_pp_eco":         career_bowl["phase_pp_eco"],
                "phase_mid_eco":        career_bowl["phase_mid_eco"],
                "phase_death_eco":      career_bowl["phase_death_eco"],
                "pressure_wickets":     career_bowl["pressure_wickets"],
                "experience_score_bowl": career_bowl["experience_score_bowl"],
            }
        else:
            bowl_rec = {k: 0.0 for k in [
                "wicket_rate","economy","dot_ball_rate","total_balls_bowled","innings_bowled",
                "phase_pp_eco","phase_mid_eco","phase_death_eco","pressure_wickets","experience_score_bowl",
            ]}

        records.append({"player": player, **bat_rec, **bowl_rec})

    agg = pd.DataFrame(records)

    # Module 9: normalize experience score ∈ [0,1]
    for exp_col in ["experience_score_bat", "experience_score_bowl"]:
        if exp_col in agg.columns:
            lo, hi = agg[exp_col].min(), agg[exp_col].max()
            if hi > lo:
                agg[exp_col] = (agg[exp_col] - lo) / (hi - lo)
            else:
                agg[exp_col] = 0.5
    agg["experience_score"] = (agg.get("experience_score_bat", 0) + agg.get("experience_score_bowl", 0)) / 2

    # Module 15: clip outliers before delivering to ratings
    for col in ["strike_rate", "economy", "avg_runs", "wicket_rate", "dot_ball_rate", "boundary_rate"]:
        if col in agg.columns:
            lo = agg[col].quantile(0.01)
            hi = agg[col].quantile(0.99)
            agg[col] = agg[col].clip(lo, hi)

    keep = [
        "player",
        "avg_runs", "strike_rate", "boundary_rate", "consistency_score",
        "wicket_rate", "economy", "dot_ball_rate",
        "phase_pp_sr", "phase_mid_sr", "phase_death_sr",
        "phase_pp_eco", "phase_mid_eco", "phase_death_eco",
        "innings_batted", "innings_bowled",
        "total_balls", "total_balls_bowled",
        "pressure_strike_rate", "pressure_consistency",
        "experience_score",
        "avg_position_weight",
    ]

    for c in keep:
        if c not in agg.columns:
            agg[c] = 0.0

    return agg[keep].fillna(0)