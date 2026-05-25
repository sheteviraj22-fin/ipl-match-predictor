# IPL Prediction Backend — Architecture & API Reference

## Directory Structure

```
ipl_engine/
├── ipl_engine/
│   ├── __init__.py     # public re-exports
│   ├── pipeline.py     # Stage 1: raw data → feature tables
│   ├── ratings.py      # Stage 2: feature tables → normalized scores
│   ├── team_model.py   # Stage 3: squad → TeamStrength + venue/toss context
│   ├── predictor.py    # Stage 4+5: match prediction + Monte Carlo
│   └── engine.py       # Public API facade (6 backend functions)
└── tests/
    └── test_engine.py  # 16-test integration suite (all green)
```

---

## Data Flow

```
Ball-by-ball CSV
Match CSV
        │
        ▼
  pipeline.py
  ─────────────────────────────────────────────
  load_ball_by_ball()     → canonical BBB frame
  load_matches()          → canonical matches frame
  compute_player_match_stats()  → player_match_stats
  compute_player_aggregate_stats() → player_aggregate_stats
        │
        ▼
  ratings.py
  ─────────────────────────────────────────────
  compute_player_ratings()  → PlayerRating table
        │
        ▼
  team_model.py
  ─────────────────────────────────────────────
  compute_team_strength()   → TeamStrength dict
  compute_venue_bias()      → venue_bias_table
  compute_toss_adjustment() → float scalar
        │
        ▼
  predictor.py
  ─────────────────────────────────────────────
  predict_match()    → MatchPrediction dict
  simulate_season()  → [SimulationResult, ...]
```

---

## Stage 1 — Feature Tables

### player_match_stats

| Column | Type | Formula |
|--------|------|---------|
| player | str | batter / bowler identifier |
| match_id | str | raw match identifier |
| role | str | `"bat"` or `"bowl"` |
| runs_scored | int | Σ batsman_run |
| balls_faced | int | count(deliveries) |
| fours | int | count(batsman_run == 4) |
| sixes | int | count(batsman_run == 6) |
| boundary_balls | int | fours + sixes |
| strike_rate | float | (runs_scored / balls_faced) × 100 |
| wickets_taken | int | Σ iswicket_delivery |
| balls_bowled | int | count(deliveries) |
| runs_conceded | int | Σ total_run |
| economy | float | runs_conceded / (balls_bowled / 6) |
| dot_balls | int | count(batsman_run == 0) |
| dismissals_involved | int | count where bowler credited with wicket |
| phase_pp_runs/balls | int | over ∈ [0,5] |
| phase_mid_runs/balls | int | over ∈ [6,14] |
| phase_death_runs/balls | int | over ∈ [15,19] |

### player_aggregate_stats

| Column | Type | Formula |
|--------|------|---------|
| avg_runs | float | total_runs / innings_batted |
| strike_rate | float | (total_runs / total_balls) × 100 |
| boundary_rate | float | total_boundaries / total_balls |
| consistency_score | float | std(strike_rate across matches) — lower = consistent |
| wicket_rate | float | total_wickets / total_balls_bowled |
| economy | float | total_runs_conceded / (total_balls_bowled / 6) |
| dot_ball_rate | float | total_dots / total_balls_bowled |
| phase_pp/mid/death_sr | float | phase_runs / phase_balls × 100 |

---

## Stage 2 — Player Rating Formulas

All sub-scores are **min-max normalized** over the qualified player pool before weighting.

### Qualification Thresholds
- Batting: ≥ 30 balls faced (career total)
- Bowling: ≥ 60 balls bowled (career total)

### BattingScore
```
BattingScore =
    0.35 × norm(strike_rate)
  + 0.25 × norm(avg_runs)
  + 0.20 × norm(boundary_rate)
  + 0.20 × norm(max(consistency_score) − consistency_score)  ← inverse
```

### BowlingScore
```
BowlingScore =
    0.40 × norm(wicket_rate)
  + 0.30 × norm(max(economy) − economy)   ← inverse
  + 0.30 × norm(dot_ball_rate)
```

### AllRounderScore
```
AllRounderScore = 0.50 × BattingScore + 0.50 × BowlingScore
                  (only if qualified for BOTH batting AND bowling)
```

---

## Stage 3 — Team Strength

```
BattingUnit       = mean(top-7 BattingScore from squad)
BowlingUnit       = mean(top-5 BowlingScore from squad)
AllRounderBalance = mean(AllRounderScore of qualified all-rounders in squad)

TeamStrength = 0.45 × BattingUnit
             + 0.40 × BowlingUnit
             + 0.15 × AllRounderBalance
```

### VenueBiasFactor
```
venue_bias_factor = P(toss_winner_wins | venue) − 0.5
Range: [−0.5, +0.5]
Shrunk to 0 if sample_size < 5.
```

### TossAdjustment
```
toss_adjustment = P(toss_winner_wins, global) − 0.5
Applied as ±adjustment depending on which team won the toss.
```

---

## Stage 4 — Match Prediction

```
BaseDiff   = TeamStrength_A − TeamStrength_B
VenueAdj   = venue_bias_factor[venue]  (0 if unknown)
TossAdj    = +toss_adjustment  if toss_winner == A
           = −toss_adjustment  if toss_winner == B
           =  0                if no toss info

FinalScore = BaseDiff + VenueAdj + TossAdj

WinProbability_A = 1 / (1 + exp(−k × FinalScore))
WinProbability_B = 1 − WinProbability_A
```

**k (scaling constant):** default = 10.0. Tune via log-loss minimization on held-out seasons. k=10 maps a strength gap of 0.10 → ~73% win probability.

---

## Stage 5 — Monte Carlo Simulation

```
FOR i in range(N):
    points = {team: 0}
    FOR each fixture (A vs B):
        p_A = WinProbability_A (precomputed)
        winner = A if random() < p_A else B
        points[winner] += 2
    RANK teams by points
    title_count[rank[0]] += 1
    playoff_count[rank[1:4]] += 1

title_prob[team]   = title_count[team] / N
playoff_prob[team] = playoff_count[team] / N
avg_points[team]   = Σ points[team] / N
```

---

## JSON Data Contracts

### PlayerRating
```json
{
  "player":                 "V Kohli",
  "batting_score":          0.872341,
  "bowling_score":          0.031200,
  "allrounder_score":       0.0,
  "is_batting_qualified":   true,
  "is_bowling_qualified":   false,
  "is_allrounder":          false
}
```

### TeamStrength
```json
{
  "team":               "Mumbai Indians",
  "batting_unit":       0.743210,
  "bowling_unit":       0.681450,
  "allrounder_balance": 0.512300,
  "total_strength":     0.697834,
  "squad_size":         15,
  "squad_matched":      13
}
```

### MatchPrediction
```json
{
  "teamA":             "Mumbai Indians",
  "teamB":             "Chennai Super Kings",
  "win_probability_A": 0.613200,
  "win_probability_B": 0.386800,
  "base_diff":         0.041200,
  "venue_adjustment":  0.012000,
  "toss_adjustment":   0.008000,
  "final_score":       0.061200,
  "venue":             "Wankhede Stadium",
  "toss_winner":       "Mumbai Indians"
}
```

### SimulationResult
```json
{
  "team":         "Mumbai Indians",
  "title_prob":   0.243100,
  "playoff_prob": 0.718400,
  "avg_points":   14.2300,
  "avg_wins":     7.1150
}
```

---

## Public API — 6 Backend Functions

```python
from ipl_engine.engine import (
    compute_player_stats,     # raw CSV → (player_match_stats, player_aggregate_stats)
    compute_player_ratings,   # agg_stats → PlayerRating DataFrame
    compute_team_strength,    # squad + ratings → TeamStrength dict
    predict_match,            # teamA, teamB, context → MatchPrediction dict
    simulate_season,          # fixtures + squads → [SimulationResult]
    build_context_tables,     # matches CSV → {venue_bias_table, toss_adjustment}
)
```

### Minimal Usage

```python
from ipl_engine.engine import *

# Stage 1
pms, agg = compute_player_stats("ball_by_ball.csv")

# Stage 2
ratings = compute_player_ratings(agg)

# Stage 3
ctx = build_context_tables("matches.csv")

squads = {
    "Mumbai Indians":      ["JJ Bumrah", "RG Sharma", ...],
    "Chennai Super Kings": ["MS Dhoni",  "RA Jadeja", ...],
}

# Stage 4
pred = predict_match(
    teamA="Mumbai Indians",
    teamB="Chennai Super Kings",
    squads=squads,
    player_ratings=ratings,
    context={"venue": "Wankhede Stadium", "toss_winner": "Mumbai Indians"},
    **ctx,
)

# Stage 5
fixtures = [{"match_id": "1", "teamA": "Mumbai Indians",
             "teamB": "Chennai Super Kings", "venue": "Wankhede Stadium"}]

results = simulate_season(
    fixtures=fixtures, squads=squads,
    player_ratings=ratings, N=10_000, **ctx,
)
```

---

## Tunable Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k` | 10.0 | Logistic steepness; higher → more decisive wins |
| `MIN_BALLS_FACED` | 30 | Batting qualification cutoff |
| `MIN_BALLS_BOWLED` | 60 | Bowling qualification cutoff |
| `TOP_BATTERS` | 7 | Squad selection for BattingUnit |
| `TOP_BOWLERS` | 5 | Squad selection for BowlingUnit |
| `ALLROUNDER_BATTING/BOWLING_WEIGHT` | 0.50/0.50 | AR score split |
| `PLAYOFF_SPOTS` | 4 | Teams qualifying for playoffs |

---

## Test Coverage

16 integration tests across all 5 stages:

| Stage | Tests |
|-------|-------|
| Pipeline schema | 3 |
| Ratings bounds + logic | 3 |
| Team strength | 3 |
| Match prediction | 3 |
| Simulation | 3 |
| JSON serialisability | 1 |

All 16 pass on real IPL 2008–2022 data (225,954 deliveries, 950 matches).
