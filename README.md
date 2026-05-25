# IPL Prediction API — Run Instructions

## Setup

```bash
pip install -r requirements.txt
```

Place `engine.py` and the `ipl_engine/` package in the same directory as `main.py`.

## Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

API available at: http://localhost:8000
Swagger docs at:  http://localhost:8000/docs

## Usage Order

1. POST /load-data        — must be called first
2. POST /team-strength    — per-team query
3. POST /predict-match    — single match prediction
4. POST /simulate-season  — Monte Carlo season simulation

## Example Requests

### 1. Load data
```bash
curl -X POST http://localhost:8000/load-data \
  -H "Content-Type: application/json" \
  -d '{"bbb_path": "ball_by_ball.csv", "matches_path": "matches.csv"}'
```

### 2. Team strength
```bash
curl -X POST http://localhost:8000/team-strength \
  -H "Content-Type: application/json" \
  -d '{"team": "Mumbai Indians", "squad": ["JJ Bumrah", "RG Sharma"]}'
```

### 3. Predict match
```bash
curl -X POST http://localhost:8000/predict-match \
  -H "Content-Type: application/json" \
  -d '{
    "teamA": "Mumbai Indians",
    "teamB": "Chennai Super Kings",
    "squads": {
      "Mumbai Indians": ["JJ Bumrah", "RG Sharma"],
      "Chennai Super Kings": ["MS Dhoni", "RA Jadeja"]
    },
    "venue": "Wankhede Stadium",
    "toss_winner": "Mumbai Indians"
  }'
```

### 4. Simulate season
```bash
curl -X POST http://localhost:8000/simulate-season \
  -H "Content-Type: application/json" \
  -d '{
    "fixtures": [
      {"match_id": "1", "teamA": "Mumbai Indians", "teamB": "Chennai Super Kings", "venue": "Wankhede Stadium"}
    ],
    "squads": {
      "Mumbai Indians": ["JJ Bumrah", "RG Sharma"],
      "Chennai Super Kings": ["MS Dhoni", "RA Jadeja"]
    },
    "simulations": 10000
  }'
```

### 5. Health check
```bash
curl http://localhost:8000/health
```
