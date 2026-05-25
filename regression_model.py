"""
regression_model.py  (v8)
────────────────────────────────────────────────────────────────────────────────
Changes vs v7:
  • Replaced XGBoost with LogisticRegression (C=0.5)
  • Comprehensive feature set (10 feature groups, walk-forward safe)
  • Multicollinearity control: drops features with pairwise corr > 0.80
  • StandardScaler on all numeric features
  • Playing-XI-based player quality, form, experience, depth, balance
  • Venue features: avg runs, wickets, type, team win-rate diff
  • Match context: home advantage, chasing advantage, toss
  • Controlled interaction terms (batting×venue, bowling×venue, batting×chasing)
────────────────────────────────────────────────────────────────────────────────
Paths (hardcoded):
  MATCHES_PATH    → dataset/matches.csv
  BBB_PATH        → dataset/IPL.csv
  PLAYING_XI_PATH → dataset/playing_xi.json
  HOME_PATH       → dataset/IPL_Home_Ground_Table.csv
────────────────────────────────────────────────────────────────────────────────
"""
from sklearn.calibration import CalibratedClassifierCV
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

MATCHES_PATH    = "dataset/matches.csv"
BBB_PATH        = "dataset/IPL.csv"
PLAYING_XI_PATH = "dataset/playing_xi.json"
HOME_PATH       = "dataset/IPL_Home_Ground_Table.csv"
TRAIN_RATIO     = 0.80

NON_BOWLER_WICKETS = {
    "run out", "retired hurt", "retired out",
    "obstructing the field", "handled the ball", "timed out",
}
MIN_BALLS_BAT       = 30
MIN_BALLS_BOWL      = 24
BATTING_THRESHOLD   = 1.25
BOWLING_THRESHOLD   = 1.40
STAR_PERCENTILE     = 0.90
FORM_WINDOW         = 5
PREVIOUS_OOS_ACC    = 0.5256
CORR_DROP_THRESHOLD = 0.80

VENUE_MAP: dict = {
    "Feroz Shah Kotla":                     "Arun Jaitley Stadium",
    "Arun Jaitley Stadium, Delhi":          "Arun Jaitley Stadium",
    "M.Chinnaswamy Stadium":                "M Chinnaswamy Stadium",
    "M Chinnaswamy Stadium, Bengaluru":     "M Chinnaswamy Stadium",
    "Eden Gardens, Kolkata":                "Eden Gardens",
    "Wankhede Stadium, Mumbai":             "Wankhede Stadium",
    "Brabourne Stadium, Mumbai":            "Brabourne Stadium",
    "Dr DY Patil Sports Academy, Mumbai":   "Dr DY Patil Sports Academy",
    "MA Chidambaram Stadium, Chepauk":              "MA Chidambaram Stadium",
    "MA Chidambaram Stadium, Chepauk, Chennai":     "MA Chidambaram Stadium",
    "Rajiv Gandhi International Stadium, Uppal":
        "Rajiv Gandhi International Stadium",
    "Rajiv Gandhi International Stadium, Uppal, Hyderabad":
        "Rajiv Gandhi International Stadium",
    "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam":
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh":
        "Punjab Cricket Association IS Bindra Stadium",
    "Punjab Cricket Association Stadium, Mohali":
        "Punjab Cricket Association IS Bindra Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, Mullanpur":
        "Maharaja Yadavindra Singh International Cricket Stadium",
    "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh":
        "Maharaja Yadavindra Singh International Cricket Stadium",
    "Sardar Patel Stadium, Motera":         "Narendra Modi Stadium, Ahmedabad",
    "Maharashtra Cricket Association Stadium, Pune":
        "Maharashtra Cricket Association Stadium",
    "Subrata Roy Sahara Stadium":
        "Maharashtra Cricket Association Stadium",
    "Sawai Mansingh Stadium, Jaipur":       "Sawai Mansingh Stadium",
    "Himachal Pradesh Cricket Association Stadium, Dharamsala":
        "Himachal Pradesh Cricket Association Stadium",
    "Zayed Cricket Stadium, Abu Dhabi":     "Sheikh Zayed Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow":
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
}


def normalize_venue(v: str) -> str:
    return VENUE_MAP.get(v.strip(), v.strip())


def get_season_keys(season: str) -> list:
    s = str(season).strip()
    keys = [s]
    try:
        y = int(s)
        keys.append(f"{y}/{str(y + 1)[-2:]}")
        keys.append(f"{y - 1}/{str(y)[-2:]}")
    except ValueError:
        pass
    return keys


TEAM_TO_ABBREV: dict = {
    "Chennai Super Kings":          "CSK",
    "Deccan Chargers":              "Deccan",
    "Delhi Capitals":               "DC",
    "Delhi Daredevils":             "DC",
    "Gujarat Lions":                "GL",
    "Gujarat Titans":               "GT",
    "Kings XI Punjab":              "PBKS",
    "Kochi Tuskers Kerala":         "KTK",
    "Kolkata Knight Riders":        "KKR",
    "Lucknow Super Giants":         "LSG",
    "Mumbai Indians":               "MI",
    "Pune Warriors":                "PWI",
    "Punjab Kings":                 "PBKS",
    "Rajasthan Royals":             "RR",
    "Rising Pune Supergiant":       "RPSG",
    "Rising Pune Supergiants":      "RPSG",
    "Royal Challengers Bangalore":  "RCB",
    "Royal Challengers Bengaluru":  "RCB",
    "Sunrisers Hyderabad":          "SRH",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_home_ground(path: str) -> dict:
    df = pd.read_csv(path)
    season_cols = [c for c in df.columns if c not in ("Row Labels", "Grand Total")]
    result: dict = {}
    for _, row in df.iterrows():
        norm_venue = normalize_venue(str(row["Row Labels"]))
        for season in season_cols:
            val = row[season]
            if pd.isna(val) or str(val).strip().upper() == "NEUTRAL":
                continue
            for abbrev in str(val).split(","):
                abbrev = abbrev.strip()
                if abbrev and abbrev != "NEUTRAL":
                    result.setdefault((str(season), norm_venue), abbrev)
    return result


def load_data(matches_path, bbb_path, playing_xi_path, home_path):
    matches = pd.read_csv(matches_path)
    bbb     = pd.read_csv(bbb_path, low_memory=False)
    with open(playing_xi_path) as fh:
        playing_xi = json.load(fh)
    home_ground_dict = load_home_ground(home_path)

    matches["date"] = pd.to_datetime(matches["date"], dayfirst=True)
    matches = matches.sort_values("date").reset_index(drop=True)

    print(f"  Matches loaded       : {len(matches):,}")
    print(f"  Deliveries           : {len(bbb):,}")
    print(f"  Playing XI entries   : {len(playing_xi)}")
    print(f"  Home-ground entries  : {len(home_ground_dict)}")
    return matches, bbb, playing_xi, home_ground_dict


def build_bbb_match_index(bbb: pd.DataFrame) -> dict:
    return {mid: grp for mid, grp in bbb.groupby("match_id")}


def build_toss_decision_index(bbb: pd.DataFrame) -> dict:
    cols = [c for c in ["match_id", "toss_decision", "toss_winner"] if c in bbb.columns]
    sub  = bbb[cols].drop_duplicates("match_id").set_index("match_id")
    return sub.to_dict("index")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CUMULATIVE STAT TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class CumulativeStats:
    def __init__(self):
        self.batting:        dict = {}
        self.bowling:        dict = {}
        self.matches_played: dict = {}
        self.recent_scores:  dict = {}
        self.venue_runs:     dict = {}
        self.venue_wickets:  dict = {}
        self.venue_team:     dict = {}

    def batting_sr(self, player):
        d = self.batting.get(player)
        if d and d["balls"] >= MIN_BALLS_BAT:
            return d["runs"] / d["balls"]
        return None

    def bowling_eco(self, player):
        d = self.bowling.get(player)
        if d and d["balls"] >= MIN_BALLS_BOWL:
            return d["runs"] / d["balls"]
        return None

    def player_form(self, player) -> float:
        scores = self.recent_scores.get(player)
        return float(np.mean(scores)) if scores else 0.0

    def player_experience(self, player) -> int:
        return self.matches_played.get(player, 0)

    def xi_top_batting(self, players, top_n: int = 5) -> float:
        vals = sorted(
            (self.batting_sr(p) for p in players if self.batting_sr(p) is not None),
            reverse=True
        )[:top_n]
        return float(np.mean(vals)) if vals else 0.0

    def xi_top_bowling(self, players, top_n: int = 4) -> float:
        vals = sorted(
            (self.bowling_eco(p) for p in players if self.bowling_eco(p) is not None)
        )[:top_n]
        return float(np.mean(vals)) if vals else 0.0

    def xi_allrounder_count(self, players) -> int:
        return sum(
            1 for p in players
            if self.batting_sr(p) is not None and self.bowling_eco(p) is not None
        )

    def xi_batting_depth(self, players) -> int:
        return sum(1 for p in players if (self.batting_sr(p) or 0.0) >= BATTING_THRESHOLD)

    def xi_bowling_depth(self, players) -> int:
        return sum(
            1 for p in players
            if self.bowling_eco(p) is not None and self.bowling_eco(p) <= BOWLING_THRESHOLD
        )

    def xi_star_count(self, players, bat_cut: float, bowl_cut: float) -> int:
        count = 0
        for p in players:
            bsr = self.batting_sr(p)
            bec = self.bowling_eco(p)
            if (bsr is not None and bsr >= bat_cut) or (bec is not None and bec <= bowl_cut):
                count += 1
        return count

    def xi_balance_index(self, players) -> float:
        bat = self.xi_top_batting(players)
        bwl = self.xi_top_bowling(players)
        return abs(bat - (2.0 - bwl))

    def xi_recent_form(self, players) -> float:
        scores = [self.player_form(p) for p in players if self.player_form(p) != 0.0]
        return float(np.mean(scores)) if scores else 0.0

    def xi_experience(self, players) -> float:
        exps = [self.player_experience(p) for p in players]
        return float(np.mean(exps)) if exps else 0.0

    def venue_avg_runs(self, venue) -> float:
        vals = self.venue_runs.get(venue, [])
        return float(np.mean(vals)) if vals else 160.0

    def venue_avg_wickets(self, venue) -> float:
        vals = self.venue_wickets.get(venue, [])
        return float(np.mean(vals)) if vals else 13.0

    def venue_type_score(self, venue) -> float:
        avg_r = self.venue_avg_runs(venue)
        if avg_r >= 170:
            return 1.0
        elif avg_r <= 145:
            return -1.0
        return 0.0

    def team_venue_winrate(self, team, venue) -> float:
        d = self.venue_team.get(venue, {}).get(team, {})
        if d.get("matches", 0) >= 3:
            return d["wins"] / d["matches"]
        return 0.5

    def global_percentile_cutoffs(self):
        bat_vals  = [d["runs"] / d["balls"] for d in self.batting.values() if d["balls"] >= MIN_BALLS_BAT]
        bowl_vals = [d["runs"] / d["balls"] for d in self.bowling.values() if d["balls"] >= MIN_BALLS_BOWL]
        bat_cut   = float(np.percentile(bat_vals,  STAR_PERCENTILE * 100))        if bat_vals  else 2.0
        bowl_cut  = float(np.percentile(bowl_vals, (1 - STAR_PERCENTILE) * 100))  if bowl_vals else 1.0
        return bat_cut, bowl_cut

    def update(self, match_row: pd.Series, match_bbb, xi1_players, xi2_players):
        winner     = match_row["match_won_by"]
        norm_venue = normalize_venue(str(match_row["venue"]))
        team1      = match_row["team1"]
        team2      = match_row["team2"]

        self.venue_team.setdefault(norm_venue, {})
        for team in (team1, team2):
            self.venue_team[norm_venue].setdefault(team, {"wins": 0, "matches": 0})
            self.venue_team[norm_venue][team]["matches"] += 1
            if team == winner:
                self.venue_team[norm_venue][team]["wins"] += 1

        if match_bbb is None or match_bbb.empty:
            return

        wkt_rows      = match_bbb[
            match_bbb["wicket_kind"].notna()
            & ~match_bbb["wicket_kind"].isin(NON_BOWLER_WICKETS)
        ]
        total_runs    = int(match_bbb["runs_total"].sum())
        total_wickets = len(wkt_rows)
        self.venue_runs.setdefault(norm_venue, []).append(total_runs)
        self.venue_wickets.setdefault(norm_venue, []).append(total_wickets)

        bat = match_bbb.groupby("batter").agg(
            runs=("runs_batter", "sum"), balls=("valid_ball", "sum")
        )
        for player, r in bat.iterrows():
            d = self.batting.setdefault(player, {"runs": 0, "balls": 0})
            d["runs"]  += int(r["runs"])
            d["balls"] += int(r["balls"])

        bowl = match_bbb.groupby("bowler").agg(
            runs=("runs_bowler", "sum"), balls=("valid_ball", "sum")
        )
        for player, r in bowl.iterrows():
            d = self.bowling.setdefault(player, {"runs": 0, "balls": 0, "wickets": 0})
            d["runs"]  += int(r["runs"])
            d["balls"] += int(r["balls"])

        if not wkt_rows.empty:
            for player, cnt in wkt_rows.groupby("bowler").size().items():
                self.bowling.setdefault(player, {"runs": 0, "balls": 0, "wickets": 0})
                self.bowling[player]["wickets"] += int(cnt)

        all_players = list(set(xi1_players + xi2_players))
        for p in all_players:
            self.matches_played[p] = self.matches_played.get(p, 0) + 1

        for player, r in bat.iterrows():
            score = r["runs"] / r["balls"] if int(r["balls"]) >= 6 else 0.0
            q = self.recent_scores.setdefault(player, deque(maxlen=FORM_WINDOW))
            q.append(score)

        for player, r in bowl.iterrows():
            if player not in bat.index and int(r["balls"]) >= 6:
                score = 2.0 - (r["runs"] / r["balls"])
                q = self.recent_scores.setdefault(player, deque(maxlen=FORM_WINDOW))
                q.append(score)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

BASE_FEATURE_NAMES = [
    "batting_strength_diff",
    "bowling_strength_diff",
    "allrounder_diff",
    "batting_depth_diff",
    "bowling_depth_diff",
    "balance_index_diff",
    "star_player_diff",
    "home_ground_advantage",
    "chasing_advantage",
    "toss_advantage",
    "venue_avg_runs",
    "venue_wickets_avg",
    "venue_type",
    "venue_winrate_diff",
    "recent_form_diff",
    "experience_diff",
    "bat_x_venue",
    "bowl_x_venue",
    "bat_x_chasing",
]


def get_xi_players(playing_xi: dict, mid, team1: str, team2: str):
    xi_entry = playing_xi.get(str(mid), {})
    xi1_data = xi_entry.get("team1", {})
    xi2_data = xi_entry.get("team2", {})
    if xi1_data.get("name") == team1:
        return xi1_data.get("players", []), xi2_data.get("players", [])
    elif xi2_data.get("name") == team1:
        return xi2_data.get("players", []), xi1_data.get("players", [])
    return xi1_data.get("players", []), xi2_data.get("players", [])


def build_features(matches: pd.DataFrame, bbb: pd.DataFrame,
                   playing_xi: dict, home_ground_dict: dict):
    bbb_grouped    = build_bbb_match_index(bbb)
    toss_dec_index = build_toss_decision_index(bbb)
    stats          = CumulativeStats()

    rows, targets, dates_list = [], [], []

    for _, row in matches.iterrows():
        team1      = row["team1"]
        team2      = row["team2"]
        winner     = row["match_won_by"]
        norm_venue = normalize_venue(str(row["venue"]))
        season     = str(row["season"])
        toss_w     = row["toss_winner"]
        mid        = row["match_id"]
        date       = row["date"]

        valid_result = winner in (team1, team2)

        home_abbrev = None
        for sk in get_season_keys(season):
            key = (sk, norm_venue)
            if key in home_ground_dict:
                home_abbrev = home_ground_dict[key]
                break

        abbrev1 = TEAM_TO_ABBREV.get(team1)
        abbrev2 = TEAM_TO_ABBREV.get(team2)

        if home_abbrev is not None and home_abbrev == abbrev2:
            team1, team2     = team2, team1
            abbrev1, abbrev2 = abbrev2, abbrev1

        xi1_players, xi2_players = get_xi_players(playing_xi, mid, team1, team2)

        bat_cut, bowl_cut = stats.global_percentile_cutoffs()

        # ── PLAYER QUALITY ───────────────────────────────────────────────────
        bat1 = stats.xi_top_batting(xi1_players)
        bat2 = stats.xi_top_batting(xi2_players)
        batting_strength_diff = bat1 - bat2

        eco1 = stats.xi_top_bowling(xi1_players)
        eco2 = stats.xi_top_bowling(xi2_players)
        bowling_strength_diff = eco2 - eco1

        allrounder_diff = (stats.xi_allrounder_count(xi1_players) -
                           stats.xi_allrounder_count(xi2_players))

        # ── TEAM COMPOSITION ─────────────────────────────────────────────────
        batting_depth_diff = (stats.xi_batting_depth(xi1_players) -
                              stats.xi_batting_depth(xi2_players))
        bowling_depth_diff = (stats.xi_bowling_depth(xi1_players) -
                              stats.xi_bowling_depth(xi2_players))
        balance_index_diff = (stats.xi_balance_index(xi1_players) -
                              stats.xi_balance_index(xi2_players))
        star_player_diff   = (stats.xi_star_count(xi1_players, bat_cut, bowl_cut) -
                              stats.xi_star_count(xi2_players, bat_cut, bowl_cut))

        # ── MATCH CONTEXT ────────────────────────────────────────────────────
        if home_abbrev == abbrev1:
            home_ground_advantage = 1
        elif home_abbrev == abbrev2:
            home_ground_advantage = -1
        else:
            home_ground_advantage = 0

        if toss_w == team1:
            toss_advantage = 1
        elif toss_w == team2:
            toss_advantage = -1
        else:
            toss_advantage = 0

        toss_info       = toss_dec_index.get(mid, {})
        toss_dec        = toss_info.get("toss_decision", "")
        toss_winner_bbb = toss_info.get("toss_winner", "")
        if   toss_winner_bbb == team1 and toss_dec == "field":
            chasing_advantage = 1
        elif toss_winner_bbb == team2 and toss_dec == "bat":
            chasing_advantage = 1
        elif toss_winner_bbb == team1 and toss_dec == "bat":
            chasing_advantage = -1
        elif toss_winner_bbb == team2 and toss_dec == "field":
            chasing_advantage = -1
        else:
            chasing_advantage = 0

        # ── VENUE FEATURES ───────────────────────────────────────────────────
        venue_avg_runs    = stats.venue_avg_runs(norm_venue)
        venue_wickets_avg = stats.venue_avg_wickets(norm_venue)
        venue_type        = stats.venue_type_score(norm_venue)
        venue_winrate_diff = (stats.team_venue_winrate(team1, norm_venue) -
                              stats.team_venue_winrate(team2, norm_venue))

        # ── FORM & EXPERIENCE ────────────────────────────────────────────────
        recent_form_diff = (stats.xi_recent_form(xi1_players) -
                            stats.xi_recent_form(xi2_players))
        experience_diff  = (stats.xi_experience(xi1_players) -
                            stats.xi_experience(xi2_players))

        # ── INTERACTION TERMS ────────────────────────────────────────────────
        bat_x_venue   = batting_strength_diff * venue_type
        bowl_x_venue  = bowling_strength_diff * venue_type
        bat_x_chasing = batting_strength_diff * chasing_advantage

        if valid_result:
            rows.append({
                "batting_strength_diff": batting_strength_diff,
                "bowling_strength_diff": bowling_strength_diff,
                "allrounder_diff":       allrounder_diff,
                "batting_depth_diff":    batting_depth_diff,
                "bowling_depth_diff":    bowling_depth_diff,
                "balance_index_diff":    balance_index_diff,
                "star_player_diff":      star_player_diff,
                "home_ground_advantage": home_ground_advantage,
                "chasing_advantage":     chasing_advantage,
                "toss_advantage":        toss_advantage,
                "venue_avg_runs":        venue_avg_runs,
                "venue_wickets_avg":     venue_wickets_avg,
                "venue_type":            venue_type,
                "venue_winrate_diff":    venue_winrate_diff,
                "recent_form_diff":      recent_form_diff,
                "experience_diff":       experience_diff,
                "bat_x_venue":           bat_x_venue,
                "bowl_x_venue":          bowl_x_venue,
                "bat_x_chasing":         bat_x_chasing,
            })
            targets.append(1 if winner == team1 else 0)
            dates_list.append(date)

        stats.update(row, bbb_grouped.get(mid), xi1_players, xi2_players)

    X     = pd.DataFrame(rows, columns=BASE_FEATURE_NAMES)
    y     = np.array(targets)
    dates = pd.Series(dates_list)
    return X, y, dates


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MULTICOLLINEARITY CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def drop_correlated_features(X: pd.DataFrame, threshold: float = CORR_DROP_THRESHOLD):
    corr  = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            drop.add(col)
    kept = [c for c in X.columns if c not in drop]
    if drop:
        print(f"\n  Dropped (|corr| > {threshold}): {sorted(drop)}")
    return X[kept], kept


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def chrono_split(X, y, dates, train_ratio=TRAIN_RATIO):
    n      = len(y)
    cutoff = int(n * train_ratio)
    return (X.iloc[:cutoff], X.iloc[cutoff:],
            y[:cutoff],      y[cutoff:],
            dates.iloc[:cutoff], dates.iloc[cutoff:])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_tr, y_tr):
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr)

    base_clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42
    )

    # 🔥 ADD THIS
    clf = CalibratedClassifierCV(
        base_clf,
        method="sigmoid",   # or "isotonic" (try later)
        cv=3
    )

    clf.fit(X_sc, y_tr)

    return clf, scaler


# ══════════════════════════════════════════════════════════════════════════════
# 7.  OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

SEP = "─" * 72


def print_section(title):
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print(f"{'═' * 72}")


def print_coefficients(clf, feature_names):
    base_clf = clf.calibrated_classifiers_[0].estimator
    coeffs = base_clf.coef_[0]

    print(f"\n  Intercept  :  {base_clf.intercept_[0]:+.4f}\n")
    print(f"  {'Feature':<30}  {'Coefficient':>12}  {'|Coef|':>8}  Rank")
    print(f"  {SEP[:65]}")

    pairs = sorted(zip(feature_names, coeffs), key=lambda x: abs(x[1]), reverse=True)

    for rank, (feat, coef) in enumerate(pairs, 1):
        bar = "█" * int(abs(coef) * 15)
        print(f"  {feat:<30}  {coef:>+12.4f}  {abs(coef):>8.4f}  #{rank}  {bar}")        
def print_oos_results(y_te, probs, preds):
    acc     = accuracy_score(y_te, preds)
    n       = len(y_te)
    correct = int(np.sum(y_te == preds))
    print(f"\n  Total matches       : {n}")
    print(f"  Correct predictions : {correct}")
    print(f"  Accuracy            : {acc:.2%}")

    avg_pred  = float(np.mean(probs))
    actual_wr = float(np.mean(y_te))
    print(f"\n  ── Calibration ───────────────────────────────")
    print(f"  Avg predicted P(team1 wins) : {avg_pred:.3f}")
    print(f"  Actual team1 win rate       : {actual_wr:.3f}")
    diff     = abs(avg_pred - actual_wr)
    cal_flag = "✓ well-calibrated" if diff < 0.03 else "⚠ mild bias"
    print(f"  Calibration gap             : {diff:.3f}  ({cal_flag})")

    print(f"\n  ── Probability Bins ──────────────────────────")
    print(f"  {'Bin':>14}  {'Matches':>8}  {'Pred %':>7}  {'Actual %':>9}")
    print(f"  {SEP[:55]}")
    for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask  = (probs >= lo) & (probs < hi)
        n_bin = int(mask.sum())
        if n_bin > 0:
            print(f"  [{lo:.1f} – {hi:.1f})  {n_bin:>10}  "
                  f"{np.mean(probs[mask]):>6.1%}  {np.mean(y_te[mask]):>9.1%}")
    print("\n  ── Calibration Check ───────────────────────────────")
    print(f"  Mean predicted probability : {np.mean(probs):.3f}")
    print(f"  Actual win rate            : {np.mean(y_te):.3f}")

def print_insights(clf, scaler, feature_names, X_tr, y_tr, X_te, y_te):
    X_tr_sc = scaler.transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    is_acc  = accuracy_score(y_tr, clf.predict(X_tr_sc))
    oos_acc = accuracy_score(y_te, clf.predict(X_te_sc))

    print(f"\n  In-sample accuracy   : {is_acc:.2%}")
    print(f"  Out-of-sample acc.   : {oos_acc:.2%}")
    gap  = is_acc - oos_acc
    flag = "✓ healthy" if gap <= 0.10 else "⚠ overfit"
    print(f"  Overfitting gap      : {gap:.2%}  {flag}")

    delta = oos_acc - PREVIOUS_OOS_ACC
    arrow = "▲" if delta >= 0 else "▼"
    print(f"\n  ── vs baseline ({PREVIOUS_OOS_ACC:.2%}) ─────────────────────────────────────")
    print(f"  OOS Δ  : {arrow} {abs(delta):.2%}  "
          f"({'improvement' if delta >= 0 else 'regression'})")

    print(f"\n  ── Feature Weights (sorted by |coef|) ────────────────────────────")
    # get underlying logistic model
    base_clf = clf.calibrated_classifiers_[0].estimator
    coeffs   = base_clf.coef_[0]
    for feat, coef in sorted(zip(feature_names, coeffs), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {feat:<32} →  {abs(coef):.4f}")

def print_significance_table(X_tr, y_tr, feature_names):
    print_section("VARIABLE SIGNIFICANCE (WALD TEST)")

    X_sm = sm.add_constant(X_tr)

    model = sm.Logit(y_tr, X_sm)
    result = model.fit(disp=False)

    summary = pd.DataFrame({
        "Variable": X_sm.columns,
        "Coef": result.params,
        "StdErr": result.bse,
        "Z": result.tvalues,
        "P_Value": result.pvalues
    })

    summary["Significant_5pct"] = summary["P_Value"] < 0.05
    summary = summary.sort_values("P_Value")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )
# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print_section("LOADING DATA")
    matches, bbb, playing_xi, home_ground_dict = load_data(
        MATCHES_PATH, BBB_PATH, PLAYING_XI_PATH, HOME_PATH
    )

    print_section("BUILDING FEATURES  (chronological, no lookahead)")
    print("  Processing matches …")
    X, y, dates = build_features(matches, bbb, playing_xi, home_ground_dict)
    print(f"  Valid matches for modelling : {len(y)}")
    print(f"  Feature matrix shape        : {X.shape}")
    print(f"  Base rate (team1 wins)      : {y.mean():.2%}")

    hga = X["home_ground_advantage"]
    print(f"\n  home_ground_advantage coverage:")
    print(f"    +1 (home team)  : {(hga == 1).sum()} matches")
    print(f"     0 (neutral)    : {(hga == 0).sum()} matches")

    print_section("MULTICOLLINEARITY CONTROL")
    X_clean, kept_features = drop_correlated_features(X)
    print(f"\n  Features retained : {len(kept_features)} / {len(BASE_FEATURE_NAMES)}")
    print(f"  Retained          : {kept_features}")

    X_tr, X_te, y_tr, y_te, d_tr, d_te = chrono_split(X_clean, y, dates)
    print(f"\n  Train : {d_tr.min().date()} → {d_tr.max().date()}  ({len(y_tr)} matches)")
    print(f"  Test  : {d_te.min().date()} → {d_te.max().date()}  ({len(y_te)} matches)")

    print_section("MODEL TRAINING  (LogisticRegression, C=0.5, StandardScaler)")
    clf, scaler = train_model(X_tr, y_tr)

    print_section("COEFFICIENTS & FEATURE IMPORTANCE")
    print_coefficients(clf, kept_features)

    print_section("OUT-OF-SAMPLE EVALUATION")
    X_te_sc = scaler.transform(X_te)
    probs   = clf.predict_proba(X_te_sc)[:, 1]
    preds   = (probs > 0.5).astype(int)
    print_oos_results(y_te, probs, preds)
    # ── FILTERED (HIGH CONFIDENCE) RESULTS ─────────────────────────
    threshold = 0.60

    mask = (probs > threshold) | (probs < (1 - threshold))

    filtered_preds = preds[mask]
    filtered_y     = y_te[mask]
    filtered_probs = probs[mask]

    print("\n════════════════════════════════════════════════════════════")
    print("  FILTERED RESULTS (HIGH CONFIDENCE ONLY)")
    print("════════════════════════════════════════════════════════════")

    print(f"  Matches used      : {len(filtered_y)}")

    if len(filtered_y) > 0:
        acc = accuracy_score(filtered_y, filtered_preds)
        print(f"  Filtered accuracy : {acc:.2%}")
        print(f"  Avg confidence    : {np.mean(filtered_probs):.3f}")
    else:
        print("  No matches passed threshold")

    print_section("SUMMARY INSIGHTS")
    print_insights(clf, scaler, kept_features, X_tr, y_tr, X_te, y_te)
    print_significance_table(X_tr, y_tr, kept_features)

    print(f"\n{'═' * 72}\n")


if __name__ == "__main__":
    main()