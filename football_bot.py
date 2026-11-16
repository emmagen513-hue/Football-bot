"""Football Telegram bot: Dixon-Coles model + bookmaker blend + accumulator builder. ONE file.

Before running: paste your NEW Telegram token and Gemini key below (keep this file private).
"""
import os
import logging

# ---------- PUT YOUR NEW KEYS HERE (between the quotes) ----------
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "PASTE-YOUR-NEW-TELEGRAM-TOKEN-HERE")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE-YOUR-NEW-GEMINI-KEY-HERE")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# -----------------------------------------------------------------

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
from google import genai

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
for _noisy in ("httpx", "httpcore"):   # these log request URLs, which contain the bot token
    logging.getLogger(_noisy).setLevel(logging.WARNING)
client = None

import io
import urllib.request
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10
XI = 0.0018   # time decay per day (about a 1-year half-life)
REG = 0.5     # small penalty that stops extreme ratings for teams with few games
BASE = "https://www.football-data.co.uk"
HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------- DATA ----------------
def season_codes(n=2):
    """['2627', '2526'] style codes, current season first."""
    now = datetime.now()
    year = now.year if now.month >= 7 else now.year - 1
    return [f"{str(y)[2:]}{str(y + 1)[2:]}" for y in range(year, year - n, -1)]


def load_results(div="E0", n_seasons=2):
    """Download results for a league (E0=EPL, E1=Championship, SP1, I1, D1)."""
    frames = []
    for code in season_codes(n_seasons):
        url = f"{BASE}/mmz4281/{code}/{div}.csv"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(raw))
            df.columns = [c.strip() for c in df.columns]
            frames.append(df)
            print(f"Loaded {div} {code}: {len(df)} rows")
        except Exception as e:
            print(f"Could not load {div} {code}: {e}")
    if not frames:
        raise RuntimeError("No data could be downloaded.")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG", "Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    return df


# ---------------- MODEL ----------------
def _tau(x, y, lh, la, rho):
    """Dixon-Coles low-score correction, vectorised."""
    t = np.ones_like(lh)
    m = (x == 0) & (y == 0); t[m] = 1 - lh[m] * la[m] * rho
    m = (x == 0) & (y == 1); t[m] = 1 + lh[m] * rho
    m = (x == 1) & (y == 0); t[m] = 1 + la[m] * rho
    m = (x == 1) & (y == 1); t[m] = 1 - rho
    return np.maximum(t, 1e-6)


class DixonColes:
    def fit(self, df, xi=XI, reg=REG):
        teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        h = df["HomeTeam"].map(idx).to_numpy()
        a = df["AwayTeam"].map(idx).to_numpy()
        x = df["FTHG"].to_numpy()
        y = df["FTAG"].to_numpy()

        days = (df["Date"].max() - df["Date"]).dt.days.to_numpy()
        w = np.exp(-xi * days)  # recent matches weigh more

        def unpack(p):
            att = p[:n] - p[:n].mean()
            dfn = p[n:2 * n] - p[n:2 * n].mean()
            return att, dfn, p[2 * n], p[2 * n + 1], p[2 * n + 2]

        def nll_grad(p):
            """Negative log-likelihood and its exact gradient (makes the fit far faster)."""
            att, dfn, home, mu, rho = unpack(p)
            lh = np.exp(mu + home + att[h] - dfn[a])
            la = np.exp(mu + att[a] - dfn[h])
            tau = _tau(x, y, lh, la, rho)
            ll = w * (
                np.log(tau)
                + x * np.log(lh) - lh - gammaln(x + 1)
                + y * np.log(la) - la - gammaln(y + 1)
            )
            f = -ll.sum() + reg * (np.sum(att ** 2) + np.sum(dfn ** 2))

            gH = w * (x - lh)          # d(log-lik)/d(log home rate)
            gA = w * (y - la)          # d(log-lik)/d(log away rate)
            gR = np.zeros_like(lh)     # d(log-lik)/d(rho)
            m = (x == 0) & (y == 0)
            c = -w[m] * lh[m] * la[m] * rho / tau[m]
            gH[m] += c; gA[m] += c; gR[m] = -w[m] * lh[m] * la[m] / tau[m]
            m = (x == 0) & (y == 1)
            gH[m] += w[m] * lh[m] * rho / tau[m]; gR[m] = w[m] * lh[m] / tau[m]
            m = (x == 1) & (y == 0)
            gA[m] += w[m] * la[m] * rho / tau[m]; gR[m] = w[m] * la[m] / tau[m]
            m = (x == 1) & (y == 1)
            gR[m] = -w[m] / tau[m]

            g_att = -(np.bincount(h, gH, n) + np.bincount(a, gA, n)) + 2 * reg * att
            g_def = (np.bincount(a, gH, n) + np.bincount(h, gA, n)) + 2 * reg * dfn
            g = np.zeros(2 * n + 3)
            g[:n] = g_att - g_att.mean()          # centring the ratings removes the mean
            g[n:2 * n] = g_def - g_def.mean()
            g[2 * n] = -gH.sum()
            g[2 * n + 1] = -(gH.sum() + gA.sum())
            g[2 * n + 2] = -gR.sum()
            return f, g

        p0 = np.zeros(2 * n + 3)
        p0[2 * n] = 0.25      # home advantage
        p0[2 * n + 1] = 0.2   # baseline log goals
        bounds = [(None, None)] * (2 * n + 2) + [(-0.2, 0.2)]
        res = minimize(nll_grad, p0, jac=True, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            print("Warning: optimiser did not fully converge:", res.message)

        self.teams = teams
        self.att, self.dfn, self.home, self.mu, self.rho = unpack(res.x)
        self.idx = idx
        self.n_matches = len(df)
        return self

    def resolve(self, name):
        """Exact (case-insensitive) match first, then substring."""
        q = name.strip().lower()
        for t in self.teams:
            if t.lower() == q:
                return t
        for t in self.teams:
            if q and q in t.lower():
                return t
        raise ValueError(f"Team not found: {name}")

    def _lambdas(self, home, away):
        i, j = self.idx[home], self.idx[away]
        lh = float(np.exp(self.mu + self.home + self.att[i] - self.dfn[j]))
        la = float(np.exp(self.mu + self.att[j] - self.dfn[i]))
        return lh, la

    def score_matrix(self, home, away):
        lh, la = self._lambdas(home, away)
        k = np.arange(MAX_GOALS + 1)
        ph = np.exp(-lh + k * np.log(lh) - gammaln(k + 1))
        pa = np.exp(-la + k * np.log(la) - gammaln(k + 1))
        m = np.outer(ph, pa)
        m[0, 0] *= max(1e-6, 1 - lh * la * self.rho)
        m[0, 1] *= 1 + lh * self.rho
        m[1, 0] *= 1 + la * self.rho
        m[1, 1] *= 1 - self.rho
        return m / m.sum(), lh, la

    def markets(self, home_name, away_name):
        home, away = self.resolve(home_name), self.resolve(away_name)
        if home == away:
            raise ValueError("Home and away team are the same.")
        m, lh, la = self.score_matrix(home, away)
        i, j = np.indices(m.shape)
        p = {
            "home_win": m[i > j].sum(),
            "draw": m[i == j].sum(),
            "away_win": m[i < j].sum(),
            "over_1_5": m[i + j > 1].sum(),
            "over_2_5": m[i + j > 2].sum(),
            "btts": m[(i > 0) & (j > 0)].sum(),
        }
        p["1X"] = p["home_win"] + p["draw"]
        p["X2"] = p["draw"] + p["away_win"]
        p["12"] = p["home_win"] + p["away_win"]
        out = {k: round(float(v), 4) for k, v in p.items()}
        out.update({
            "fixture": f"{home} vs {away}",
            "home_xg": round(lh, 2),
            "away_xg": round(la, 2),
        })
        return out



# ---------------- ODDS BACKTEST ----------------
# (home, draw, away) odds columns, best source first: average closing odds,
# then Pinnacle closing, then Bet365 closing, then the same with opening odds.
TRIPLES = [
    ("AvgCH", "AvgCD", "AvgCA"),
    ("PSCH", "PSCD", "PSCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgH", "AvgD", "AvgA"),
    ("PSH", "PSD", "PSA"),
    ("B365H", "B365D", "B365A"),
]
# (over 2.5, under 2.5) odds columns
PAIRS = [
    ("AvgC>2.5", "AvgC<2.5"),
    ("PC>2.5", "PC<2.5"),
    ("B365C>2.5", "B365C<2.5"),
    ("Avg>2.5", "Avg<2.5"),
    ("P>2.5", "P<2.5"),
    ("B365>2.5", "B365<2.5"),
]


def pick_columns(df, options, min_share=0.8):
    tail = df.tail(200)
    for cols in options:
        if all(c in df.columns for c in cols) and all(tail[c].notna().mean() >= min_share for c in cols):
            return list(cols)
    return None


def remove_margin(odds):
    """Turn bookmaker odds into probabilities that sum to 1."""
    inv = 1.0 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def backtest(df, n_test=150, refit_every=10, min_train=200):
    tri = pick_columns(df, TRIPLES)
    pair = pick_columns(df, PAIRS)
    if tri is None:
        raise RuntimeError("No usable 1X2 odds columns found in the data.")
    print("1X2 odds columns:", tri)
    print("Over/Under odds columns:", pair)

    start = max(min_train, len(df) - n_test)
    bets, ll_model, ll_market = [], [], []

    for block_start in range(start, len(df), refit_every):
        model = DixonColes().fit(df.iloc[:block_start])
        block = df.iloc[block_start:block_start + refit_every]
        print(f"Tested up to match {min(block_start + refit_every, len(df))} of {len(df)}")

        for _, r in block.iterrows():
            if r["HomeTeam"] not in model.idx or r["AwayTeam"] not in model.idx:
                continue  # team with no history (e.g. just promoted)
            res = model.markets(r["HomeTeam"], r["AwayTeam"])
            hg, ag = r["FTHG"], r["FTAG"]

            # ---- 1X2 ----
            odds = [r[c] for c in tri]
            if not any(pd.isna(odds)) and min(odds) > 1:
                p_model = [res["home_win"], res["draw"], res["away_win"]]
                p_market = remove_margin(odds)
                actual = 0 if hg > ag else (1 if hg == ag else 2)
                ll_model.append(-np.log(max(p_model[actual], 1e-9)))
                ll_market.append(-np.log(max(p_market[actual], 1e-9)))
                for k, name in enumerate(["home", "draw", "away"]):
                    bets.append({
                        "market": "1X2", "sel": name,
                        "p_model": p_model[k], "p_market": p_market[k], "odds": odds[k],
                        "edge": p_model[k] * odds[k] - 1, "won": actual == k,
                    })

            # ---- Over/Under 2.5 ----
            if pair:
                odds = [r[c] for c in pair]
                if not any(pd.isna(odds)) and min(odds) > 1:
                    p_model = [res["over_2_5"], 1 - res["over_2_5"]]
                    p_market = remove_margin(odds)
                    over = (hg + ag) > 2
                    for k, name in enumerate(["over 2.5", "under 2.5"]):
                        bets.append({
                            "market": "O/U 2.5", "sel": name,
                            "p_model": p_model[k], "p_market": p_market[k], "odds": odds[k],
                            "edge": p_model[k] * odds[k] - 1,
                            "won": over if k == 0 else not over,
                        })
    return bets, ll_model, ll_market


def summarize(bets, thresholds=(0.0, 0.03, 0.05, 0.10)):
    print("\nFlat 1-unit stakes, by minimum edge (edge = model prob x odds - 1)")
    for market in ["1X2", "O/U 2.5", "ALL"]:
        pool = [b for b in bets if market == "ALL" or b["market"] == market]
        print(f"\n{market}")
        print(f"{'min edge':>9} {'bets':>5} {'wins':>5} {'profit':>8} {'ROI':>7}")
        for t in thresholds:
            sel = [b for b in pool if b["edge"] >= t]
            if not sel:
                print(f"{t:>9.0%} {0:>5} {0:>5} {0:>8.2f} {'-':>7}")
                continue
            profit = sum((b["odds"] - 1) if b["won"] else -1 for b in sel)
            wins = sum(b["won"] for b in sel)
            print(f"{t:>9.0%} {len(sel):>5} {wins:>5} {profit:>8.2f} {profit / len(sel):>7.1%}")



# ---------------- SETTINGS (from your test results) ----------------
XI_USED = 0.005        # ~4.5 month fade
MODEL_WEIGHT = 0.05    # 5% model, 95% bookmaker (the model did not beat the bookmaker in tests)
OVERROUND = 1.06       # margin used to ESTIMATE odds we have no data for
HAIRCUT = 0.04         # lowers every leg probability (the model was ~5 points too confident)
DAYS_AHEAD = 4         # only use fixtures in the next few days


# ---------------- CANDIDATE LEGS ----------------
def est_odds(p):
    return max(1.01, round(1.0 / (p * OVERROUND), 2))


def match_candidates(model, home, away, odds_hda, odds_ou=None, w=MODEL_WEIGHT):
    """All possible legs for one match, with blended probabilities."""
    res = model.markets(home, away)
    mk = remove_margin(odds_hda)
    pH = w * res["home_win"] + (1 - w) * mk[0]
    pD = w * res["draw"] + (1 - w) * mk[1]
    pA = w * res["away_win"] + (1 - w) * mk[2]
    fixture = f"{home} vs {away}"

    # Over 1.5: no free odds exist. Adjust the model's number using the market's
    # view of Over 2.5 (heuristic), so it is not purely the model's opinion.
    p15 = res["over_1_5"]
    if odds_ou is not None:
        mo = remove_margin(odds_ou)[0]
        blend_o25 = w * res["over_2_5"] + (1 - w) * mo
        under_ratio = (1 - blend_o25) / max(1 - res["over_2_5"], 1e-6)
        p15 = float(np.clip(1 - (1 - res["over_1_5"]) * under_ratio, 0, 1))

    def leg(key, label, p, odds, source):
        return {"fixture": fixture, "teams": [home, away], "key": key, "pick": label,
                "p": round(max(0.0, float(p) - HAIRCUT), 4), "odds": round(float(odds), 2), "odds_source": source}

    return [
        leg("home", f"{home} to win", pH, odds_hda[0], "bookmaker"),
        leg("away", f"{away} to win", pA, odds_hda[2], "bookmaker"),
        leg("1X", f"{home} or draw", pH + pD, est_odds(pH + pD), "estimated"),
        leg("X2", f"{away} or draw", pD + pA, est_odds(pD + pA), "estimated"),
        leg("o15", "Over 1.5 goals", p15, est_odds(p15), "estimated"),
    ]



# ================= ACCUMULATOR FILTER (any target odds) =================
import os
import re
import sys
import time
import math
import json
import asyncio
import logging
import threading

FLOORS = [0.68, 0.60, 0.52, 0.45, 0.38, 0.30, 0.25]  # minimum leg probability, relaxed step by step
MIN_LEG_ODDS = 1.15
MIN_TARGET, MAX_TARGET = 1.5, 5000.0


def _best_combo(groups, low, high, max_legs):
    """One option per match at most. Finds the set of legs with the highest combined
    probability whose odds multiply to between low and high."""
    step = max(0.002, math.log(high) / 2500)
    B = int(math.floor(math.log(high) / step))
    blow = int(math.ceil(math.log(low) / step))
    K = min(len(groups), max_legs)
    NEG = -1e18

    def weight(c):
        return int(round(math.log(c["odds"]) / step))

    best = np.full((K + 1, B + 1), NEG)
    best[0, 0] = 0.0
    choices = []
    for opts in groups:
        new = best.copy()
        ch = np.full((K + 1, B + 1), -1, dtype=np.int16)
        for i, c in enumerate(opts):
            w = weight(c)
            if w <= 0 or w > B:
                continue
            cand = best[:-1, : B + 1 - w] + math.log(c["p"])
            tgt = new[1:, w:]
            chv = ch[1:, w:]
            mask = cand > tgt
            tgt[mask] = cand[mask]
            chv[mask] = i
        choices.append(ch)
        best = new

    window = best[1:, blow:B + 1]
    if window.size == 0 or window.max() < -1e17:
        return None
    k_idx, b_idx = np.unravel_index(np.argmax(window), window.shape)
    k, b = int(k_idx) + 1, int(b_idx) + blow
    chosen = []
    for g in range(len(groups) - 1, -1, -1):
        i = int(choices[g][k, b])
        if i >= 0:
            c = groups[g][i]
            chosen.append(c)
            b -= weight(c)
            k -= 1
    return chosen


def build_accumulator(cands, target=10.0, max_legs=40):
    target = float(min(max(target, MIN_TARGET), MAX_TARGET))
    # small targets are held close to what was asked; big ones get more room
    low, high = (target * 0.95, target * 1.10) if target <= 50 else (target * 0.9, target * 1.15)

    by_fix = {}
    for c in cands:
        if c["odds"] >= MIN_LEG_ODDS and c["p"] > 0:
            by_fix.setdefault(c["fixture"], []).append(c)
    if not by_fix:
        return {"error": "no_legs"}

    chosen, used_floor = None, None
    for floor in FLOORS:
        groups = [[c for c in legs if c["p"] >= floor] for legs in by_fix.values()]
        groups = [g for g in groups if g]
        if not groups:
            continue
        chosen = _best_combo(groups, low, high, max_legs)
        if chosen:
            used_floor = floor
            break

    if not chosen:
        tops = sorted((max(l["odds"] for l in legs if l["p"] >= FLOORS[-1])
                       for legs in by_fix.values() if any(l["p"] >= FLOORS[-1] for l in legs)),
                      reverse=True)[:max_legs]
        reach = float(np.prod(tops)) if tops else 0.0
        return {"error": "unreachable", "target_odds": target, "max_odds": round(reach, 1),
                "matches": len(by_fix)}

    chosen = sorted(chosen, key=lambda c: -c["p"])
    total = float(np.prod([c["odds"] for c in chosen]))
    combined = float(np.prod([c["p"] for c in chosen]))
    warnings = []
    if used_floor < FLOORS[0]:
        warnings.append(f"To reach {target:g} odds I had to include riskier legs "
                        f"(down to {used_floor:.0%} chance each).")
    if any(c["odds_source"] == "estimated" for c in chosen):
        warnings.append("Odds marked (est.) are estimates. Check real prices before betting.")
    return {
        "target_odds": target,
        "total_odds": round(total, 2),
        "combined_probability": combined,
        "legs": [{**{k: c[k] for k in ("fixture", "pick", "p", "odds", "odds_source", "key", "teams")},
                  "date": c.get("date"), "div": c.get("div"), "kickoff": c.get("kickoff")} for c in chosen],
        "warnings": warnings,
    }


# ================= DATA (cached) =================
LEAGUE_CODES = {   # name: football-data division code (top leagues first)
    "EPL": "E0", "La Liga": "SP1", "Serie A": "I1", "Bundesliga": "D1",
    "Championship": "E1", "League One": "E2", "League Two": "E3",
    "Serie B": "I2", "La Liga 2": "SP2", "Bundesliga 2": "D2",
    "Ligue 1": "F1", "Ligue 2": "F2", "Eredivisie": "N1",
    "Primeira Liga": "P1", "Super Lig": "T1",
}
CODE_TO_NAME = {v: k for k, v in LEAGUE_CODES.items()}
# Division codes to keep OUT of accumulators (fill in from league_check.py), e.g. {"E3", "T1"}
ACCA_EXCLUDE = set()
FAIL_TTL = 900   # seconds before retrying a league whose data failed to load
MODEL_TTL = 6 * 3600
FIX_TTL = 3600
_MODELS, _FX = {}, {"df": None, "t": 0.0}
_LOCKS, _GUARD, _FX_LOCK, _FAILED = {}, threading.Lock(), threading.Lock(), {}


def load_fixtures():
    req = urllib.request.Request(f"{BASE}/fixtures.csv", headers=HEADERS)
    with urllib.request.urlopen(req, timeout=20) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    fx = pd.read_csv(io.StringIO(raw))
    fx.columns = [c.strip() for c in fx.columns]
    fx["Date"] = pd.to_datetime(fx["Date"], dayfirst=True, errors="coerce")
    return fx


def _lock_for(code):
    with _GUARD:
        return _LOCKS.setdefault(code, threading.Lock())


def get_model(code):
    """Fitted model for a league, or None if its data could not be loaded."""
    with _lock_for(code):
        entry = _MODELS.get(code)
        if entry is not None and time.time() - entry[1] <= MODEL_TTL:
            return entry[0]
        if time.time() - _FAILED.get(code, 0) < FAIL_TTL:
            return entry[0] if entry else None
        try:
            model = DixonColes().fit(load_results(code, n_seasons=3), xi=XI_USED)
        except Exception as e:
            logging.warning("League %s unavailable: %s", code, e)
            _FAILED[code] = time.time()
            return entry[0] if entry else None
        _MODELS[code] = (model, time.time())
        return model


def prefetch(codes, workers=4):
    """Download and fit several leagues in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(get_model, codes))


def get_fixtures():
    with _FX_LOCK:
        if _FX["df"] is None or time.time() - _FX["t"] > FIX_TTL:
            _FX["df"] = load_fixtures()
            _FX["t"] = time.time()
        return _FX["df"]


def row_odds(r, options):
    """First odds set (a tuple of columns) that is complete for this fixture row."""
    for cols in options:
        vals = [r.get(c) for c in cols]
        if all(v is not None and not pd.isna(v) and v > 1 for v in vals):
            return [float(v) for v in vals]
    return None


# ================= KICKOFF TIMES =================
from zoneinfo import ZoneInfo

UK_TZ = ZoneInfo("Europe/London")   # football-data lists match times in UK time (assumed)


def not_started(row, buffer_minutes=10):
    """True if the match has not kicked off yet (with a small safety buffer)."""
    now = pd.Timestamp.now(tz="UTC")
    day = pd.Timestamp(row["Date"])
    t = row.get("Time")
    if t is not None and not pd.isna(t):
        try:
            hh, mm = str(t).strip().split(":")[:2]
            ko = day.replace(hour=int(hh), minute=int(mm)).tz_localize(
                UK_TZ, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")
            return ko > now + pd.Timedelta(minutes=buffer_minutes)
        except Exception:
            pass
    return day.date() > now.date()   # no usable time: only trust tomorrow onwards


def kickoff_label(row):
    label = pd.Timestamp(row["Date"]).strftime("%a %d %b")
    t = row.get("Time")
    if t is not None and not pd.isna(t):
        label += f" {str(t).strip()[:5]} UK"
    return label


def compute_candidates(when=None):
    """Returns (candidates, None) or (None, message for the user).
    when = (start_date, end_date, label) or None for the default window."""
    fx = get_fixtures()
    if when:
        start, end, label = pd.Timestamp(when[0]), pd.Timestamp(when[1]), when[2]
    else:
        start = pd.Timestamp.now().normalize()
        end = start + pd.Timedelta(days=DAYS_AHEAD)
        label = f"the next {DAYS_AHEAD} days"
    fx = fx[fx["Div"].isin(set(LEAGUE_CODES.values()) - ACCA_EXCLUDE)]
    fx = fx[(fx["Date"] >= start) & (fx["Date"] <= end)]
    if not fx.empty:
        fx = fx[fx.apply(not_started, axis=1)]   # drop matches that have already kicked off
    if fx.empty:
        return None, (f"No fixtures that haven't kicked off yet for {label}. The fixtures file is "
                      "refreshed on Friday afternoon (weekend games) and Tuesday (midweek games).")
    tri_opts = [c for c in TRIPLES if all(x in fx.columns for x in c)]
    pair_opts = [c for c in PAIRS if all(x in fx.columns for x in c)]
    if not tri_opts:
        return None, "The fixtures have no usable odds yet. Try again closer to kick-off."

    prefetch(sorted(fx["Div"].unique()))   # download and fit the leagues in parallel

    cands, used = [], set()
    for _, r in fx.sort_values("Date").iterrows():   # a team is used only once
        h, a = r["HomeTeam"], r["AwayTeam"]
        if not isinstance(h, str) or not isinstance(a, str) or h in used or a in used:
            continue
        odds = row_odds(r, tri_opts)
        if odds is None:
            continue
        model = get_model(r["Div"])
        if model is None or h not in model.idx or a not in model.idx:
            continue
        ou = row_odds(r, pair_opts) if pair_opts else None
        used |= {h, a}
        new = match_candidates(model, h, a, odds, ou)
        for c in new:
            c["date"] = str(r["Date"].date())
            c["div"] = r["Div"]
            c["kickoff"] = kickoff_label(r)
        cands += new
    if not cands:
        return None, "None of the listed fixtures could be matched to team data."
    return cands, None


def blend_with_market(res, odds_hda, odds_ou=None, w=MODEL_WEIGHT):
    """Replace the model's win/draw/loss (and Over 2.5) with a mostly-bookmaker mix."""
    out = dict(res)
    mk = remove_margin(odds_hda)
    ph = w * res["home_win"] + (1 - w) * mk[0]
    pdr = w * res["draw"] + (1 - w) * mk[1]
    pa = w * res["away_win"] + (1 - w) * mk[2]
    out.update(home_win=round(float(ph), 4), draw=round(float(pdr), 4), away_win=round(float(pa), 4))
    out["1X"], out["X2"], out["12"] = round(float(ph + pdr), 4), round(float(pdr + pa), 4), round(float(ph + pa), 4)
    if odds_ou:
        o25 = w * res["over_2_5"] + (1 - w) * remove_margin(odds_ou)[0]
        ratio = (1 - o25) / max(1 - res["over_2_5"], 1e-6)
        out["over_2_5"] = round(float(o25), 4)
        out["over_1_5"] = round(float(np.clip(1 - (1 - res["over_1_5"]) * ratio, 0, 1)), 4)
    out["market_blend"] = True
    return out


def find_fixture(res):
    """Next listed fixture for a predicted pairing, or None."""
    try:
        home, away = res["fixture"].split(" vs ", 1)
        fx = get_fixtures()
        today = pd.Timestamp.now().normalize()
        m = fx[(fx["HomeTeam"] == home) & (fx["AwayTeam"] == away) & (fx["Date"] >= today)]
        if not m.empty:
            m = m[m.apply(not_started, axis=1)]
        if m.empty:
            return None
        r = m.sort_values("Date").iloc[0]
        tri_opts = [c for c in TRIPLES if all(x in fx.columns for x in c)]
        pair_opts = [c for c in PAIRS if all(x in fx.columns for x in c)]
        return {"date": str(r["Date"].date()), "div": r["Div"], "home": home, "away": away,
                "kickoff": kickoff_label(r),
                "odds_hda": row_odds(r, tri_opts) if tri_opts else None,
                "odds_ou": row_odds(r, pair_opts) if pair_opts else None}
    except Exception as e:
        logging.warning("find_fixture failed: %s", e)
        return None


def make_prediction(home_q, away_q):
    prefetch(list(LEAGUE_CODES.values()))
    for name, code in LEAGUE_CODES.items():
        model = get_model(code)
        if model is None:
            continue
        try:
            h, a = model.resolve(home_q), model.resolve(away_q)
        except ValueError:
            continue
        if h != a:
            res = model.markets(h, a)
            res["league"] = name
            return res
    return None


# ================= MESSAGES (numbers come from Python only) =================
HELP = (
    "I can build accumulators and predict matches.\n\n"
    "Accumulator: say 'acca', '2 odds', '10 odds', '50 odds' or '1000 odds'. "
    "Add a day: 'acca 10 odds tomorrow', 'today', 'this weekend', 'saturday'. "
    "Or use /acca 10.\n"
    "Match: 'Arsenal vs Chelsea' or /predict Arsenal vs Chelsea.\n"
    "/stats shows how my logged picks actually did. /leagues lists the leagues I cover.\n\n"
    "These are statistical estimates, not guarantees. Never stake money you "
    "can't afford to lose."
)


def format_acca(acc):
    if acc.get("error") == "no_legs":
        return "I couldn't find any usable legs right now."
    if acc.get("error") == "unreachable":
        return (f"I can't build {acc['target_odds']:g} odds from the {acc['matches']} matches "
                f"available. The most I can reach is about {acc['max_odds']:g}. "
                "Try a smaller target.")
    p = acc["combined_probability"]
    one_in = max(1, round(1 / p)) if p > 0 else 0
    lines = [f"Accumulator, about {acc['target_odds']:g} odds",
             f"Total odds: {acc['total_odds']:.2f}",
             f"Chance every leg wins: {p:.2%} (about 1 in {one_in:,})", ""]
    for i, leg in enumerate(acc["legs"], 1):
        est = " (est.)" if leg["odds_source"] == "estimated" else ""
        lines.append(f"{i}. {leg['pick']} - {leg['fixture']}")
        ko = f" | {leg['kickoff']}" if leg.get("kickoff") else ""
        lines.append(f"   chance {leg['p']:.0%} | odds {leg['odds']:.2f}{est}{ko}")
    lines.append("")
    lines += acc["warnings"]
    lines.append("Based mostly on bookmaker odds. Estimates, not guarantees.")
    return "\n".join(lines)


def format_prediction(res):
    if res.get("market_blend"):
        note = ("Mostly bookmaker odds (the model's share is small). Injuries and lineups "
                "are not included.")
    else:
        note = ("Model only: no bookmaker odds were found for this match, and the model alone "
                "was less accurate than bookmaker odds in tests. Injuries and lineups are not included.")

    def fo(k):
        return f"{1 / res[k]:.2f}" if res[k] > 0 else "-"
    return (
        f"{res['fixture']} ({res.get('league', '')})\n"
        f"Expected goals: {res['home_xg']} - {res['away_xg']}\n\n"
        f"Home win {res['home_win']:.0%} | Draw {res['draw']:.0%} | Away win {res['away_win']:.0%}\n"
        f"Home or draw {res['1X']:.0%} | Draw or away {res['X2']:.0%}\n"
        f"Over 1.5 goals {res['over_1_5']:.0%} | Over 2.5 goals {res['over_2_5']:.0%}\n"
        f"Both teams score {res['btts']:.0%}\n\n"
        f"Fair odds: home {fo('home_win')}, draw {fo('draw')}, away {fo('away_win')}\n"
        + note
    )


# ================= UNDERSTANDING MESSAGES =================
ACCA_WORDS = ("acca", "accumulator", "accy", "accum", "slip", "ticket", "parlay", "odds")


def parse_target(text):
    t = text.lower().replace(",", "")
    m = (re.search(r"(\d+(?:\.\d+)?)\s*(k)?\s*(?:odds?|x)\b", t)
         or re.search(r"(?:odds?|acca|accumulator|accy|slip|ticket|parlay)\D{0,12}?(\d+(?:\.\d+)?)\s*(k)?\b", t)
         or re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(k)?\s*", t))
    if not m:
        return None
    return float(m.group(1)) * (1000 if m.group(2) else 1)


def parse_match(text):
    m = re.search(r"(.+?)\s+(?:vs\.?|v\.?|versus|against)\s+(.+)", text.strip(), flags=re.I)
    if not m:
        return None
    home = m.group(1).strip()
    home = re.sub(r"^(?:please\s+)?(?:what(?:'s|\s+is|\s+are)?\s+(?:the\s+)?)?"
                  r"(?:odds|chances?|prediction|tip)\s+(?:for|of|on)\s+", "", home, flags=re.I)
    home = re.sub(r"^(?:please\s+)?(?:predict|prediction|tip|pick|analy[sz]e|who wins|who will win"
                  r"|what about|how about)[:\s]+", "", home, flags=re.I)
    away = re.sub(r"[?!.]+$", "", m.group(2).strip())
    return (home.strip(), away.strip()) if home.strip() and away.strip() else None


async def classify_with_gemini(text):
    prompt = (
        "You read messages sent to a football odds bot. Reply with JSON only:\n"
        '{"intent": "acca" or "predict" or "help" or "other", "target_odds": number or null, '
        '"home": string or null, "away": string or null}\n'
        "acca = the user wants an accumulator (multi-bet slip), maybe with a total odds target "
        "such as '2 odds', '50 odds' or '1k odds' (1k means 1000). "
        "predict = the user wants the chances for one match; fill in home and away team names. "
        "help = the user asks what the bot can do. other = anything else.\n"
        f"Message: {text}"
    )
    resp = await client.aio.models.generate_content(
        model=GEMINI_MODEL, contents=prompt, config={"response_mime_type": "application/json"})
    raw = re.sub(r"^```(?:json)?|```$", "", resp.text.strip(), flags=re.M).strip()
    data = json.loads(raw)
    intent = data.get("intent")
    if intent not in ("acca", "predict", "help", "other"):
        return {"intent": "other"}
    out = {"intent": intent}
    try:
        out["target"] = float(data["target_odds"]) if data.get("target_odds") is not None else None
    except (TypeError, ValueError):
        out["target"] = None
    out["home"], out["away"] = data.get("home"), data.get("away")
    return out


async def understand(text):
    low = text.lower()
    target, match = parse_target(text), parse_match(text)
    if match and target is None:
        return {"intent": "predict", "home": match[0], "away": match[1]}
    if any(w in low for w in ACCA_WORDS):
        return {"intent": "acca", "target": target}
    try:
        return await classify_with_gemini(text)
    except Exception as e:
        logging.error("Gemini parse error: %s", e)
        return {"intent": "other"}


# ================= DATES =================
from datetime import date, timedelta
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def parse_when(text):
    """Understands today, tonight, tomorrow, this weekend, a weekday name, 'next N days'.
    Returns (start_date, end_date, label) or None."""
    low = text.lower()
    today = date.today()
    wd = today.weekday()
    if "tomorrow" in low:
        d = today + timedelta(days=1)
        return d, d, "tomorrow"
    if "today" in low or "tonight" in low:
        return today, today, "today"
    if "weekend" in low:
        if wd == 6:
            return today, today, "this weekend"
        start = today if wd == 5 else today + timedelta(days=5 - wd)
        return start, start + timedelta(days=1), "this weekend"
    m = re.search(r"next\s+(\d+)\s+days?", low)
    if m:
        n = min(int(m.group(1)), 14)
        return today, today + timedelta(days=n), f"the next {n} days"
    for i, name in enumerate(WEEKDAYS):
        if name in low:
            d = today + timedelta(days=(i - wd) % 7)
            if d == today and "next" in low:
                d += timedelta(days=7)
            return d, d, name.capitalize()
    return None


# ================= PREDICTION LOG (SQLite) =================
import sqlite3
import uuid

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:   # some phone editors run the file without __file__
    _HERE = os.getcwd()
# On a host, set DB_PATH to a folder that survives restarts (a persistent disk or volume).
DB_PATH = os.environ.get("DB_PATH") or os.path.join(_HERE, "football_bot.db")


def db():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS legs (
        id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, kind TEXT, acca_id TEXT, fp TEXT,
        match_date TEXT, div TEXT, home TEXT, away TEXT, mkey TEXT, pick TEXT,
        p REAL, odds REAL, odds_source TEXT, result INTEGER)""")
    return con


def leg_hit(key, hg, ag):
    return {"home": hg > ag, "away": hg < ag, "draw": hg == ag, "1X": hg >= ag, "X2": hg <= ag,
            "o15": hg + ag >= 2, "o25": hg + ag >= 3, "btts": hg > 0 and ag > 0}[key]


def log_acca(acc):
    legs = acc.get("legs") or []
    if not legs:
        return None
    fp = "|".join(sorted(f"{l['fixture']}:{l['key']}:{l['date']}" for l in legs))
    con = db()
    try:
        if con.execute("SELECT 1 FROM legs WHERE kind='acca' AND fp=? AND result IS NULL LIMIT 1",
                       (fp,)).fetchone():
            return None   # same slip already logged and still waiting for results
        acca_id = uuid.uuid4().hex[:8]
        now = datetime.now().isoformat(timespec="seconds")
        for l in legs:
            home, away = l["teams"]
            con.execute("INSERT INTO legs (created_at, kind, acca_id, fp, match_date, div, home, away,"
                        " mkey, pick, p, odds, odds_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, "acca", acca_id, fp, l["date"], l["div"], home, away, l["key"],
                         l["pick"], l["p"], l["odds"], l["odds_source"]))
        con.commit()
        return acca_id
    finally:
        con.close()


def log_single(res, fx):
    """Log the model-only probabilities of a /predict for a match with a known date."""
    h, a = fx["home"], fx["away"]
    items = [("home", "home_win", f"{h} to win"), ("draw", "draw", "Draw"),
             ("away", "away_win", f"{a} to win"), ("o15", "over_1_5", "Over 1.5 goals"),
             ("o25", "over_2_5", "Over 2.5 goals"), ("btts", "btts", "Both teams score")]
    con = db()
    try:
        if con.execute("SELECT 1 FROM legs WHERE kind='single' AND home=? AND away=? AND match_date=?"
                       " AND result IS NULL LIMIT 1", (h, a, fx["date"])).fetchone():
            return
        now = datetime.now().isoformat(timespec="seconds")
        for key, field, label in items:
            con.execute("INSERT INTO legs (created_at, kind, match_date, div, home, away, mkey, pick, p)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (now, "single", fx["date"], fx["div"], h, a, key, label, res[field]))
        con.commit()
    finally:
        con.close()


def settle_pending():
    """Look up finished matches in the results files and mark logged legs hit or missed."""
    con = db()
    try:
        rows = con.execute("SELECT id, match_date, div, home, away, mkey FROM legs WHERE result IS NULL"
                           " AND match_date IS NOT NULL AND match_date <= ?",
                           (date.today().isoformat(),)).fetchall()
        by_div = {}
        for r in rows:
            by_div.setdefault(r[2], []).append(r)
        settled = 0
        for code, items in by_div.items():
            try:
                df = load_results(code, n_seasons=2)
            except Exception as e:
                logging.warning("settle: could not load %s: %s", code, e)
                continue
            for rid, md, _, home, away, mkey in items:
                d0 = pd.Timestamp(md)
                m = df[(df["HomeTeam"] == home) & (df["AwayTeam"] == away)
                       & ((df["Date"] - d0).abs() <= pd.Timedelta(days=1))]
                if m.empty:
                    continue   # not in the results file yet
                hg, ag = int(m.iloc[0]["FTHG"]), int(m.iloc[0]["FTAG"])
                con.execute("UPDATE legs SET result=? WHERE id=?", (int(leg_hit(mkey, hg, ag)), rid))
                settled += 1
        con.commit()
        return settled
    finally:
        con.close()


def stats_text():
    con = db()
    try:
        pending = con.execute("SELECT COUNT(*) FROM legs WHERE result IS NULL").fetchone()[0]
        lines = ["Prediction log"]
        bands = [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
        any_data = False
        for kind, label in (("acca", "Accumulator legs"), ("single", "Match predictions (model only)")):
            rows = con.execute("SELECT p, result FROM legs WHERE kind=? AND result IS NOT NULL",
                               (kind,)).fetchall()
            if not rows:
                continue
            any_data = True
            ps = np.array([r[0] for r in rows])
            hs = np.array([r[1] for r in rows], dtype=float)
            lines += ["", f"{label}: {len(rows)} settled",
                      f"Predicted {ps.mean():.1%} | actually hit {hs.mean():.1%}"]
            for lo, hi in bands:
                mask = (ps >= lo) & (ps < hi)
                if mask.sum() >= 5:
                    lines.append(f"  {lo:.0%}-{min(hi, 1):.0%}: predicted {ps[mask].mean():.0%}, "
                                 f"hit {hs[mask].mean():.0%} ({int(mask.sum())})")
        rows = con.execute("SELECT acca_id, p, result FROM legs WHERE kind='acca'").fetchall()
        slips = {}
        for aid, p, r in rows:
            slips.setdefault(aid, []).append((p, r))
        done = [v for v in slips.values() if all(r is not None for _, r in v)]
        if done:
            won = sum(all(r == 1 for _, r in v) for v in done)
            exp = sum(float(np.prod([p for p, _ in v])) for v in done)
            lines += ["", f"Accumulators settled: {len(done)} | won {won} | model expected about {exp:.1f}"]
        if not any_data and not done:
            lines += ["", "Nothing settled yet. Picks are logged when you ask for an accumulator or a "
                          "/predict for a listed fixture, and settle after the match appears in the "
                          "results file (updated about twice a week)."]
        last = con.execute("SELECT acca_id, MIN(created_at) FROM legs WHERE kind='acca' "
                           "GROUP BY acca_id ORDER BY MIN(created_at) DESC LIMIT 3").fetchall()
        if last:
            lines += ["", "Last slips"]
        for aid, created in last:
            legs = con.execute("SELECT pick, home, away, result FROM legs WHERE acca_id=? ORDER BY id",
                               (aid,)).fetchall()
            done = [l for l in legs if l[3] is not None]
            hits = sum(1 for l in done if l[3] == 1)
            state = "waiting" if len(done) < len(legs) else ("WON" if hits == len(legs) else "lost")
            lines.append(f"{created[:10]}: {hits}/{len(legs)} legs hit ({state})")
            for pick, h, a, r in legs:
                mark = "\u2705" if r == 1 else ("\u274C" if r == 0 else "\u23F3")
                lines.append(f"  {mark} {pick} ({h} vs {a})")
        lines += ["", f"Waiting for results: {pending} legs"]
        return "\n".join(lines)
    finally:
        con.close()


# ================= LOG BACKUP IN A PRIVATE GITHUB REPO (for free hosts that wipe files) =================
import base64
import urllib.error

GH_TOKEN = os.environ.get("GITHUB_TOKEN")
GH_REPO = os.environ.get("GITHUB_DATA_REPO")            # "your-name/your-private-data-repo"
GH_PATH = os.environ.get("GITHUB_DATA_PATH", "football_bot.db")
GH_BRANCH = os.environ.get("GITHUB_DATA_BRANCH", "main")
_SYNC_LOCK = threading.Lock()


def _gh_request(method, payload=None):
    url = f"https://api.github.com/repos/{GH_REPO}/contents/{GH_PATH}"
    if method == "GET":
        url += f"?ref={GH_BRANCH}"
    req = urllib.request.Request(
        url, method=method, data=json.dumps(payload).encode() if payload else None,
        headers={"Authorization": f"Bearer {GH_TOKEN}", "Accept": "application/vnd.github+json",
                 "User-Agent": "football-bot", "X-GitHub-Api-Version": "2022-11-28"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def restore_db():
    """Download the saved log at startup. Does nothing if the backup is not configured."""
    if not (GH_TOKEN and GH_REPO):
        return False
    try:
        info = _gh_request("GET")
        if not info.get("content"):
            logging.warning("Backup file is too large for the contents API; starting without it.")
            return False
        os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
        with open(DB_PATH, "wb") as f:
            f.write(base64.b64decode(info["content"]))
        logging.info("Prediction log restored from GitHub.")
        return True
    except urllib.error.HTTPError as e:
        if e.code != 404:
            logging.warning("Could not restore the log: %s", e)
    except Exception as e:
        logging.warning("Could not restore the log: %s", e)
    return False


def _err_text(e):
    if isinstance(e, urllib.error.HTTPError):
        try:
            msg = json.loads(e.read().decode()).get("message", "")
        except Exception:
            msg = ""
        return f"GitHub said {e.code}: {msg}".strip()
    return str(e)


def push_db():
    """Upload the log after a change. Never raises. Returns (ok, message)."""
    if not (GH_TOKEN and GH_REPO):
        return False, "GITHUB_TOKEN or GITHUB_DATA_REPO is not set in the host settings."
    if not os.path.exists(DB_PATH):
        return False, "There is no log yet. Ask for an accumulator first."
    with _SYNC_LOCK:
        try:
            with open(DB_PATH, "rb") as f:
                raw = f.read()
            sha = None
            try:
                sha = _gh_request("GET").get("sha")
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            payload = {"message": "update prediction log", "branch": GH_BRANCH,
                       "content": base64.b64encode(raw).decode()}
            if sha:
                payload["sha"] = sha
            _gh_request("PUT", payload)
            return True, f"Saved {GH_PATH} to {GH_REPO}."
        except Exception as e:
            text = _err_text(e)
            logging.warning("Could not back up the log: %s", text)
            return False, text


def reset_log():
    con = db()
    try:
        n = con.execute("SELECT COUNT(*) FROM legs").fetchone()[0]
        con.execute("DELETE FROM legs")
        con.commit()
        return n
    finally:
        con.close()


# ================= TELEGRAM =================
async def send_long(update, text):
    for i in range(0, len(text), 4000):
        await update.message.reply_text(text[i:i + 4000])


async def run_acca(update, target, when=None):
    target = 10.0 if target is None else target
    note = ""
    if target < MIN_TARGET:
        note, target = f"The lowest target I use is {MIN_TARGET:g} odds.\n\n", MIN_TARGET
    elif target > MAX_TARGET:
        note, target = f"The highest target I use is {MAX_TARGET:g} odds.\n\n", MAX_TARGET
    await update.message.reply_text("Working on it. The first request can take a minute...")
    try:
        cands, msg = await asyncio.to_thread(compute_candidates, when)
        if cands is None:
            await update.message.reply_text(msg)
            return
        acc = await asyncio.to_thread(build_accumulator, cands, target)
        text = note + format_acca(acc)
        if "legs" in acc:
            try:
                if await asyncio.to_thread(log_acca, acc):
                    text += "\nSaved to the prediction log. /stats shows how past picks did."
                    await asyncio.to_thread(push_db)
            except Exception:
                logging.exception("could not write the log")
        await send_long(update, text)
    except Exception as e:
        logging.exception("acca failed")
        await update.message.reply_text(f"Something went wrong: {e}")


async def run_predict(update, home, away):
    await update.message.reply_text("Working on it. The first request can take a minute...")
    try:
        res = await asyncio.to_thread(make_prediction, home, away)
        if res is None:
            await update.message.reply_text(
                "I couldn't find both teams in my leagues (see /leagues). Check the spelling.")
            return
        fx = await asyncio.to_thread(find_fixture, res)
        if fx and fx.get("odds_hda"):
            res = blend_with_market(res, fx["odds_hda"], fx.get("odds_ou"))
        text = format_prediction(res)
        if fx:
            text += f"\n\nNext listed fixture: {fx['kickoff']}. Saved to the prediction log."
            try:
                await asyncio.to_thread(log_single, res, fx)
                await asyncio.to_thread(push_db)
            except Exception:
                logging.exception("could not write the log")
        else:
            text += "\n\nNo upcoming fixture listed for this pairing, so it was not logged."
        await update.message.reply_text(text)
    except Exception as e:
        logging.exception("predict failed")
        await update.message.reply_text(f"Something went wrong: {e}")


async def leagues_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    names = [n for n, c in LEAGUE_CODES.items() if c not in ACCA_EXCLUDE]
    text = "Leagues I cover:\n" + ", ".join(names)
    if ACCA_EXCLUDE:
        text += "\n\nLeft out of accumulators: " + ", ".join(CODE_TO_NAME.get(c, c) for c in sorted(ACCA_EXCLUDE))
    await update.message.reply_text(text)


async def backup_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    ok, msg = await asyncio.to_thread(push_db)
    await update.message.reply_text(("Backup OK. " if ok else "Backup failed. ") + msg)


async def resetlog_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if " ".join(context.args).strip().lower() != "yes":
        await update.message.reply_text("This deletes the whole prediction log. "
                                        "To go ahead, send: /resetlog yes")
        return
    n = await asyncio.to_thread(reset_log)
    await asyncio.to_thread(push_db)
    await update.message.reply_text(f"Prediction log cleared ({n} rows removed).")


async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Checking for new results...")
    try:
        if await asyncio.to_thread(settle_pending):
            await asyncio.to_thread(push_db)
        await send_long(update, await asyncio.to_thread(stats_text))
    except Exception as e:
        logging.exception("stats failed")
        await update.message.reply_text(f"Something went wrong: {e}")


async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP)


async def acca_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = " ".join(context.args)
    await run_acca(update, parse_target(args), parse_when(args))


async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    match = parse_match(" ".join(context.args))
    if not match:
        await update.message.reply_text("Format like: /predict Arsenal vs Chelsea")
        return
    await run_predict(update, *match)


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    info = await understand(update.message.text)
    intent = info.get("intent")
    if intent == "acca":
        await run_acca(update, info.get("target"), parse_when(update.message.text))
    elif intent == "predict" and info.get("home") and info.get("away"):
        await run_predict(update, info["home"], info["away"])
    else:
        await update.message.reply_text(HELP)


if __name__ == "__main__":
    if "PASTE" in TELEGRAM_TOKEN or "PASTE" in GEMINI_API_KEY:
        raise SystemExit("Put your NEW Telegram token and Gemini key at the top of the file first.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    restore_db()
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], start_cmd))
    app.add_handler(CommandHandler("acca", acca_cmd))
    app.add_handler(CommandHandler("predict", predict_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("leagues", leagues_cmd))
    app.add_handler(CommandHandler("backup", backup_cmd))
    app.add_handler(CommandHandler("resetlog", resetlog_cmd))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), message_handler))
    threading.Thread(target=lambda: prefetch(list(LEAGUE_CODES.values())), daemon=True).start()

    base_url = os.environ.get("WEBHOOK_URL") or os.environ.get("RENDER_EXTERNAL_URL")
    if base_url:   # hosted: Telegram calls us, so a sleeping free host wakes up on your message
        import hashlib
        path = hashlib.sha256(("path:" + TELEGRAM_TOKEN).encode()).hexdigest()[:40]
        secret = hashlib.sha256(("header:" + TELEGRAM_TOKEN).encode()).hexdigest()[:40]
        print("Bot starting in webhook mode.")
        app.run_webhook(listen="0.0.0.0", port=int(os.environ.get("PORT", "10000")),
                        url_path=path, webhook_url=f"{base_url.rstrip('/')}/{path}",
                        secret_token=secret)
    else:          # on your phone or a computer
        print("Bot is running (loading league data in the background). Press stop to quit.")
        app.run_polling()
