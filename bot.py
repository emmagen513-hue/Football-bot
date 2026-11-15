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
MODEL_WEIGHT = 0.25    # 25% model, 75% bookmaker
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
                  "date": c.get("date"), "div": c.get("div")} for c in chosen],
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
        if time.time() 