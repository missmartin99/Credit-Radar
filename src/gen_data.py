"""Synthetic Paytm-style payments panel (users x days) with gali-level shocks.

REPLACE THIS with real data: the rest of the pipeline only needs the panel schema in README.md.
Everything here is made up. Locality coordinates are approximate and for illustration only.

Usage: python src/gen_data.py [out_dir]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from events import intensity  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "data"
SEED = 7
START, END = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-11-18")
PEER_GALI = float(os.environ.get("PEER_GALI", 9.0))
PEER_LOC = float(os.environ.get("PEER_LOC", 3.0))
BASE_LOGIT = float(os.environ.get("BASE_LOGIT", -7.3))  # calibrates overall loan-taking rate

# name, code, lat, lon, income tier, merchant share, gig share
LOCALITIES = [
    ("Malviya Nagar", "MN", 28.5355, 77.2090, 1.30, 0.30, 0.15),
    ("Saket", "SK", 28.5245, 77.2066, 1.55, 0.15, 0.10),
    ("Hauz Khas", "HK", 28.5494, 77.2001, 1.60, 0.20, 0.10),
    ("Lajpat Nagar", "LN", 28.5677, 77.2432, 1.15, 0.40, 0.15),
    ("Kalkaji", "KJ", 28.5405, 77.2588, 1.00, 0.30, 0.20),
    ("Nehru Place", "NP", 28.5491, 77.2533, 1.25, 0.35, 0.15),
    ("Karol Bagh", "KB", 28.6514, 77.1907, 1.10, 0.45, 0.15),
    ("Rajouri Garden", "RG", 28.6492, 77.1226, 1.20, 0.35, 0.12),
    ("Janakpuri", "JP", 28.6219, 77.0878, 1.20, 0.20, 0.15),
    ("Dwarka", "DW", 28.5921, 77.0460, 1.25, 0.12, 0.15),
    ("Laxmi Nagar", "LX", 28.6304, 77.2773, 0.90, 0.35, 0.25),
    ("Mayur Vihar", "MV", 28.6089, 77.2954, 1.10, 0.20, 0.15),
    ("Preet Vihar", "PV", 28.6414, 77.2951, 1.10, 0.30, 0.12),
    ("Rohini", "RH", 28.7160, 77.1120, 1.00, 0.20, 0.20),
    ("Pitampura", "PT", 28.7000, 77.1314, 1.15, 0.25, 0.12),
    ("Chandni Chowk", "CC", 28.6506, 77.2303, 0.95, 0.60, 0.15),
    ("Sadar Bazar", "SB", 28.6575, 77.2106, 0.90, 0.60, 0.20),
    ("Okhla", "OK", 28.5459, 77.2732, 0.85, 0.25, 0.30),
    ("Jamia Nagar", "JN", 28.5619, 77.2800, 0.80, 0.30, 0.25),
    ("Govindpuri", "GP", 28.5330, 77.2640, 0.80, 0.25, 0.30),
    ("Sangam Vihar", "SV", 28.4970, 77.2380, 0.70, 0.20, 0.40),
    ("Mehrauli", "ME", 28.5200, 77.1850, 0.90, 0.30, 0.25),
    ("Vasant Kunj", "VK", 28.5200, 77.1580, 1.50, 0.10, 0.10),
    ("Munirka", "MU", 28.5575, 77.1740, 1.00, 0.25, 0.25),
    ("Uttam Nagar", "UN", 28.6200, 77.0550, 0.80, 0.20, 0.35),
    ("Tilak Nagar", "TN", 28.6400, 77.0970, 1.00, 0.30, 0.20),
    ("Shahdara", "SH", 28.6700, 77.2900, 0.85, 0.30, 0.30),
    ("Seelampur", "SE", 28.6700, 77.2700, 0.75, 0.30, 0.35),
    ("Burari", "BU", 28.7500, 77.2000, 0.70, 0.20, 0.40),
    ("Narela", "NA", 28.8528, 77.0929, 0.70, 0.20, 0.40),
]
GALI_OFFSETS = [(0.003, 0.003), (0.003, -0.003), (-0.003, 0.003), (-0.003, -0.003)]
GALI_LETTERS = "ABCD"

rng = np.random.default_rng(SEED)
dates = pd.date_range(START, END)
T = len(dates)

# ---------------------------------------------------------------- geography + users
cells, users = [], []
for li, (name, code, lat, lon, tier, ms, gs) in enumerate(LOCALITIES):
    for gi, (dlat, dlon) in enumerate(GALI_OFFSETS):
        cid = len(cells)
        n = max(40, int(rng.poisson(70)))
        cells.append(dict(cell=cid, cell_code=f"{code}-{GALI_LETTERS[gi]}", loc=li, loc_name=name,
                          name=f"{name} · Lane {GALI_LETTERS[gi]}", lat=lat + dlat, lon=lon + dlon, n=n))
        seg = rng.choice(3, size=n, p=[1 - ms - gs, gs, ms])  # 0 salaried, 1 gig, 2 merchant
        for s in seg:
            users.append((cid, li, tier, s))
users = pd.DataFrame(users, columns=["cell", "loc", "loc_tier", "seg"])
N = len(users)
users["user_id"] = [f"U-{i:05d}" for i in range(N)]
cells = pd.DataFrame(cells)
seg, cell, loc, tier = users.seg.values, users.cell.values, users["loc"].values, users.loc_tier.values

# ---------------------------------------------------------------- static user attributes
income = np.where(seg == 0, rng.lognormal(np.log(24000), 0.40, N) * tier,
                  np.where(seg == 1, rng.lognormal(np.log(15000), 0.35, N) * tier ** 0.7,
                           rng.lognormal(np.log(90000), 0.50, N) * tier ** 0.6))  # merchants: monthly sales
spend_mult = rng.normal(1.04, 0.13, N).clip(0.7, 1.5)
salary_day = rng.integers(1, 8, N)
rent_day = rng.integers(1, 6, N)
bill_day = rng.integers(8, 28, N)
has_rent = rng.random(N) < np.where(seg == 2, 0.9, 0.65)
rent_amt = has_rent * income * np.where(seg == 0, 0.27, np.where(seg == 1, 0.20, 0.06)) * spend_mult
bill_amt = income * 0.05 * spend_mult
base_daily = np.where(seg == 0, 0.55, np.where(seg == 1, 0.60, 0.15)) * income / 30 * spend_mult
lump_rate = 0.03
lump_size = np.where(seg == 2, 0.04, 0.12) * income
propensity = rng.normal(0, 0.6, N)
has_balance = rng.random(N) < 0.55
lam_txn = np.where(seg == 0, 2.2, np.where(seg == 1, 3.0, 12.0))

# synthetic operational flags for the offer engine
users["marketing_consent"] = rng.random(N) < 0.72
users["dnd"] = rng.random(N) < 0.06
users["nudges_30d"] = rng.poisson(0.7, N)
users["has_balance"] = has_balance

# ---------------------------------------------------------------- shocks (area-level, with contagion delay)
K = 3
sh_start = np.full((N, K), 10 ** 6)
sh_end = np.full((N, K), 10 ** 6)
sh_sev = np.zeros((N, K))
sh_kind = np.zeros((N, K), dtype=int)  # 0 market disruption, 1 cost spike
sh_fill = np.zeros(N, dtype=int)
n_shocks = 0
for li in range(len(LOCALITIES)):
    for _ in range(rng.poisson(4.0)):
        scope_loc = rng.random() < 0.30
        gl = cells.index[cells["loc"] == li].values
        members = np.where(np.isin(cell, gl if scope_loc else [rng.choice(gl)]))[0]
        start = int(rng.integers(30, T - 40))
        dur = int(rng.integers(14, 29))
        sev = rng.uniform(0.4, 0.85)
        kind = int(rng.random() < 0.4)
        hit = members[rng.random(len(members)) < 0.8]
        delay = rng.integers(0, 11, len(hit))  # peers get hit first -> area signal leads individual stress
        slot = sh_fill[hit]
        ok = slot < K
        hit, delay, slot = hit[ok], delay[ok], slot[ok]
        sh_start[hit, slot] = start + delay
        sh_end[hit, slot] = start + delay + dur
        sh_sev[hit, slot] = sev
        sh_kind[hit, slot] = kind
        sh_fill[hit] += 1
        n_shocks += 1

# ---------------------------------------------------------------- simulation state
bal = base_daily * 30 * rng.uniform(0.3, 1.2, N)
out_ew = base_daily + 1.0
short_ew = np.zeros(N)
arrears = np.zeros(N)
last_loan = np.full(N, -999)
emi_next = np.full(N, 10 ** 6)
emi_left = np.zeros(N, dtype=int)
emi_amt = np.zeros(N)
sup_next = rng.integers(0, 8, N)
in_shock_prev = np.zeros(N, dtype=bool)
DOW_M = np.array([0.95, 0.95, 0.95, 1.0, 1.05, 1.15, 1.10])

names = ["inflow", "outflow", "ntxn", "fails", "p2p", "loan", "emi", "bal", "shock"]
P = {k: np.zeros((T, N), dtype=np.float32) for k in names}

sigmoid = lambda z: 1 / (1 + np.exp(-z))

for t, date in enumerate(dates):
    dom, dow = date.day, date.dayofweek
    I_restock, I_sales, I_cons = intensity(date, 20), intensity(date, 8), intensity(date, 10)
    tt = np.full(N, t)

    # ---- shock state today
    active = (tt[:, None] >= sh_start) & (tt[:, None] < sh_end)
    mk_sev = np.where(active & (sh_kind == 0), sh_sev, 0).max(axis=1)
    cost_now = (tt[:, None] == sh_start) & (sh_kind == 1)
    cost_lump = (np.where(cost_now, sh_sev, 0).max(axis=1)) * 0.5 * income
    in_shock = active.any(axis=1)

    # ---- inflows
    inflow = np.zeros(N)
    m0, m1, m2 = seg == 0, seg == 1, seg == 2
    inflow[m0] = (dom == salary_day[m0]) * income[m0] * rng.normal(1, 0.03, m0.sum())
    work = rng.random(N) < 0.82
    inflow[m1] = (work * income / 30 / 0.82 * rng.lognormal(-0.08, 0.4, N) * (1 - 0.5 * mk_sev))[m1]
    inflow[m2] = (income / 30 * DOW_M[dow] * (1 + 0.6 * I_sales) * rng.lognormal(-0.02, 0.2, N)
                  * (1 - 0.8 * mk_sev))[m2]
    inflow += (rng.random(N) < 0.05) * rng.lognormal(np.log(300), 0.7, N)  # refunds / cashback

    # ---- loan decision (uses state at end of yesterday)
    runway = bal / out_ew
    # ASSUMPTION (flagged in README): word-of-mouth contagion in borrowing, strongest within a lane (gali)
    recent = ((t - last_loan) < 14).astype(float)
    peer_gali = (np.bincount(cell, weights=recent) / np.bincount(cell))[cell]
    peer_loc = (np.bincount(loc, weights=recent) / np.bincount(loc))[loc]
    logit = (PEER_GALI * np.minimum(peer_gali, 0.12) + PEER_LOC * np.minimum(peer_loc, 0.12) + BASE_LOGIT + 3.0 * (short_ew > 0.2) + 1.8 * (runway < 7) + 1.2 * (runway < 3)
             + 0.8 * in_shock_prev + 1.2 * m2 * (I_restock > 0.3) + 0.6 * (arrears > 0) + propensity
             - 1.5 * ((t - last_loan) < 30))
    take = rng.random(N) < sigmoid(logit)
    loan_amt = np.where(take, np.clip(income * rng.lognormal(np.log(np.where(m2, 0.6, 0.4)), 0.4, N),
                                      3000, 150000), 0)
    loan_amt = np.round(loan_amt / 500) * 500
    inflow_total = inflow + loan_amt
    last_loan = np.where(take, t, last_loan)
    emi_left = np.where(take, 6, emi_left)
    emi_amt = np.where(take, loan_amt * 1.15 / 6, emi_amt)
    emi_next = np.where(take, t + 30, emi_next)

    # ---- p2p borrowing from friends when stressed
    stressed = (runway < 5) | (short_ew > 0.2)
    p2p = ((rng.random(N) < np.where(stressed, 0.10, 0.01)) * rng.lognormal(np.log(1500), 0.6, N))
    inflow_total += p2p

    # ---- desired outflow
    cons_mult = 1 + 0.35 * I_cons * (~m2)
    base = base_daily * rng.lognormal(-0.05, 0.35, N) * cons_mult
    rent = (dom == rent_day) * rent_amt
    bill = (dom == bill_day) * bill_amt
    lump = (rng.random(N) < lump_rate) * lump_size * rng.lognormal(-0.18, 0.6, N)
    emi_due = np.where((t == emi_next) & (emi_left > 0), emi_amt, 0.0)
    emi_left = emi_left - (emi_due > 0)
    emi_next = np.where(emi_due > 0, emi_next + 30, emi_next)
    sup_due = (m2 & (t >= sup_next))
    supplier = sup_due * 0.62 * income / 30 * 7 * (1 + 1.3 * I_restock) * rng.lognormal(0, 0.1, N)
    sup_next = np.where(sup_due, t + rng.integers(6, 9, N), sup_next)

    desired = base + rent + bill + lump + cost_lump + emi_due + supplier + arrears
    avail = bal + inflow_total
    paid = np.minimum(desired, avail)
    shortfall = desired - paid
    short_flag = shortfall > 0.1 * np.maximum(desired, 1)
    fails = short_flag * (1 + rng.poisson(0.6, N))
    arrears = 0.6 * shortfall
    bal = avail - paid
    out_ew = 0.9 * out_ew + 0.1 * np.maximum(paid, 1)
    short_ew = 0.5 * short_ew + 0.5 * short_flag
    in_shock_prev = in_shock

    ratio = paid / np.maximum(desired, 1)
    P["inflow"][t] = inflow + p2p  # excluding loan disbursal
    P["outflow"][t] = paid
    P["ntxn"][t] = rng.poisson(lam_txn * (1 + 0.3 * I_cons)) * (paid > 0)
    P["fails"][t] = fails
    P["p2p"][t] = p2p
    P["loan"][t] = loan_amt
    P["emi"][t] = emi_due * ratio
    P["bal"][t] = np.where(has_balance, bal, np.nan)
    P["shock"][t] = in_shock  # ground truth, diagnostics only (never a feature)

os.makedirs(OUT, exist_ok=True)
np.savez_compressed(os.path.join(OUT, "panel.npz"), dates=dates.values.astype("datetime64[D]"), **P)
users.to_csv(os.path.join(OUT, "users.csv"), index=False)
cells.to_csv(os.path.join(OUT, "cells.csv"), index=False)

lo = P["loan"] > 0
print(f"users={N} cells={len(cells)} localities={len(LOCALITIES)} days={T} shocks={n_shocks}")
print(f"loan events/user/yr = {lo.sum() / N * 365 / T:.2f}; share of users with >=1 loan = {lo.any(0).mean():.2%}")
w14 = np.array([lo[t + 1:t + 15].any(0) for t in range(60, T - 15)])
print(f"14-day positive rate = {w14.mean():.2%}  (target 5-12%)")
print(f"insufficient-fund declines per user-week = {P['fails'].sum() / N / T * 7:.2f}")
