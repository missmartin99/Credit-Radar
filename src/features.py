"""Feature pipeline: user-level behaviour + area (locality / gali) peer signals.

Input  : panel (T x N arrays), users DataFrame (cell, loc, seg, ...). See README for the schema.
Output : one row per (snapshot day, user) with features and, when available, the 14-day label.

Leakage rules: every feature at snapshot day t uses data up to and including day t only; the label looks at
days t+1..t+14. Area features are leave-one-out (a user's own behaviour is excluded from their area average).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from events import next_event  # noqa: E402

HORIZON = 14


def _cs(x):
    return np.vstack([np.zeros((1, x.shape[1]), dtype=np.float64), np.cumsum(x, axis=0, dtype=np.float64)])


def roll(cs, t, w):
    return cs[t + 1] - cs[max(t + 1 - w, 0)]


def loo_mean(x, g, G, weight=None):
    """Leave-one-out group mean of x over group g (optionally only over members with weight==1)."""
    w = np.ones_like(x) if weight is None else weight.astype(float)
    s = np.bincount(g, weights=x * w, minlength=G)
    n = np.bincount(g, weights=w, minlength=G)
    return (s[g] - x * w) / np.maximum(n[g] - w, 1)


USER_FEATS = ["inflow_7", "inflow_30", "outflow_7", "outflow_30", "in_trend", "out_trend", "net_30_ratio",
              "fails_7", "fails_30", "txn_trend", "p2p_cnt_14", "p2p_amt_ratio", "foir", "loans_90",
              "days_since_loan", "inflow_cv_60", "bal_now", "bal_runway", "bal_min14_ratio", "dom", "seg",
              "has_balance", "days_to_event", "event_weight"]
AREA_BASE = ["fail_share", "in_trend", "merch_in_trend", "p2p_share", "loan30_share", "out_trend"]
LOC_FEATS = ["loc_tier"] + [f"loc_{c}" for c in AREA_BASE]
GALI_FEATS = [f"gali_{c}" for c in AREA_BASE]


def build_snapshot(P, users, dates, t):
    N = len(users)
    seg = users.seg.values
    cell = users.cell.values
    loc = users["loc"].values
    G, L = cell.max() + 1, loc.max() + 1
    inflow, outflow, ntxn = P["cs_inflow"], P["cs_outflow"], P["cs_ntxn"]
    f = {}

    in7, in30 = roll(inflow, t, 7), roll(inflow, t, 30)
    out7, out30 = roll(outflow, t, 7), roll(outflow, t, 30)
    f["inflow_7"], f["inflow_30"], f["outflow_7"], f["outflow_30"] = in7, in30, out7, out30
    f["in_trend"] = np.log1p(in7 / 7) - np.log1p(in30 / 30)
    f["out_trend"] = np.log1p(out7 / 7) - np.log1p(out30 / 30)
    f["net_30_ratio"] = (in30 - out30) / (in30 + 1)
    f["fails_7"], f["fails_30"] = roll(P["cs_fails"], t, 7), roll(P["cs_fails"], t, 30)
    f["txn_trend"] = np.log((roll(ntxn, t, 7) / 7 + 0.1) / (roll(ntxn, t, 30) / 30 + 0.1))
    f["p2p_cnt_14"] = roll(P["cs_p2pc"], t, 14)
    f["p2p_amt_ratio"] = roll(P["cs_p2p"], t, 30) / (in30 + 1)
    f["foir"] = roll(P["cs_emi"], t, 30) / (in30 + 1)
    f["loans_90"] = roll(P["cs_loanc"], t, 90)
    last = P["last_loan_idx"][t]
    f["days_since_loan"] = np.minimum(t - last, 365)
    n60 = min(60, t + 1)
    s1, s2 = roll(inflow, t, 60), roll(P["cs_inflow2"], t, 60)
    mean = s1 / n60
    f["inflow_cv_60"] = np.sqrt(np.maximum(s2 / n60 - mean ** 2, 0)) / (mean + 1)

    b = P["bal"]
    bal_now = b[t]
    f["bal_now"] = bal_now
    f["bal_runway"] = bal_now / (out30 / 30 + 1)
    with np.errstate(all="ignore"):
        win = b[max(t - 13, 0):t + 1]
        bmin = np.where(np.isnan(win).all(0), np.nan, np.nanmin(np.where(np.isnan(win), np.inf, win), axis=0))
    f["bal_min14_ratio"] = bmin / (out30 / 30 + 1)
    f["dom"] = np.full(N, dates[t].day, dtype=float)
    f["seg"] = seg.astype(float)
    f["has_balance"] = users.has_balance.values.astype(float)
    name, dte, w = next_event(dates[t])
    f["days_to_event"] = np.full(N, dte, dtype=float)
    f["event_weight"] = np.full(N, w, dtype=float)

    # ---- area signals (leave-one-out so a user never "sees" themselves)
    fail_ind = (f["fails_7"] > 0).astype(float)
    p2p_ind = (f["p2p_cnt_14"] > 0).astype(float)
    loan30 = (f["days_since_loan"] < 30).astype(float)
    is_m = (seg == 2)
    for prefix, g, GG in (("gali", cell, G), ("loc", loc, L)):
        f[f"{prefix}_fail_share"] = loo_mean(fail_ind, g, GG)
        f[f"{prefix}_in_trend"] = loo_mean(f["in_trend"], g, GG)
        f[f"{prefix}_merch_in_trend"] = loo_mean(f["in_trend"], g, GG, weight=is_m)
        f[f"{prefix}_p2p_share"] = loo_mean(p2p_ind, g, GG)
        f[f"{prefix}_loan30_share"] = loo_mean(loan30, g, GG)
        f[f"{prefix}_out_trend"] = loo_mean(f["out_trend"], g, GG)
    f["loc_tier"] = users.loc_tier.values

    df = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    df.insert(0, "t", t)
    df.insert(1, "uidx", np.arange(N))
    return df, name


def prepare_panel(panel):
    """Cumulative sums etc. computed once."""
    P = {"bal": panel["bal"]}
    for k, src in [("inflow", "inflow"), ("outflow", "outflow"), ("ntxn", "ntxn"), ("fails", "fails"),
                   ("p2p", "p2p"), ("emi", "emi")]:
        P["cs_" + k] = _cs(panel[src])
    P["cs_inflow2"] = _cs(panel["inflow"].astype(np.float64) ** 2)
    P["cs_p2pc"] = _cs((panel["p2p"] > 0).astype(np.float64))
    P["cs_loanc"] = _cs((panel["loan"] > 0).astype(np.float64))
    T, N = panel["loan"].shape
    idx = np.where(panel["loan"] > 0, np.arange(T)[:, None], -999)
    P["last_loan_idx"] = np.maximum.accumulate(idx, axis=0)
    return P


def build_dataset(panel, users, dates, snapshots, with_labels=True):
    P = prepare_panel(panel)
    loan = panel["loan"] > 0
    frames = []
    for t in snapshots:
        df, _ = build_snapshot(P, users, dates, t)
        if with_labels:
            win = loan[t + 1:t + 1 + HORIZON]
            df["y"] = win.any(0).astype(np.int8)
            first = np.where(win.any(0), win.argmax(0) + 1, 0)
            df["days_to_loan"] = first.astype(np.int8)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
