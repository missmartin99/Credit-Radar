"""Train -> evaluate -> ablate granularity -> score live -> decide -> export JSON for the dashboard.

Usage: python src/run_pipeline.py [data_dir] [out_dir]
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
import offer_engine as oe  # noqa: E402
from events import next_event  # noqa: E402
from features import (GALI_FEATS, HORIZON, LOC_FEATS, USER_FEATS, build_dataset, build_snapshot,  # noqa: E402
                      prepare_panel)

warnings.filterwarnings("ignore")
DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
OUT = sys.argv[2] if len(sys.argv) > 2 else "out"
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
log = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)

# ------------------------------------------------------------------ load
z = np.load(os.path.join(DATA, "panel.npz"))
panel = {k: z[k] for k in z.files if k != "dates"}
dates = pd.DatetimeIndex(z["dates"])
users = pd.read_csv(os.path.join(DATA, "users.csv"))
cells = pd.read_csv(os.path.join(DATA, "cells.csv"))
T, N = panel["loan"].shape
ASOF = T - 1
log(f"loaded panel T={T} N={N}; live as-of {dates[ASOF].date()}")

# ------------------------------------------------------------------ dataset (weekly snapshots)
first = int(np.where(dates == pd.Timestamp("2026-01-15"))[0][0])
snaps = list(range(first, T - 1 - HORIZON + 1, 7))
ds = build_dataset(panel, users, dates, snaps)
ds["date"] = dates[ds.t.values]
log(f"dataset rows={len(ds):,} positive rate={ds.y.mean():.2%}")

tr = ds[ds.date < "2026-08-01"]
va = ds[(ds.date >= "2026-08-15") & (ds.date < "2026-09-16")]
te = ds[ds.date >= "2026-10-01"]
log(f"train={len(tr):,} val={len(va):,} test={len(te):,} (time-based split, 14d gaps)")

FULL = USER_FEATS + LOC_FEATS + GALI_FEATS
VARIANTS = {"User behaviour only": USER_FEATS,
            "+ Locality signals": USER_FEATS + LOC_FEATS,
            "+ Gali (lane) signals": FULL}


def make(n_iter):
    return HistGradientBoostingClassifier(learning_rate=0.06, max_iter=n_iter, max_depth=5,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def best_iters(feats):
    m = make(400).fit(tr[feats], tr.y)
    losses = [log_loss(va.y, p[:, 1]) for p in m.staged_predict_proba(va[feats])]
    return int(np.argmin(losses)) + 1


def capture_at(y, s, frac):
    k = int(len(s) * frac)
    top = np.argsort(-s, kind="stable")[:k]
    return y[top].sum() / max(y.sum(), 1), y[top].mean()


def evaluate(y, s):
    y = np.asarray(y)
    c10, p10 = capture_at(y, s, 0.10)
    c20, _ = capture_at(y, s, 0.20)
    return dict(auc=roc_auc_score(y, s), prauc=average_precision_score(y, s), capture10=c10, capture20=c20,
                precision10=p10, lift10=p10 / y.mean())


# ------------------------------------------------------------------ train variants + ablation
n_best = best_iters(FULL)
log(f"best iterations on validation (full model) = {n_best}")
ablation, models = [], {}
for name, feats in VARIANTS.items():
    m = make(n_best).fit(tr[feats], tr.y)
    models[name] = m
    r = evaluate(te.y.values, m.predict_proba(te[feats])[:, 1])
    seg_auc = {}
    pt = m.predict_proba(te[feats])[:, 1]
    for sid, sname in enumerate(["Salaried", "Gig", "Merchant"]):
        mk = (te.seg == sid).values
        seg_auc[sname] = roc_auc_score(te.y.values[mk], pt[mk])
    ablation.append(dict(name=name, **r, seg_auc=seg_auc))
    log(f"{name:24s} AUC={r['auc']:.3f} PR-AUC={r['prauc']:.3f} capture@10%={r['capture10']:.1%} "
        f"| merchant AUC={seg_auc['Merchant']:.3f}")
model = models["+ Gali (lane) signals"]

# ------------------------------------------------------------------ calibration, baselines, slices
p_va = model.predict_proba(va[FULL])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va, va.y)
p_te_raw = model.predict_proba(te[FULL])[:, 1]
p_te = iso.predict(p_te_raw)
full = evaluate(te.y.values, p_te_raw)
brier_raw, brier_cal = brier_score_loss(te.y, p_te_raw), brier_score_loss(te.y, p_te)

rng = np.random.default_rng(0)
rule = ((te.fails_7 > 0).astype(float) * 2 + (te.p2p_cnt_14 > 0) + (te.bal_runway.fillna(99) < 7)).values
rule = rule + rng.random(len(rule)) * 1e-3  # random tie-break
base_rule = evaluate(te.y.values, rule)
rnd = evaluate(te.y.values, rng.random(len(te)))

thr = float(np.quantile(iso.predict(p_va), 0.90))
oe.CFG["need_threshold"] = thr
flag = p_te >= thr
thr_precision, thr_capture = te.y.values[flag].mean(), te.y.values[flag].sum() / te.y.sum()

by_balance = {}
for hb, lab in [(1, "Balance visible (AA / wallet)"), (0, "Flows only")]:
    mk = (te.has_balance == hb).values
    by_balance[lab] = dict(auc=roc_auc_score(te.y.values[mk], p_te_raw[mk]),
                           capture10=capture_at(te.y.values[mk], p_te_raw[mk], 0.1)[0])

cut = np.quantile(p_te_raw, 0.90)
lead = []
for lo, hi, lab in [(1, 3, "needs loan in 1-3 days"), (4, 7, "4-7 days"), (8, 14, "8-14 days")]:
    mk = ((te.days_to_loan >= lo) & (te.days_to_loan <= hi)).values
    lead.append(dict(bucket=lab, n=int(mk.sum()), captured=float((p_te_raw[mk] >= cut).mean())))

# area-level forecast quality: does the gali heatmap (mean predicted need per lane) track actual borrowing?
tc = te[["t", "uidx", "y"]].copy()
tc["cell"] = users.cell.values[tc.uidx.values]
tc["p"] = p_te
tc["rec"] = (te.days_since_loan < 30).values.astype(float)  # naive baseline: lane's recent borrowing share
gc = tc.groupby(["t", "cell"]).agg(p=("p", "mean"), y=("y", "mean"), rec=("rec", "mean")).reset_index()
within = lambda col: gc[col] - gc.groupby("cell")[col].transform("mean")
area_forecast = dict(
    r_model=float(np.corrcoef(gc.p, gc.y)[0, 1]), r_naive=float(np.corrcoef(gc.rec, gc.y)[0, 1]),
    r_model_within=float(np.corrcoef(within("p"), within("y"))[0, 1]),
    r_naive_within=float(np.corrcoef(within("rec"), within("y"))[0, 1]), n_cell_weeks=int(len(gc)))
log("area forecast:", {k: round(v, 3) for k, v in area_forecast.items()})

# time-to-need regressor (positives only)
trp, tep = tr[tr.y == 1], te[te.y == 1]
reg = HistGradientBoostingRegressor(learning_rate=0.06, max_iter=200, max_depth=4, min_samples_leaf=50,
                                    random_state=0).fit(trp[FULL], trp.days_to_loan)
mae = float(np.abs(reg.predict(tep[FULL]) - tep.days_to_loan).mean())
mae_base = float(np.abs(trp.days_to_loan.median() - tep.days_to_loan).mean())
log(f"time-to-need MAE={mae:.2f}d vs constant {mae_base:.2f}d")

samp = te.sample(20000, random_state=0)
pi = permutation_importance(model, samp[FULL], samp.y, scoring="roc_auc", n_repeats=2, random_state=0, n_jobs=2)
imp = sorted(zip(FULL, pi.importances_mean), key=lambda x: -x[1])[:12]

metrics = dict(
    n_train=len(tr), n_test=len(te), base_rate=float(te.y.mean()), horizon_days=HORIZON, full=full,
    random=rnd, rule_baseline=base_rule, ablation=ablation, brier_raw=brier_raw, brier_cal=brier_cal,
    threshold=thr, threshold_precision=float(thr_precision), threshold_capture=float(thr_capture),
    threshold_flag_rate=float(flag.mean()), by_balance=by_balance, lead_time=lead,
    ttn_mae=mae, ttn_mae_baseline=mae_base, area_forecast=area_forecast, importance=[dict(f=f, v=float(v)) for f, v in imp])
log(f"FULL: AUC={full['auc']:.3f} PR-AUC={full['prauc']:.3f} capture@10%={full['capture10']:.1%} "
    f"lift@10%={full['lift10']:.2f}x | rule baseline capture@10%={base_rule['capture10']:.1%} "
    f"| random={rnd['capture10']:.1%}")
log(f"threshold={thr:.3f} flags {flag.mean():.1%} of rows with precision {thr_precision:.1%} "
    f"capturing {thr_capture:.1%} of loan-takers; Brier raw={brier_raw:.4f} cal={brier_cal:.4f}")
log("lead-time capture:", [(d['bucket'], round(d['captured'], 2)) for d in lead])

# ------------------------------------------------------------------ live scoring as of last day
P = prepare_panel(panel)
live, ev_name = build_snapshot(P, users, dates, ASOF)
ev_days = int(live.days_to_event.iloc[0])
live = pd.concat([live, users[["marketing_consent", "dnd", "nudges_30d", "cell", "loc", "user_id"]]], axis=1)
live["marketing_consent"] = live.marketing_consent.astype(bool)
live["dnd"] = live.dnd.astype(bool)
p_live = iso.predict(model.predict_proba(live[FULL])[:, 1])
ttn = reg.predict(live[FULL])
dec = oe.decide(live, p_live, ttn)
live = pd.concat([live, dec], axis=1)
fun = oe.funnel(live, p_live)
log("funnel:", [(f["step"], f["n"]) for f in fun])

# ------------------------------------------------------------------ reason codes (occlusion) for displayed users
TOPK = 10
top_idx = live.sort_values("p_need", ascending=False).groupby("cell").head(TOPK).index.values
med = tr[FULL].median()
X = live.loc[top_idx, FULL].copy()
base_p = model.predict_proba(X)[:, 1]
delta = np.zeros((len(X), len(FULL)))
for j, f in enumerate(FULL):
    Xj = X.copy()
    Xj[f] = med[f]
    d = base_p - model.predict_proba(Xj)[:, 1]
    d[np.isnan(X[f].values)] = 0
    delta[:, j] = d
SKIP = {"seg", "has_balance", "event_weight"}
FMT = {
    "bal_runway": lambda v: f"Cash runway ≈ {v:.0f} days",
    "bal_min14_ratio": lambda v: f"Balance dipped to ≈ {v:.1f} days of spend (14d)",
    "fails_7": lambda v: f"{v:.0f} insufficient-fund declines this week",
    "fails_30": lambda v: f"{v:.0f} insufficient-fund declines in 30d",
    "p2p_cnt_14": lambda v: f"{v:.0f} peer-transfer credits in 14d",
    "p2p_amt_ratio": lambda v: f"Peer credits = {v:.0%} of inflow (30d)",
    "in_trend": lambda v: f"Inflow {np.expm1(v):+.0%} vs 30d average",
    "out_trend": lambda v: f"Spend {np.expm1(v):+.0%} vs 30d average",
    "txn_trend": lambda v: f"Txn count {np.expm1(v):+.0%} vs 30d average",
    "net_30_ratio": lambda v: f"Net cash flow {v:+.0%} of inflow (30d)",
    "days_to_event": lambda v: f"{ev_name} in {v:.0f} days",
    "dom": lambda v: f"Day {v:.0f} of the month (salary cycle)",
    "foir": lambda v: f"EMI load {v:.0%} of inflow",
    "days_since_loan": lambda v: f"Last borrowed {v:.0f} days ago",
    "loans_90": lambda v: f"{v:.0f} loans in last 90 days",
    "inflow_cv_60": lambda v: f"Volatile income (CV {v:.1f})",
    "inflow_30": lambda v: f"30-day inflow ₹{v:,.0f}",
    "outflow_30": lambda v: f"30-day outflow ₹{v:,.0f}",
    "inflow_7": lambda v: f"7-day inflow ₹{v:,.0f}",
    "outflow_7": lambda v: f"7-day outflow ₹{v:,.0f}",
    "bal_now": lambda v: f"Balance ₹{v:,.0f}",
}
AREA_FMT = {
    "fail_share": lambda v, s: f"{v:.0%} of neighbours ({s}) had declines this week",
    "in_trend": lambda v, s: f"Neighbours' inflow {np.expm1(v):+.0%} vs 30d ({s})",
    "merch_in_trend": lambda v, s: f"Neighbouring merchants' sales {np.expm1(v):+.0%} ({s})",
    "p2p_share": lambda v, s: f"{v:.0%} of neighbours borrowing from peers ({s})",
    "loan30_share": lambda v, s: f"{v:.0%} of neighbours took a loan in 30d ({s})",
    "out_trend": lambda v, s: f"Neighbours' spend {np.expm1(v):+.0%} vs 30d ({s})",
}


def reason(f, v):
    if f.startswith("gali_"):
        return AREA_FMT[f[5:]](v, "gali")
    if f.startswith("loc_") and f != "loc_tier":
        return AREA_FMT[f[4:]](v, "locality")
    if f == "loc_tier":
        return f"Locality income tier {v:.2f}"
    return FMT.get(f, lambda x: f"{f} = {x:.2f}")(v)


reasons = {}
for r, ix in enumerate(top_idx):
    order = [j for j in np.argsort(-delta[r]) if FULL[j] not in SKIP and delta[r, j] > 0.003][:3]
    reasons[ix] = [dict(text=reason(FULL[j], float(X.iloc[r, j])), w=float(delta[r, j])) for j in order]

# ------------------------------------------------------------------ export
city = dict(fail=float((live.fails_7 > 0).mean()),
            merch_in=float(np.expm1(live.loc[live.seg == 2, "in_trend"].mean())),
            p2p=float((live.p2p_cnt_14 > 0).mean()), need=float(live.p_need.mean()))


def agg(g):
    return dict(n=int(len(g)), mean_need=float(g.p_need.mean()), expected_seekers=float(g.p_need.sum()),
                offers=int((g.action == "Offer").sum()), wellbeing=int((g.action == "Wellbeing nudge").sum()),
                need_by_seg=[float(g.loc[g.seg == s, "p_need"].mean()) if (g.seg == s).any() else None
                             for s in range(3)],
                seg_n=[int((g.seg == s).sum()) for s in range(3)],
                fail=float((g.fails_7 > 0).mean()),
                merch_in=float(np.expm1(g.loc[g.seg == 2, "in_trend"].mean())) if (g.seg == 2).any() else None,
                p2p=float((g.p2p_cnt_14 > 0).mean()))


cells_out = []
for _, c in cells.iterrows():
    g = live[live.cell == c.cell]
    if len(g) < 30:  # k-anonymity guard: never show tiny cells
        continue
    top = g.loc[[i for i in g.sort_values("p_need", ascending=False).index[:TOPK]]]
    ulist = []
    for i, r in top.iterrows():
        msg = (oe.generate_message(int(r.seg), int(r.amount), r.lender, ev_name, ev_days)
               if r.action == "Offer" else None)
        ulist.append(dict(id=r.user_id, seg=int(r.seg), p=float(r.p_need), d=int(r.days_to_need),
                          send=int(r.send_in_days), act=r.action, why=r.reason, amt=int(r.amount),
                          lender=r.lender, msg=msg, viol=(oe.compliance_check(msg) if msg else []),
                          reasons=reasons.get(i, []), inflow=float(r.inflow_30), foir=float(r.foir),
                          consent=bool(r.marketing_consent)))
    cells_out.append(dict(id=c.cell_code, name=c["name"], loc=c.loc_name, lat=float(c.lat), lon=float(c.lon),
                          **agg(g), users=ulist))
locs_out = []
for ln, g in live.merge(cells[["cell", "loc_name"]], on="cell").groupby("loc_name"):
    ce = cells[cells.loc_name == ln].iloc[0]
    locs_out.append(dict(name=ln, lat=float(cells[cells.loc_name == ln].lat.mean()),
                         lon=float(cells[cells.loc_name == ln].lon.mean()), **agg(g)))

out = dict(meta=dict(asof=str(dates[ASOF].date()), n_users=int(N), n_cells=len(cells_out), horizon=HORIZON,
                     next_event=ev_name, days_to_event=ev_days, threshold=thr, synthetic=True),
           metrics=metrics, funnel=fun, city=city, cells=cells_out, locs=locs_out)
json.dump(out, open(os.path.join(OUT, "dashboard_data.json"), "w"), default=float)
live[["user_id", "cell", "p_need", "days_to_need", "capacity_ok", "amount", "action", "reason", "send_in_days"]] \
    .to_csv(os.path.join(OUT, "scores_asof.csv"), index=False)
json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
log("exported", OUT)
