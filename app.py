"""Credit Radar: Streamlit front-end.

Serves pre-computed outputs (out/, dashboard/), so nothing is trained at deploy time and the app starts fast.
To refresh the outputs: `bash run_all.sh` (needs numpy, pandas, scikit-learn; see requirements-pipeline.txt).
All data is synthetic.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SEG = {0: "Salaried", 1: "Gig worker", 2: "Merchant"}

st.set_page_config(page_title="Credit Radar", page_icon="📍", layout="wide")

REQUIRED = [ROOT / "out" / "scores_asof.csv", ROOT / "out" / "metrics.json", ROOT / "data" / "cells.csv",
            ROOT / "data" / "users.csv", ROOT / "dashboard" / "index.html"]
missing = [str(p.relative_to(ROOT)) for p in REQUIRED if not p.exists()]
if missing:
    st.error("Missing files: " + ", ".join(missing) + ". Run `bash run_all.sh` first.")
    st.stop()


@st.cache_data
def load():
    scores = pd.read_csv(ROOT / "out" / "scores_asof.csv")
    cells = pd.read_csv(ROOT / "data" / "cells.csv")
    users = pd.read_csv(ROOT / "data" / "users.csv")[["user_id", "seg"]]
    metrics = json.loads((ROOT / "out" / "metrics.json").read_text(encoding="utf-8"))
    html = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    scores = (scores.merge(cells[["cell", "name", "loc_name"]], on="cell", how="left")
                    .merge(users, on="user_id", how="left"))
    scores["segment"] = scores["seg"].map(SEG)
    scores["need_pct"] = scores["p_need"] * 100
    return scores, metrics, html


def embed_html(html: str, height: int) -> None:
    """st.iframe replaces st.components.v1.html (deprecated); fall back for older Streamlit versions."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=True)


scores, metrics, dash_html = load()

with st.sidebar:
    st.header("Credit Radar")
    st.caption("Hyperlocal liquidity-need forecast: who is likely to need credit in the next 14 days, "
               "by locality and lane (gali), with offer guardrails.")
    st.warning("All data here is synthetic. Metrics show the pipeline works, not how it will perform on real "
               "Paytm data.")
    st.caption("Lending partners underwrite and approve loans. This is lead scoring and timing only.")

st.title("Credit Radar")
tab_dash, tab_scores, tab_method = st.tabs(["Interactive dashboard", "Scores explorer", "Method and results"])

# ------------------------------------------------------------------ tab 1
with tab_dash:
    st.caption("Click a locality bubble to zoom into its lanes, click a lane, then a user to see the decision "
               "and message preview.")
    embed_html(dash_html, height=2200)

# ------------------------------------------------------------------ tab 2
with tab_scores:
    st.caption("Live scores for every synthetic user as of the last day of data, with the decision from the "
               "guardrail engine.")
    c1, c2, c3 = st.columns(3)
    sel_loc = c1.multiselect("Locality (empty = all)", sorted(scores["loc_name"].dropna().unique()))
    sel_seg = c2.multiselect("Segment", list(SEG.values()), default=list(SEG.values()))
    actions = sorted(scores["action"].unique())
    sel_act = c3.multiselect("Decision", actions, default=actions)
    min_need = st.slider("Minimum need score (14-day)", 0.0, 1.0, 0.0, 0.01)

    f = scores
    if sel_loc:
        f = f[f["loc_name"].isin(sel_loc)]
    f = f[f["segment"].isin(sel_seg) & f["action"].isin(sel_act) & (f["p_need"] >= min_need)]
    f = f.sort_values("p_need", ascending=False)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Users shown", f"{len(f):,}")
    m2.metric("Offers cleared", f"{int((f['action'] == 'Offer').sum()):,}")
    m3.metric("Wellbeing nudges", f"{int((f['action'] == 'Wellbeing nudge').sum()):,}")
    m4.metric("Expected borrowers (14d)", f"{f['p_need'].sum():,.0f}")

    if len(f):
        st.subheader("Top lanes by expected borrowers")
        top = f.groupby("name")["p_need"].sum().sort_values(ascending=False).head(15)
        st.bar_chart(top)

    show = f[["user_id", "name", "segment", "need_pct", "days_to_need", "action", "reason", "amount",
              "send_in_days"]].rename(columns={"name": "lane", "need_pct": "need (14d)",
                                               "days_to_need": "days to need", "amount": "suggested ₹",
                                               "send_in_days": "send in (days)"})
    st.dataframe(show, hide_index=True, column_config={
        "need (14d)": st.column_config.ProgressColumn("need (14d)", min_value=0, max_value=100, format="%.0f%%"),
    })
    st.download_button("Download filtered rows (CSV)", show.to_csv(index=False).encode("utf-8"),
                       file_name="credit_radar_scores.csv", mime="text/csv")

# ------------------------------------------------------------------ tab 3
with tab_method:
    full, rule, rnd, af = metrics["full"], metrics["rule_baseline"], metrics["random"], metrics["area_forecast"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Future borrowers in top 10% of scores", f"{full['capture10']:.0%}",
              help=f"Simple low-balance rule: {rule['capture10']:.0%}. Random: {rnd['capture10']:.0%}.")
    k2.metric("Ranking quality (AUC)", f"{full['auc']:.2f}")
    k3.metric("Lane-level forecast correlation", f"r = {af['r_model']:.2f}",
              help=f"Naive baseline (who borrowed last month): r = {af['r_naive']:.2f}.")
    k4.metric("Base rate (14-day borrowing)", f"{metrics['base_rate']:.1%}")

    st.subheader("Does finer geography help individual scores?")
    abl = pd.DataFrame([{"Feature set": a["name"], "AUC": round(a["auc"], 3),
                         "Captured in top 10%": f"{a['capture10']:.1%}",
                         "Merchant AUC": round(a["seg_auc"]["Merchant"], 3)} for a in metrics["ablation"]])
    st.dataframe(abl, hide_index=True)
    st.markdown("The three rows differ by less than run-to-run noise, so we do **not** claim lane-level features "
                "improve individual scores. The value of the gali view is the aggregated lane forecast. "
                "Re-run this ablation on real data before collecting anything finer than pincode.")

    st.subheader("How early can we see it?")
    lead = pd.DataFrame([{"Borrows in": x["bucket"].replace("needs loan in ", ""),
                          "Captured in top 10%": f"{x['captured']:.0%}", "Borrowers": x["n"]}
                         for x in metrics["lead_time"]])
    st.dataframe(lead, hide_index=True)

    with st.expander("Full README: evaluation, real-data schema, pilot plan, risks"):
        st.markdown((ROOT / "README.md").read_text(encoding="utf-8"))
