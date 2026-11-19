"""Decision layer: turns a calibrated need score into an action, with guardrails.

Key design choice: NEED is not the same as ELIGIBILITY/AFFORDABILITY. A high need score for someone who
cannot afford a loan routes to a non-credit "wellbeing" path, never to a loan offer. In production the
lending partner (bank/NBFC) underwrites and approves; Paytm's role is lead scoring and distribution, so the
capacity rules below are a pre-screen to be replaced by the partner's credit policy.
"""
import numpy as np
import pandas as pd

LENDERS = ["Partner NBFC A", "Partner Bank B"]  # placeholders
KFS_LINK = "paytm.example/kfs"                  # placeholder

CFG = dict(
    need_threshold=0.20,      # overwritten at runtime from validation data
    min_inflow_30=10000,       # Rs
    max_foir=0.15,            # EMI / inflow
    max_loans_90=1,
    max_fails_30=3,
    recent_loan_days=30,
    max_nudges_30d=2,
    cap_individual=50000,
    cap_merchant=100000,
)

BANNED = ["urgent", "last chance", "only today", "guaranteed", "instant cash", "low balance", "balance kam",
          "paise khatam", "tight", "emergency", "abhi nahi to", "hurry", "turant"]
REQUIRED = ["kfs", "zarurat"]  # must mention key-fact statement + choice framing


def capacity_ok(df):
    return ((df.inflow_30 >= CFG["min_inflow_30"]) & (df.foir <= CFG["max_foir"])
            & (df.loans_90 <= CFG["max_loans_90"]) & (df.fails_30 <= CFG["max_fails_30"]))


def suggest_amount(df):
    cap = np.where(df.seg == 2, CFG["cap_merchant"], CFG["cap_individual"])
    frac = np.where(df.seg == 2, 0.70, 0.50)
    amt = np.minimum(frac * df.outflow_30, 0.8 * df.inflow_30)
    amt = np.clip(np.round(amt / 1000) * 1000, 5000, cap)
    return amt.astype(int)


def decide(df, p_need, days_to_need):
    """df: feature frame joined with user flags (marketing_consent, dnd, nudges_30d). Returns decision frame."""
    out = pd.DataFrame(index=df.index)
    out["p_need"] = p_need
    out["days_to_need"] = np.clip(np.round(days_to_need), 1, 14).astype(int)
    out["capacity_ok"] = capacity_ok(df).values
    out["amount"] = suggest_amount(df)
    action = np.full(len(df), "Offer", dtype=object)
    reason = np.full(len(df), "Passed all guardrails", dtype=object)

    def apply(mask, a, r):
        m = mask & (action == "Offer")
        action[m] = a
        reason[m] = r

    apply(p_need < CFG["need_threshold"], "No action", "Need score below threshold")
    apply(~out["capacity_ok"].values, "Wellbeing nudge", "Need high but affordability weak: no credit offer")
    apply(~df.marketing_consent.values | df.dnd.values, "Suppress", "No marketing consent / DND")
    apply((df.days_since_loan < CFG["recent_loan_days"]).values, "Suppress", "Borrowed in last 30 days")
    apply((df.nudges_30d >= CFG["max_nudges_30d"]).values, "Suppress", "Frequency cap reached")
    out["action"], out["reason"] = action, reason
    # send ~3 days ahead of predicted need, never later than today
    out["send_in_days"] = np.maximum(out["days_to_need"] - 3, 0)
    out["lender"] = [LENDERS[i % 2] for i in df.uidx.values]
    return out


def funnel(df, p_need):
    steps = [("Users scored", np.ones(len(df), bool))]
    m = p_need >= CFG["need_threshold"]
    steps.append(("Predicted need in 14d", m))
    m = m & capacity_ok(df).values
    steps.append(("Affordability pre-screen OK", m))
    m = m & df.marketing_consent.values & ~df.dnd.values
    steps.append(("Consent & not DND", m))
    m = m & (df.days_since_loan >= CFG["recent_loan_days"]).values
    steps.append(("Not borrowed in last 30d", m))
    m = m & (df.nudges_30d < CFG["max_nudges_30d"]).values
    steps.append(("Within frequency cap -> OFFER", m))
    return [dict(step=s, n=int(x.sum())) for s, x in steps]


# ------------------------------------------------------------------ message generation
def template_message(seg, amount, lender, event_name=None, days_to_event=None):
    amt = f"₹{amount:,}"
    if seg == 2 and event_name and days_to_event is not None and days_to_event <= 25:
        hook = f"{event_name} se pehle stock ke liye "
        kind = "business loan"
        lead = "aapke"
    else:
        hook = ""
        kind = "loan"
        lead = "Aapke"
    return (f"{hook}{lead} liye {amt} tak ka {kind} offer available hai ({lender}). Zarurat ho tabhi lein. "
            f"Rate aur charges pehle dekhein (KFS): {KFS_LINK}")


def compliance_check(msg):
    low = msg.lower()
    v = [f"banned phrase: '{b}'" for b in BANNED if b in low]
    v += [f"missing required element: '{r}'" for r in REQUIRED if r not in low]
    if "₹" not in msg:
        v.append("missing amount")
    return v


LLM_PROMPT = """You write a single push-notification for a lending offer in Hinglish (Roman script).
Rules: neutral tone; never mention or imply the customer's balance, spending, stress or area; no urgency or
scarcity; state the amount ₹{amount} and lender {lender}; say the customer should borrow only if needed
('zarurat ho tabhi'); include 'KFS' and the link {kfs}. Max 220 characters. Context (public only): segment={seg},
upcoming={event}."""


def generate_message(seg, amount, lender, event_name=None, days_to_event=None, llm=None):
    """`llm`: optional callable(prompt)->str (e.g. a Claude API call). Output must pass compliance_check,
    otherwise we fall back to the template. The LLM never sees personal data, only segment/amount/lender."""
    if llm is not None:
        prompt = LLM_PROMPT.format(amount=f"{amount:,}", lender=lender, kfs=KFS_LINK,
                                   seg=["salaried", "gig", "merchant"][int(seg)], event=event_name)
        try:
            msg = llm(prompt)
            if not compliance_check(msg):
                return msg
        except Exception:
            pass
    return template_message(seg, amount, lender, event_name, days_to_event)
