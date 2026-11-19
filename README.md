# Credit Radar: forecasting who will need credit, and where, before they ask

Hackathon MVP for Paytm. Predicts, per user and aggregated per locality / gali (lane), the chance of needing a
loan in the next 14 days, then decides whether (and when) a pre-approved offer should be shown, with guardrails.

**Everything in this repo runs on synthetic data.** Metrics prove the pipeline works, not that it will perform on
Paytm data. Swap in real data using the schema below.

```
bash run_all.sh            # data -> model -> offers -> dashboard (needs numpy, pandas, scikit-learn)
open dashboard/index.html
```

## 1. Verdict on the idea

Strong direction, but three framing changes make it much more defensible and more likely to win.

| Original framing | Recommended framing | Why |
|---|---|---|
| Predict need at gali level, then sell the loan | **Score users, aggregate to gali** (view), and test whether gali features help individual scores | In our ablation, lane features add ~nothing to individual AUC; the aggregate lane forecast is the useful product. Prove it on real data before collecting finer location. |
| Sell the loan before they need it | **Pre-approved, neutral, opt-in offer timed ~3 days ahead**, with an affordability gate and a non-credit "wellbeing" path | Predicting distress and pushing debt is the fastest way to a regulatory and PR problem. Need is not eligibility. |
| Consumers first | **Merchants first** (restocking before festivals / wedding season), consumers second | Paytm's lending push is merchant loans and larger-ticket credit; small-ticket consumer credit (<₹50k, mostly Postpaid) was de-emphasised. Merchant cash-flow seasonality is also the most predictable signal. |

Context checked on 19 Nov 2026 (links at the bottom):
- Paytm is a loan **distributor**: partner banks/NBFCs underwrite and book the loans. So Paytm's product is lead scoring, timing and distribution, not credit decisions.
- Merchant loans are the focus; lending reaches only ~7% of subscription merchants (Q4 FY26), so there is large headroom.
- RBI Digital Lending Directions 2025: Paytm acts as a Lending Service Provider (agent of the regulated entity); data collection must be need-based with prior explicit consent and audit trails.
- DPDP Rules 2025 were notified in Nov 2025 with an 18-month phase-in (to ~May 2027).

## 2. How it works

```
payments panel ─► features (user + locality + gali, leave-one-out) ─► GBM need model (14d) ─► isotonic calibration
                                                                  └► time-to-need model (days)
                                         live score ─► decision engine ─► offer / wellbeing nudge / suppress
                                                                       └► message (template or LLM, linted) + reason codes
                          user scores ─► aggregate to gali/locality (k≥30) ─► dashboard heatmap
```

| File | Role |
|---|---|
| `src/gen_data.py` | Synthetic Delhi panel: 30 localities x 4 lanes, salaried / gig / merchant users, festival calendar, area shocks, loan events. Replace with real data. |
| `src/features.py` | Rolling user features (inflow/outflow trends, declines, peer-transfer credits, EMI load, balance runway where visible) + area features at gali and locality level. No look-ahead. |
| `src/run_pipeline.py` | Time-split training, ablation (user-only / +locality / +gali), calibration, baselines, lane-level forecast check, live scoring, reason codes, export. |
| `src/offer_engine.py` | Guardrails, amount sizing, send timing, message generation with a compliance linter and an optional LLM hook. |
| `dashboard/` | Self-contained interactive demo (map, drill-down, per-user decision and message preview). |
| `out/` | `metrics.json`, `scores_asof.csv` (live scores and decisions), `dashboard_data.json`. |

### Guardrails (in order)
1. Need score below threshold: no action.
2. Affordability pre-screen fails (income, EMI load, loan stacking, declines): **no credit offer**, wellbeing path instead.
3. No marketing consent / DND: suppress.
4. Borrowed in last 30 days: suppress (no cannibalisation, no stacking).
5. Frequency cap: suppress.
6. Message must pass the compliance linter (no urgency/scarcity, no inferred distress, KFS link, "borrow only if needed" wording). Customer copy never references balance, spending or neighbours.
7. Area views suppress any cell with fewer than 30 users.

In production the partner's credit policy replaces the affordability pre-screen.

## 3. Results on synthetic data (time-split, 14-day horizon)

See `out/metrics.json`. Headline numbers from the shipped run:
- AUC 0.82; top-10% of scores contains 46% of future borrowers (simple low-balance rule: 20%; random: 10%).
- Lane-level forecast correlation r = 0.69 vs 0.25 for "who borrowed last month" (600 lane-weeks).
- Time-to-need model MAE 2.9 days vs 3.5 for a constant guess.
- Ablation: user-only 0.818, +locality 0.816, +gali 0.821 AUC. Differences are within noise. **We deliberately do not claim gali features help.**
- Synthetic-data caveats: the generator embeds an assumed neighbour-to-neighbour borrowing contagion; the real world may show more, less or none.

## 4. Plugging in real data

`features.py` expects a panel of arrays shaped `[days, users]` (see `gen_data.py` for the saved format) plus a users table:

| Field | Meaning | Real-world source |
|---|---|---|
| `inflow`, `outflow` (excl. loan disbursals) | daily money in/out | Paytm UPI / wallet / merchant settlements |
| `ntxn` | transaction count | same |
| `fails` | declines for insufficient funds | UPI response codes on Paytm PSP |
| `p2p` | inbound peer-transfer credits | UPI P2P |
| `loan` | loan disbursal amount (label source) | lender credits seen on Paytm rails, partner loan data, Account Aggregator |
| `emi` | EMI / autopay debits | UPI AutoPay / mandates |
| `bal` | end-of-day balance (NaN if not visible) | Paytm Payments Bank wallet / Account Aggregator with consent |
| users: `cell`, `loc`, `seg`, `loc_tier`, flags | gali / locality ids (derive from H3 or pincode + merchant geo), segment, consent / DND / nudge counts | CRM + consent manager |

Label caveat: Paytm sees only its own rails. "Took a loan" must be defined from observable proxies (lender credits, Postpaid/BNPL spikes, partner loan data, AA). Expect label noise; measure it.

## 5. Execution plan

### Hackathon (48h)
| When | Do |
|---|---|
| Hours 0-6 | Lock scope: **merchant working-capital before the next festival/wedding season** as the hero use case. Agree the data you can show (synthetic + any sample Paytm gives). |
| 6-18 | Run `run_all.sh`; if sample real data exists, map it to the panel schema and re-run the ablation. |
| 18-30 | Polish dashboard for the story: city heatmap → lane → user decision → message. Add Hindi/Hinglish copy variants; wire the LLM hook (`generate_message(llm=...)`) with the linter as a safety net. |
| 30-42 | Business case slide (formula below), risk/compliance slide, 5-minute demo script. |
| 42-48 | Rehearse; prepare answers to "is this predatory?", "who underwrites?", "what is incremental?". |

### Pilot after the hackathon (8-12 weeks)
1. **Weeks 1-3: data and labels.** Build the real panel; define `need` labels; offline backtest (time-split); run the geography ablation. Decide gali vs pincode.
2. **Weeks 3-5: partner and compliance.** Agree with lending partner: pre-approval policy, KFS, consent flows, DND handling; legal review (RBI Digital Lending Directions, DPDP).
3. **Weeks 5-10: randomised pilot.** Merchants in 2-3 localities. Arms: (a) no proactive offer, (b) offer at predicted-need time, (c) existing at-need flow. Measure **incremental** disbursals (not just conversion), 30/60-day delinquency (from partner), opt-outs and complaints.
4. **Week 10+: decide** scale, uplift modelling (who is *persuadable*), add AA balance, extend to consumers.

Business case (fill with real numbers): `incremental disbursal = offers sent × (conv_treatment − conv_control) × avg ticket`; revenue = disbursal × distribution fee; subtract delinquency-linked clawbacks and comms cost.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Looks predatory ("we predict you're broke") | Neutral copy, opt-in, wellbeing path, affordability gate, frequency caps, no inference reveal |
| Consent / data-use limits | Use first-party payments data under explicit consent; document purpose; consent manager for AA data |
| Geo-discrimination (redlining) | Never use lane/locality to deny; use only for aggregate planning and as one feature if it proves useful; fairness audit by area income tier |
| Label noise / selection bias | Treat as proxy; validate with partner data; randomised holdout |
| Cannibalisation (they'd have borrowed anyway) | Incrementality test; uplift model later |
| Drift (festival calendar, shocks) | Weekly retrain; calibration monitoring; event calendar as an input |
| LLM copy risk | LLM sees only segment/amount/lender; output must pass the linter or falls back to template |

## 7. Five-slide pitch outline
1. Problem: loan marketing is reactive; users compare options only when already stressed; merchants miss pre-festival stocking.
2. Insight: payment rails show cash-flow turns 1-2 weeks early, and merchants' needs are seasonal and local.
3. Demo: heatmap → lane → user decision → compliant message.
4. Trust: need ≠ eligibility, wellbeing path, k-anonymity, partner underwrites, honest metrics.
5. Plan and ask: pilot design, incrementality metric, data access needed.

## Sources
- RBI (Digital Lending) Directions, 2025: https://www.rbi.org.in/scripts/FS_Notification.aspx?Id=12848
- Paytm loan distribution expansion: https://paytm.com/blog/investor-relations/paytm-expands-credit-distribution-business-to-enhance-focus-on-higher-ticket-loans-for-consumers-and-merchants-in-partnership-with-banks-and-nbfcs
- Paytm Q4 FY26 results: https://paytm.com/document/ir/financial-results/fy2025-26/Paytm_Earning_Release_Q4-FY-2026.pdf
- Inc42 on merchant loan penetration: https://inc42.com/features/payments-vs-lending-the-tale-of-paytms-two-engines/
- DPDP Rules 2025 (PIB): https://static.pib.gov.in/WriteReadData/specificdocs/documents/2025/nov/doc20251117695301.pdf

This is not legal advice; have counsel review before any real deployment.
