"""Festival / season calendar shared by the data generator and the feature pipeline.

Dates are APPROXIMATE (verify before relying on them). `weight` = rough retail intensity.
The wedding-season entry lies after the end of the synthetic history on purpose: it is the
"next event" the live scoring run looks ahead to.
"""
import numpy as np
import pandas as pd

EVENTS = [
    ("Christmas / New Year", "2025-12-25", 0.4),
    ("Lohri / Makar Sankranti", "2026-01-14", 0.3),
    ("Holi", "2026-03-04", 0.6),
    ("Eid-ul-Fitr", "2026-03-20", 0.7),
    ("Raksha Bandhan", "2026-08-28", 0.6),
    ("Dussehra", "2026-10-20", 0.6),
    ("Diwali", "2026-11-08", 1.0),
    ("Wedding season", "2026-11-21", 0.8),
]
_EV = [(n, pd.Timestamp(d), w) for n, d, w in EVENTS]


def _ramp(delta_days, lead, decay=3):
    """1.0 on the event day, ramps up linearly over `lead` days before it, decays over `decay` days after."""
    if 0 <= delta_days <= lead:
        return 1.0 - delta_days / lead
    if -decay <= delta_days < 0:
        return 1.0 + delta_days / decay
    return 0.0


def intensity(date, lead):
    return min(1.2, sum(w * _ramp((d - date).days, lead) for _, d, w in _EV))


def next_event(date):
    """(name, days_to_event, weight) of the next event on/after `date`; (None, 60, 0) if none within 60d."""
    best = (None, 60, 0.0)
    for n, d, w in _EV:
        k = (d - date).days
        if 0 <= k < best[1]:
            best = (n, k, w)
    return best
