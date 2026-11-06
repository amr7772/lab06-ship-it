"""Score one QuickBite order: check it, predict it, decide on the voucher.

The saved pipeline and model_info.json are loaded once, when this module is
imported. Nothing else in the repository talks to the model; app.py only
calls predict_order().
"""

import json
import math
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"

# Loaded once, at import time. A joblib file can run code when it is loaded,
# so this only ever reads the file that this repository ships.
PIPELINE = joblib.load(MODEL_DIR / "late_orders.joblib")
INFO = json.loads((MODEL_DIR / "model_info.json").read_text())

# What a dispatcher may type for each column. These limits are decisions, and
# the README explains each one.
NUMBER_LIMITS = {
    "distance_km": {"low": 0.1, "high": 30, "whole": False},
    "items": {"low": 1, "high": 20, "whole": True},
    "hour": {"low": 10, "high": 22, "whole": True},
    "prep_minutes": {"low": 1, "high": 60, "whole": False},
    "courier_months": {"low": 0, "high": 120, "whole": True},
}
CHOICES = {
    "weather": ["clear", "rain", "snow"],
    "day_type": ["weekday", "weekend"],
    "vehicle": ["bike", "scooter", "car"],
}
# The only field that may be left empty: the pipeline fills it with the
# training median. Every other missing field is refused.
OPTIONAL_FIELDS = {"prep_minutes"}


def feature_lists(info=None):
    """Return (numeric, categorical) column names from model_info.json."""
    info = INFO if info is None else info
    return list(info["numeric_features"]), list(info["categorical_features"])


def _is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return isinstance(value, float) and math.isnan(value)


def _check_number(name: str, value) -> float:
    rule = NUMBER_LIMITS[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a number, but got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, but got {value}.")
    if rule["whole"] and value != int(value):
        raise ValueError(f"{name} must be a whole number, but got {value}.")
    if not rule["low"] <= value <= rule["high"]:
        raise ValueError(
            f"{name} must be between {rule['low']} and {rule['high']}, "
            f"but got {value:g}."
        )
    return value


def _check_choice(name: str, value) -> str:
    allowed = CHOICES[name]
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(
            f"{name} must be one of {', '.join(allowed)}, but got {value!r}."
        )
    return value


def validate_order(order: dict, info=None):
    """Check one order and return (clean_order, warnings).

    Raises ValueError, naming the field and the rule, for any missing or
    impossible value. An empty prep_minutes is accepted with a warning.
    """
    numeric, categorical = feature_lists(info)
    clean, warnings = {}, []
    for name in numeric + categorical:
        if name not in NUMBER_LIMITS and name not in CHOICES:
            raise ValueError(f"{name} is in model_info.json but has no rule here.")
        value = order.get(name)
        if _is_empty(value):
            if name in OPTIONAL_FIELDS:
                clean[name] = float("nan")
                warnings.append(
                    f"{name} was left empty, so the model used the typical "
                    "value from its training data."
                )
                continue
            raise ValueError(f"{name} is required and was left empty.")
        if name in NUMBER_LIMITS:
            clean[name] = _check_number(name, value)
        else:
            clean[name] = _check_choice(name, value)
    return clean, warnings


def should_offer_voucher(probability: float, threshold=None) -> bool:
    """Offer the voucher when the chance of a late order reaches the threshold.

    The threshold comes from model_info.json, never from a number in this file.
    """
    threshold = INFO["threshold"] if threshold is None else threshold
    return probability >= threshold


def predict_order(order: dict) -> dict:
    """Check an order, score it and return the probability and the decision."""
    clean, warnings = validate_order(order)
    numeric, categorical = feature_lists()
    row = pd.DataFrame([clean], columns=numeric + categorical)
    probability = float(PIPELINE.predict_proba(row)[0, 1])
    return {
        "probability_late": probability,
        "offer_voucher": should_offer_voucher(probability),
        "threshold": INFO["threshold"],
        "warnings": warnings,
    }
