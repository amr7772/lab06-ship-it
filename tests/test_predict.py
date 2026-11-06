import importlib
import math

import joblib
import pytest

import predict

HARD_ORDER = {
    "distance_km": 12.0,
    "items": 8,
    "hour": 19,
    "courier_months": 0,
    "weather": "snow",
    "day_type": "weekend",
    "vehicle": "bike",
}
EASY_ORDER = {
    "distance_km": 1.0,
    "items": 1,
    "hour": 15,
    "courier_months": 55,
    "weather": "clear",
    "day_type": "weekday",
    "vehicle": "car",
}


def test_probability_is_between_zero_and_one(valid_order):
    result = predict.predict_order(valid_order)
    assert 0 <= result["probability_late"] <= 1


def test_harder_order_scores_higher_than_easier_order():
    hard = predict.predict_order(HARD_ORDER)["probability_late"]
    easy = predict.predict_order(EASY_ORDER)["probability_late"]
    assert hard > easy


# The threshold in model_info.json is 0.65. A flagged order is one whose
# probability is at least the threshold, so 0.65 itself is flagged.
@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (0.65, True),
        (0.6499, False),
        (0.6501, True),
        (0.0, False),
        (1.0, True),
    ],
)
def test_voucher_decision_at_the_boundary(probability, expected):
    assert predict.should_offer_voucher(probability) is expected


def test_decision_uses_the_threshold_saved_in_model_info(monkeypatch, valid_order):
    result = predict.predict_order(valid_order)
    assert result["threshold"] == predict.INFO["threshold"]

    monkeypatch.setitem(predict.INFO, "threshold", 0.9)
    assert predict.should_offer_voucher(0.7) is False
    monkeypatch.setitem(predict.INFO, "threshold", 0.1)
    assert predict.should_offer_voucher(0.7) is True


def test_missing_prep_minutes_is_accepted_with_a_warning(valid_order):
    # The Lab 05 model does not use prep_minutes, so give the check a feature
    # list that does, to prove the rule works if the model ever uses it.
    info = {
        "numeric_features": ["distance_km", "prep_minutes"],
        "categorical_features": ["weather"],
    }
    clean, warnings = predict.validate_order(
        {**valid_order, "prep_minutes": None}, info
    )
    assert math.isnan(clean["prep_minutes"])
    assert any("prep_minutes" in message for message in warnings)


def test_order_without_prep_minutes_still_scores(valid_order):
    assert "prep_minutes" not in valid_order
    assert predict.predict_order(valid_order)["probability_late"] > 0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("distance_km", 0),
        ("distance_km", 30.5),
        ("distance_km", -4),
        ("items", 0),
        ("items", 21),
        ("items", 2.5),
        ("hour", 9),
        ("hour", 23),
        ("courier_months", -1),
        ("courier_months", 121),
        ("distance_km", float("nan")),
        ("distance_km", float("inf")),
        ("distance_km", "far"),
    ],
)
def test_impossible_number_is_rejected_and_names_the_field(valid_order, field, value):
    valid_order[field] = value
    with pytest.raises(ValueError, match=field):
        predict.predict_order(valid_order)


@pytest.mark.parametrize(
    "field", ["distance_km", "items", "hour", "weather", "vehicle"]
)
@pytest.mark.parametrize("empty", [None, ""])
def test_every_other_missing_field_is_rejected(valid_order, field, empty):
    valid_order[field] = empty
    with pytest.raises(ValueError, match=field):
        predict.predict_order(valid_order)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weather", "fog"),
        ("weather", "Snow"),
        ("day_type", "holiday"),
        ("vehicle", "van"),
    ],
)
def test_unseen_category_is_rejected(valid_order, field, value):
    valid_order[field] = value
    with pytest.raises(ValueError, match=field):
        predict.predict_order(valid_order)


def test_model_file_is_loaded_only_once(monkeypatch, valid_order):
    calls = []
    real_load = joblib.load

    def counting_load(*args, **kwargs):
        calls.append(args)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(joblib, "load", counting_load)
    reloaded = importlib.reload(predict)
    assert len(calls) == 1

    for _ in range(3):
        reloaded.predict_order(valid_order)
    assert len(calls) == 1
