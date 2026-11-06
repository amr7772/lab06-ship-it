import pytest


@pytest.fixture
def valid_order():
    """An ordinary order that every check should accept."""
    return {
        "distance_km": 3.2,
        "items": 2,
        "hour": 15,
        "courier_months": 12,
        "weather": "clear",
        "day_type": "weekday",
        "vehicle": "scooter",
    }
