---
title: QuickBite Late Order Check
emoji: 🛵
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.31.0
python_version: 3.12
app_file: app.py
pinned: false
short_description: Will this order arrive late?
---

# QuickBite Late Order Check

## What it does

A dispatcher types in what is known when an order is placed, and the page says how likely the order is to arrive late and whether to offer the customer a $4 voucher. The model behind it is the Lab 05 scikit-learn pipeline, loaded as it was saved.

Live page: https://huggingface.co/spaces/amreleraqi09/lab06-ship-it

## How to run it locally

```bash
conda env create -f environment.yml
conda activate lab06-ship-it
gradio app.py
```

`gradio app.py` reloads the page when a file is saved. `python app.py` works too. Open the local address it prints.

## Decisions

The page builds its inputs from the feature lists in `models/model_info.json`, so the form cannot drift away from what the model was trained on. `predict.py` checks every value before the model sees it, and the page shows the message it raises. All of the code that touches the model is in `predict.py`; `app.py` only calls `predict_order()`.

| Field | Allowed | Why |
| --- | --- | --- |
| `distance_km` | A number from 0.1 to 30 | Zero or a negative distance is a typing mistake. 30 km is longer than any city delivery. |
| `items` | A whole number from 1 to 20 | An order has at least one item, and half an item makes no sense. |
| `hour` | A whole number from 10 to 22 | The restaurants take orders between 10 and 22, which is the range the model saw. |
| `courier_months` | A whole number from 0 to 120 | A new courier has 0 months. Ten years is a generous ceiling. |
| `weather` | clear, rain or snow | These are the only three values in the training data. |
| `day_type` | weekday or weekend | Same reason. |
| `vehicle` | bike, scooter or car | Same reason. |

Missing values are refused for every field, with a message that names the field. `prep_minutes` is the one field that may be empty in `predict.py`, because the pipeline fills it with the training median and the page warns that it did. The Lab 05 model does not use `prep_minutes` at all, since the kitchen time is not known when an order is placed, so the page does not ask for it. `order_id` and `customer_rating` are left out for the Lab 05 reasons: one is only a label, and the other is given after delivery. If a later model uses `prep_minutes`, the rule is already in place and a test covers it.

**The threshold.** The voucher is offered when the chance of a late order is at least the threshold in `model_info.json` (0.65). The number is read from that file and is not typed anywhere in the code. A test changes it in memory and checks that the decision follows. It was chosen in Lab 05 as the cheapest threshold under the cost rule ($4 per voucher, $15 per late order that gets no voucher), so it is not 0.5 on purpose.

**A misspelled category is refused, not scored.** The one-hot encoder in the pipeline was set to ignore categories it has not seen. That is fine in a training script, but on a page it hides mistakes. I scored the same order twice by calling the pipeline directly: a weekend order, 3 km, 4 items, placed at 19:00 on a bike by a courier with no experience.

| Weather typed as | Chance of late | Voucher at 0.65 |
| --- | --- | --- |
| `snow` | 65.6% | Yes |
| `Snow` | 61.4% | No |

With `Snow` the encoder saw an unknown weather value and scored the order as if the weather were none of the three, so the answer changed from yes to no and nothing looked wrong. The page offers choices only, and `predict.py` refuses anything outside the list, including `Snow`, so this cannot happen quietly.

## Result

These are the Lab 05 numbers, on the 750 orders the model never saw (25 percent of 3,000). The test set was scored once.

| | Predicted on time | Predicted late |
| --- | --- | --- |
| Arrived on time | 586 | 47 |
| Arrived late | 73 | 44 |

Precision 0.48, recall 0.38, F1 0.42, ROC-AUC 0.77, with 91 orders flagged.

| Policy | Cost on the 750 test orders |
| --- | --- |
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| This model at 0.65 | $1,459 |

The model saves $296, about 17 percent below flagging nobody. That is a real gain on data it had not seen, but it comes from one test set of 750 orders, so I would trust the direction more than the exact figure.

## Tests

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

The tests in `tests/test_predict.py` cover the behaviour that matters to a dispatcher: the probability stays between 0 and 1, a harder order scores higher than an easy one, the voucher decision at the boundary (exactly the threshold, just below, just above), the threshold coming from `model_info.json`, an empty `prep_minutes` being accepted, each impossible number being rejected with the field named, each other missing field being rejected, an unseen category being rejected, and the model file being loaded once.

`tests/test_deployment.py` checks that the files a Space reads agree: the installed scikit-learn version matches `model_info.json` and the pin in `requirements.txt`, Gradio is not in `requirements.txt`, `sdk_version` here matches the installed Gradio, and the `app_file` exists.

One thing I chose not to test is the look of the page, such as labels and layout. Checking that takes a browser, and the checks above protect the decisions that cost money.

## Limitations

- The model was trained on 3,000 invented orders, not real QuickBite deliveries. The numbers above say how it behaves on that data and nothing more.
- It cannot see anything that happens after the order is placed, such as a road closure, a courier who gets lost or a restaurant that runs out of an item. A dispatcher should not use it to judge an order that is already late.
- It misses most late orders. Recall is 0.38, so about six late orders in ten get no voucher. The threshold is cheap under the current costs, not accurate.
- The page accepts distances up to 30 km, up to 20 items and couriers up to 120 months, but the training data only reached about 15 km, 8 items and 59 months. Beyond those values the answer is an extrapolation.
- If the voucher cost or the refund cost changes, the threshold has to be chosen again in Lab 05 and the files copied over. Editing the number here would break the link to the cost rule.
- A free Space goes to sleep when nobody uses it, so the first visit after a quiet period takes a minute to wake up.
