"""The dispatchers' page. It talks to the model only through predict.py."""

import gradio as gr

import predict

# The free ZeroGPU hardware refuses to start an app with no GPU function.
# This placeholder does nothing; it only lets the Space start. On a laptop
# the spaces package is not installed, so the import error is ignored.
try:
    import spaces

    @spaces.GPU
    def gpu_placeholder():
        return None

except ImportError:
    pass

NUMBER_LABELS = {
    "distance_km": "Distance from restaurant to customer (km)",
    "items": "Number of items",
    "hour": "Hour the order was placed",
    "courier_months": "Months the courier has worked for QuickBite",
    "prep_minutes": "Kitchen preparation time (minutes, may be left empty)",
}
CHOICE_LABELS = {
    "weather": "Weather",
    "day_type": "Day type",
    "vehicle": "Courier vehicle",
}


def build_inputs():
    """One input per feature in model_info.json, in the same order."""
    numeric, categorical = predict.feature_lists()
    inputs = []
    for name in numeric:
        rule = predict.NUMBER_LIMITS[name]
        kind = "whole number" if rule["whole"] else "number"
        # No minimum or maximum on the component: predict.py does the check,
        # so the message always names the field and the rule.
        hint = f"{kind}, {rule['low']} to {rule['high']}"
        inputs.append(gr.Number(label=f"{NUMBER_LABELS[name]} [{hint}]", value=None))
    for name in categorical:
        inputs.append(
            gr.Radio(
                choices=predict.CHOICES[name],
                label=CHOICE_LABELS[name],
                value=None,
            )
        )
    return inputs, numeric + categorical


INPUTS, FIELD_NAMES = build_inputs()


def check_order(*values):
    """Score the order typed on the page and return the two outputs."""
    order = dict(zip(FIELD_NAMES, values, strict=True))
    try:
        result = predict.predict_order(order)
    except ValueError as problem:
        raise gr.Error(str(problem)) from problem

    for message in result["warnings"]:
        gr.Warning(message)

    late = result["probability_late"]
    chances = {"Late": late, "On time": 1 - late}
    if result["offer_voucher"]:
        action = (
            "Offer the customer the $4 voucher now, with an apology for a likely delay."
        )
    else:
        action = "No voucher needed. This order is likely to arrive on time."
    return chances, action


with gr.Blocks(title="QuickBite Late Order Check") as demo:
    gr.Markdown(
        "# QuickBite Late Order Check\n"
        "Enter what you know when the order is placed. The page tells you "
        "whether to offer the customer a $4 voucher."
    )
    with gr.Row():
        with gr.Column():
            for component in INPUTS:
                component.render()
            button = gr.Button("Check this order", variant="primary")
        with gr.Column():
            chances_out = gr.Label(label="Chance of each outcome")
            action_out = gr.Textbox(label="What to do", lines=2)
    button.click(check_order, inputs=INPUTS, outputs=[chances_out, action_out])

if __name__ == "__main__":
    demo.launch()
