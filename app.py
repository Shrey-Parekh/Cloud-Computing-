"""Streamlit dashboard: ARIMA workload prediction and VM provisioning.

Run with:  streamlit run app.py
Pipeline:  real Wikimedia data -> cleaning -> ARIMA -> metrics -> VM calculation
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.arima_model import (
    MODE_MULTI_STEP,
    MODE_ONE_STEP,
    ArimaError,
    forecast_future,
    predict_test_period,
    split_train_test,
    suggest_order,
)
from src.data_loader import DEFAULT_DAYS, DataLoadError, get_workload
from src.evaluation import EvaluationError, evaluate
from src.preprocessing import PreprocessingError, to_series
from src.vm_calculator import (
    DEFAULT_VM_CAPACITY,
    SCALE_DOWN,
    SCALE_UP,
    VMCalculationError,
    provisioning_quality,
    provisioning_table,
    recommend_action,
    required_vms,
)
from src.arima_model import SEASONAL

st.set_page_config(page_title="Cloud Workload Prediction", page_icon="📈", layout="wide")

MODE_LABELS = {
    "Rolling one-step-ahead (recommended)": MODE_ONE_STEP,
    "Multi-step (whole test period at once)": MODE_MULTI_STEP,
}
TRAIN_COLOR, TEST_COLOR, PRED_COLOR, BAND_COLOR = "#4C78A8", "#F58518", "#E45756", "rgba(228,87,86,0.15)"


# --------------------------------------------------------------------------- #
# Cached helpers (Streamlit re-runs the script on every widget change, so
# anything slow must be cached to avoid repeated downloads / model fits)
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=3600, show_spinner="Loading workload data from Wikimedia...")
def cached_load(days: int, force_refresh: bool):
    return get_workload(days=days, force_refresh=force_refresh)


@st.cache_data(show_spinner="Training ARIMA and predicting the test period...")
def cached_backtest(data: pd.DataFrame, order: tuple, train_ratio: float, mode: str):
    series = to_series(data)
    train, test = split_train_test(series, train_ratio)
    predicted, aic = predict_test_period(train, test, order, mode)
    return train, test, predicted, aic


@st.cache_data(show_spinner="Fitting ARIMA on all data and forecasting...")
def cached_future(data: pd.DataFrame, order: tuple, steps: int):
    return forecast_future(to_series(data), order, steps)


@st.cache_data(show_spinner="Searching for a good (p, d, q) ...")
def cached_suggestion(data: pd.DataFrame, train_ratio: float):
    train, _ = split_train_test(to_series(data), train_ratio)
    return suggest_order(train)


def fmt(n: float) -> str:
    return f"{n:,.0f}"


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.title("📈 Cloud Workload Prediction with ARIMA")
st.caption(
    "Implements the ARIMA-based workload prediction and VM provisioning concept of "
    "Calheiros, Masoumi, Ranjan & Buyya (IEEE TCC, 2015). It does not reproduce the "
    "paper's exact results. Data: real hourly English Wikipedia pageviews (Wikimedia)."
)

with st.sidebar:
    st.header("Settings")

    st.subheader("Data")
    days = st.number_input("Days of history", min_value=8, max_value=90,
                           value=DEFAULT_DAYS, step=1,
                           help="30 days = about 720 hourly observations.")
    if st.button("🔄 Refresh data from Wikimedia"):
        st.cache_data.clear()
        st.session_state["force_refresh"] = True

    st.subheader("ARIMA order (p, d, q)")
    for _key, _default in (("p", 2), ("d", 1), ("q", 2)):
        st.session_state.setdefault(_key, _default)
    if st.session_state.get("order_pending"):          # set by the "suggest" button
        st.session_state["p"], st.session_state["d"], st.session_state["q"] = \
            st.session_state.pop("order_pending")
    p = st.number_input("p (AR terms)", 0, 10, key="p", step=1)
    d = st.number_input("d (differencing)", 0, 2, key="d", step=1)
    q = st.number_input("q (MA terms)", 0, 10, key="q", step=1)
    suggest_clicked = st.button("Suggest order (ADF test + AIC)")

    st.subheader("Evaluation")
    train_ratio = st.slider("Training share", 0.60, 0.90, 0.80, 0.05,
                            help="Remaining data is the test set (default 80/20).")
    mode_label = st.radio("Test-period prediction", list(MODE_LABELS))
    mode = MODE_LABELS[mode_label]

    st.subheader("Forecast")
    horizon = st.number_input("Hours to forecast", 1, 72, 24, step=1)

    st.subheader("VM provisioning")
    vm_capacity = st.number_input(
        "VM capacity assumption (requests/hour)", min_value=1,
        value=DEFAULT_VM_CAPACITY, step=50_000,
        help="Configurable project assumption used for VM calculations; it is not an AWS VM performance specification.",
    )
    st.caption("This configurable capacity is a project assumption, not an AWS VM performance specification.")
    manual_vms = st.number_input("Currently running VMs (0 = auto)", min_value=0,
                                 value=0, step=1,
                                 help="0 means: the VMs needed for the last observed hour.")

# --------------------------------------------------------------------------- #
# Load data
# --------------------------------------------------------------------------- #
force_refresh = st.session_state.pop("force_refresh", False)
try:
    data, info = cached_load(int(days), force_refresh)
except (DataLoadError, PreprocessingError) as exc:
    st.error(f"Could not load workload data: {exc}")
    st.info("Check your internet connection (the server must reach wikimedia.org) "
            "and press 'Refresh data'.")
    st.stop()

if info["source"] == "stale-cache":
    st.warning(info["message"])
else:
    st.caption(f"Data status: {info['message']}")

series = to_series(data)

# --------------------------------------------------------------------------- #
# Optional: automatic order suggestion
# --------------------------------------------------------------------------- #
if suggest_clicked:
    try:
        best_order, candidates = cached_suggestion(data, train_ratio)
        st.session_state["order_pending"] = best_order
        st.session_state["candidates"] = candidates
        st.rerun()
    except ArimaError as exc:
        st.sidebar.error(str(exc))

if "candidates" in st.session_state:
    with st.expander("Order search results (lower AIC is better, computed on training data)"):
        st.dataframe(st.session_state["candidates"].head(10), hide_index=True)

# --------------------------------------------------------------------------- #
# Data overview
# --------------------------------------------------------------------------- #
st.subheader("1. Workload history")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Hourly observations", len(series))
c2.metric("Average requests/hour", fmt(series.mean()))
c3.metric("Peak requests/hour", fmt(series.max()))
c4.metric("Hours filled (missing data)", int(data["imputed"].sum()))
st.caption(f"Period: {series.index.min():%Y-%m-%d %H:%M} to {series.index.max():%Y-%m-%d %H:%M} UTC")

try:
    train, test, predicted, aic = cached_backtest(data, (int(p), int(d), int(q)),
                                                  float(train_ratio), mode)
except ArimaError as exc:
    st.error(f"ARIMA problem: {exc}")
    st.stop()

fig = go.Figure()
fig.add_trace(go.Scatter(x=train.index, y=train, name="Training data",
                         line=dict(color=TRAIN_COLOR, width=1.5)))
fig.add_trace(go.Scatter(x=test.index, y=test, name="Test data",
                         line=dict(color=TEST_COLOR, width=1.5)))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="Requests per hour", hovermode="x unified",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
evaluation_label = "One-step-ahead" if mode == MODE_ONE_STEP else "Multi-step"
st.subheader(
    f"2. Model accuracy - {evaluation_label} evaluation "
    f"(ARIMA({int(p)},{int(d)},{int(q)}))"
)
try:
    metrics = evaluate(test, predicted)
except EvaluationError as exc:
    st.error(f"Could not compute metrics: {exc}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("MAE", fmt(metrics["MAE"]), help="Average absolute error, in requests/hour.")
m2.metric("RMSE", fmt(metrics["RMSE"]), help="Root mean squared error; punishes big misses.")
m3.metric("MAPE", f"{metrics['MAPE']:.2f}%", help="Average percentage error.")
m4.metric("Project accuracy (100 - MAPE)", f"{metrics['Accuracy']:.2f}%")
model1, model2, model3, model4, model5 = st.columns(5)
model1.metric("ARIMA order", f"({int(p)}, {int(d)}, {int(q)})")
model2.metric("Seasonal order", str(SEASONAL))
model3.metric("Evaluation mode", evaluation_label)
model4.metric("Training observations", len(train))
model5.metric("Test observations", len(test))
st.caption(f"Training AIC: {aic:,.1f} | Selected control: {mode_label}")

fig = go.Figure()
fig.add_trace(go.Scatter(x=test.index, y=test, name="Actual",
                         line=dict(color=TEST_COLOR, width=2)))
fig.add_trace(go.Scatter(x=predicted.index, y=predicted, name="Predicted",
                         line=dict(color=PRED_COLOR, width=2, dash="dash")))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="Requests per hour", hovermode="x unified",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig)

# --------------------------------------------------------------------------- #
# Provisioning quality in the test period
# --------------------------------------------------------------------------- #
try:
    quality = provisioning_quality(test, predicted, float(vm_capacity))
except VMCalculationError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("How good would the VM provisioning have been on the test set?"):
    q1, q2, q3 = st.columns(3)
    q1.metric("Under-provisioned hours", quality["under_provisioned_hours"],
              help="Predicted VMs < VMs really needed (risk of QoS violation).")
    q2.metric("Over-provisioned hours", quality["over_provisioned_hours"],
              help="Predicted VMs > VMs really needed (wasted cost).")
    q3.metric("Exactly right hours", quality["exact_hours"])
    st.caption("Computed with ceil(requests / VM capacity) for both the actual and the "
               "predicted workload of every test hour.")

# --------------------------------------------------------------------------- #
# Future forecast and VM recommendation
# --------------------------------------------------------------------------- #
st.subheader(f"3. Future workload forecast and VM recommendation (next {int(horizon)} hours)")
try:
    forecast = cached_future(data, (int(p), int(d), int(q)), int(horizon))
    last_actual = float(series.iloc[-1])
    current_vms = int(manual_vms) if manual_vms > 0 else required_vms(last_actual, float(vm_capacity))
    plan = provisioning_table(forecast["predicted"], float(vm_capacity), current_vms)
except (ArimaError, VMCalculationError) as exc:
    st.error(str(exc))
    st.stop()

next_pred = float(forecast["predicted"].iloc[0])
next_vms = int(plan["required_vms"].iloc[0])
action = recommend_action(next_vms, current_vms)
peak_vms = int(plan["required_vms"].max())

f1, f2, f3, f4 = st.columns(4)
f1.metric("Predicted workload (next hour)", fmt(next_pred))
f2.metric("Required VMs (next hour)", next_vms, delta=next_vms - current_vms,
          delta_color="off")
f3.metric("Current VMs", current_vms)
f4.metric(f"Peak VMs (next {int(horizon)} h)", peak_vms)

message = (f"Recommendation for the next hour: **{action}** "
           f"({current_vms} → {next_vms} VMs, capacity {int(vm_capacity):,} requests/hour per VM).")
if action == SCALE_UP:
    st.warning(message)
elif action == SCALE_DOWN:
    st.info(message)
else:
    st.success(message)
st.caption("This is only a calculation. No virtual machine is launched or stopped.")

history_tail = series.iloc[-72:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=history_tail.index, y=history_tail, name="Recent actual",
                         line=dict(color=TRAIN_COLOR, width=2)))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast["upper"], line=dict(width=0),
                         showlegend=False, hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast["lower"], fill="tonexty",
                         fillcolor=BAND_COLOR, line=dict(width=0), name="95% interval",
                         hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast["predicted"], name="Forecast",
                         line=dict(color=PRED_COLOR, width=2, dash="dash")))
fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                  yaxis_title="Requests per hour", hovermode="x unified",
                  legend=dict(orientation="h", y=1.1))
st.plotly_chart(fig)

st.dataframe(
    plan.rename(columns={
        "timestamp": "Hour (UTC)", "predicted_requests": "Predicted requests",
        "required_vms": "Required VMs", "change_vs_previous": "Change vs previous hour",
        "action": "Action", "utilization_pct": "Utilization %",
    }),
    hide_index=True,
)
st.download_button("⬇ Download forecast plan (CSV)", plan.to_csv(index=False),
                   file_name="vm_provisioning_plan.csv", mime="text/csv")

with st.expander("How the numbers are calculated"):
    st.markdown(
        f"""
- **ARIMA({int(p)},{int(d)},{int(q)}) with seasonal order {SEASONAL}** is fitted on the first {train_ratio:.0%} of the data and tested on the rest using {evaluation_label.lower()} evaluation.
- **MAE** = mean(|actual − predicted|), **RMSE** = √mean((actual − predicted)²),
  **MAPE** = mean(|actual − predicted| / actual) × 100, **Project accuracy** = 100 − MAPE.
- **Required VMs** = ceil(predicted requests / VM capacity) with capacity = {int(vm_capacity):,}.
- **Action**: more VMs needed than before → Scale Up; fewer → Scale Down; same → Maintain.
        """
    )
