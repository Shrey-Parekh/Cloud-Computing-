"""VM provisioning calculation.

Formula (from the project brief):

    required_vms = ceil(predicted_requests / vm_capacity)

This module only CALCULATES a recommendation. It never launches any VM.
"""

from __future__ import annotations

import math

import pandas as pd

DEFAULT_VM_CAPACITY = 500_000  # configurable project assumption, requests/hour

SCALE_UP = "Scale Up"
SCALE_DOWN = "Scale Down"
MAINTAIN = "Maintain"


class VMCalculationError(Exception):
    """Raised for invalid VM calculation inputs."""


def required_vms(predicted_requests: float, vm_capacity: float = DEFAULT_VM_CAPACITY,
                 min_vms: int = 1) -> int:
    """Number of VMs needed for ``predicted_requests`` requests per hour.

    ``min_vms`` keeps at least one VM running (default 1). It only matters if
    the prediction is zero or negative.
    """
    if vm_capacity is None or vm_capacity <= 0:
        raise VMCalculationError("VM capacity must be greater than 0.")
    if predicted_requests is None or math.isnan(predicted_requests):
        raise VMCalculationError("Predicted requests is not a valid number.")
    needed = math.ceil(max(predicted_requests, 0.0) / vm_capacity)
    return max(int(needed), int(min_vms))


def recommend_action(required: int, current: int) -> str:
    """Compare the required VM count with the current one."""
    if required > current:
        return SCALE_UP
    if required < current:
        return SCALE_DOWN
    return MAINTAIN


def provisioning_table(predicted: pd.Series, vm_capacity: float,
                       current_vms: int) -> pd.DataFrame:
    """Build the hour-by-hour provisioning plan for a forecast.

    Columns
    -------
    predicted_requests : forecast for that hour
    required_vms       : ceil(predicted / capacity)
    change_vs_previous : difference to the previous hour's VM count
                         (the first hour is compared with ``current_vms``)
    action             : Scale Up / Scale Down / Maintain
    utilization_pct    : predicted load as % of the capacity of the VMs
    """
    if current_vms < 0:
        raise VMCalculationError("Current VM count cannot be negative.")

    needed = [required_vms(v, vm_capacity) for v in predicted.to_numpy()]
    previous = [int(current_vms)] + needed[:-1]

    table = pd.DataFrame(
        {
            "timestamp": predicted.index,
            "predicted_requests": predicted.to_numpy().round(0).astype(int),
            "required_vms": needed,
        }
    )
    table["change_vs_previous"] = table["required_vms"] - pd.Series(previous)
    table["action"] = [recommend_action(n, c) for n, c in zip(needed, previous)]
    table["utilization_pct"] = (
        table["predicted_requests"] / (table["required_vms"] * vm_capacity) * 100
    ).round(1)
    return table


def provisioning_quality(actual: pd.Series, predicted: pd.Series,
                         vm_capacity: float) -> dict:
    """Compare VMs provisioned from predictions with VMs actually needed.

    Under-provisioning (too few VMs) risks QoS violations; over-provisioning
    wastes money. This is the trade-off discussed in the paper.
    """
    need_actual = [required_vms(v, vm_capacity) for v in actual.to_numpy()]
    need_pred = [required_vms(v, vm_capacity) for v in predicted.to_numpy()]
    under = sum(p < a for p, a in zip(need_pred, need_actual))
    over = sum(p > a for p, a in zip(need_pred, need_actual))
    exact = len(need_actual) - under - over
    return {
        "hours": len(need_actual),
        "under_provisioned_hours": under,
        "over_provisioned_hours": over,
        "exact_hours": exact,
    }
