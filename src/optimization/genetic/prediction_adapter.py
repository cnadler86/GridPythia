"""Adapter: convert PredictionData -> GeneticEnergyManagementParameters.

This helper converts the prediction dataframe columns and units to the
structure expected by the genetic optimizer: lists of Wh per timestep and
per-PV forecasts mapped by name.
"""

from __future__ import annotations

from typing import Dict

from src.optimization.genetic.params import EnergyManagementParameters
from src.prediction.prediction import PredictionData


def prediction_to_genetic_params(
    pred: PredictionData,
    preis_euro_pro_wh_akku: float,
    einspeise_default: float | None = None,
) -> EnergyManagementParameters:
    """Convert PredictionData into EnergyManagementParameters.

    Args:
        pred: PredictionData as returned by `Prediction.fetch(...)`.
        preis_euro_pro_wh_akku: battery price in EUR/Wh required by the
            energy management parameter object.
        einspeise_default: fallback feed-in tariff (EUR/Wh) if prediction
            does not provide `feedintariff_eur_wh`.

    Returns:
        EnergyManagementParameters with fields populated in Wh per timestep.
    """
    n_steps = pred.steps

    # Load is already in Wh (no conversion needed)
    load_wh = pred.load_wh.to_list()

    # Electricity price per Wh (prediction already stores EUR/Wh)
    if pred.electricprice is not None:
        price_eur_per_wh = pred.electricprice.to_list()
    else:
        price_eur_per_wh = [0.0] * n_steps

    # Feed-in tariff: prefer column, otherwise use provided default or zeros
    if pred.feedintariff is not None:
        feedin = pred.feedintariff.to_list()
    else:
        if einspeise_default is None:
            feedin = [0.0] * n_steps
        else:
            feedin = [float(einspeise_default)] * n_steps

    # PV columns are named pv_{inverter_id}_wh; already in Wh (no conversion needed)
    pv_map: Dict[str, list[float]] = {}
    for inverter_id, series in pred.pv_by_inverter.items():
        pv_map[inverter_id] = series.to_list()

    return EnergyManagementParameters(
        pv_prognose_wh=pv_map,
        strompreis_euro_pro_wh=price_eur_per_wh,
        einspeiseverguetung_euro_pro_wh=feedin,
        preis_euro_pro_wh_akku=float(preis_euro_pro_wh_akku),
        gesamtlast=load_wh,
    )
