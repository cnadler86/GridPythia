"""Adapter: convert PredictionData -> GeneticEnergyManagementParameters.

This helper converts the prediction dataframe columns and units to the
structure expected by the genetic optimizer: lists of Wh per timestep and
per-PV forecasts mapped by name.
"""

from __future__ import annotations

from typing import Dict

from src.optimization.genetic.geneticparams import GeneticEnergyManagementParameters
from src.prediction.prediction import PredictionData


def prediction_to_genetic_params(
    pred: PredictionData,
    preis_euro_pro_wh_akku: float,
    einspeise_default: float | None = None,
) -> GeneticEnergyManagementParameters:
    """Convert PredictionData into GeneticEnergyManagementParameters.

    Args:
        pred: PredictionData as returned by `Prediction.fetch(...)`.
        preis_euro_pro_wh_akku: battery price in EUR/Wh required by the
            genetic parameter object.
        einspeise_default: fallback feed-in tariff (EUR/Wh) if prediction
            does not provide `feedintariff_eur_wh`.

    Returns:
        GeneticEnergyManagementParameters with fields populated and units
        converted to Wh per timestep.
    """
    # dt in hours used by the prediction window
    dt = pred.dt_hours

    df = pred.df

    # Load: convert W -> Wh for each timestep (power * hours)
    if "load_w" in df.columns:
        load_wh = (df["load_w"] * float(dt)).to_list()
    else:
        load_wh = [0.0] * len(df)

    # Electricity price per Wh (prediction already stores EUR/Wh)
    if "electricprice_eur_wh" in df.columns:
        price_eur_per_wh = df["electricprice_eur_wh"].to_list()
    else:
        price_eur_per_wh = [0.0] * len(df)

    # Feed-in tariff: prefer column, otherwise use provided default or zeros
    if "feedintariff_eur_wh" in df.columns:
        feedin = df["feedintariff_eur_wh"].to_list()
    else:
        if einspeise_default is None:
            feedin = [0.0] * len(df)
        else:
            feedin = [float(einspeise_default)] * len(df)

    # PV columns are named pv_{name}_{inverter}_w; group by pv name
    pv_map: Dict[str, list[float]] = {}
    for c in df.columns:
        if c.startswith("pv_") and c.endswith("_w"):
            # pv_{name}_{inverter}_w -> extract name
            body = c[len("pv_") : -len("_w")]
            # keep the full body as key (name_inverter) so it matches
            # inverter.parameter.pv_source used in the simulation
            pv_map[body] = (df[c] * float(dt)).to_list()

    return GeneticEnergyManagementParameters(
        pv_prognose_wh=pv_map,
        strompreis_euro_pro_wh=price_eur_per_wh,
        einspeiseverguetung_euro_pro_wh=feedin,
        preis_euro_pro_wh_akku=float(preis_euro_pro_wh_akku),
        gesamtlast=load_wh,
    )
