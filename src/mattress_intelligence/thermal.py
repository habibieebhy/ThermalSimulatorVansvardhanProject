"""Fast passive thermal screening for generated layer stacks."""

from __future__ import annotations

import numpy as np

from .models import ConfigurationCandidate, SimulationScreeningResult


def screen_configuration(
    candidate: ConfigurationCandidate,
    duration_seconds: int = 21_600,
    time_step_seconds: float = 5.0,
    ambient_temperature_c: float = 25.0,
    skin_temperature_c: float = 37.0,
    occupied_area_m2: float = 0.18,
    body_contact_conductance_w_k: float = 1.8,
) -> SimulationScreeningResult:
    """Return comparative lumped-model metrics, never certification predictions."""

    resistance_per_area = sum(
        (layer.thickness_mm / 1_000.0) / layer.conductivity_w_mk
        for layer in candidate.layers
    )
    areal_heat_capacity_j_m2k = sum(
        (layer.thickness_mm / 1_000.0)
        * layer.density_kg_m3
        * layer.specific_heat_j_kgk
        for layer in candidate.layers
    )
    occupied_heat_capacity = max(100.0, areal_heat_capacity_j_m2k * occupied_area_m2)
    room_conductance = occupied_area_m2 / max(resistance_per_area, 1e-9)

    step_count = int(round(duration_seconds / time_step_seconds))
    temperature = np.empty(step_count + 1, dtype=float)
    temperature[0] = ambient_temperature_c
    for index in range(step_count):
        heat_in = body_contact_conductance_w_k * (skin_temperature_c - temperature[index])
        heat_out = room_conductance * (temperature[index] - ambient_temperature_c)
        temperature[index + 1] = temperature[index] + (
            (heat_in - heat_out) * time_step_seconds / occupied_heat_capacity
        )
    comfort_samples = np.count_nonzero((temperature >= 28.0) & (temperature <= 32.0))
    return SimulationScreeningResult(
        configuration_id=candidate.configuration_id,
        thermal_resistance_m2k_w=round(resistance_per_area, 5),
        areal_heat_capacity_kj_m2k=round(areal_heat_capacity_j_m2k / 1_000.0, 5),
        estimated_final_interface_temperature_c=round(float(np.mean(temperature[-181:])), 3),
        comfort_zone_minutes=round(comfort_samples * time_step_seconds / 60.0, 2),
        peak_interface_temperature_c=round(float(np.max(temperature)), 3),
        screening_only=True,
    )

