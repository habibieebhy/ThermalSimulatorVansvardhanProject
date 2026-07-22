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

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be greater than zero.")

    if time_step_seconds <= 0:
        raise ValueError("time_step_seconds must be greater than zero.")

    if occupied_area_m2 <= 0:
        raise ValueError("occupied_area_m2 must be greater than zero.")

    if body_contact_conductance_w_k <= 0:
        raise ValueError(
            "body_contact_conductance_w_k must be greater than zero."
        )

    if not candidate.layers:
        raise ValueError(
            f"Configuration {candidate.configuration_id!r} has no layers."
        )

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

    occupied_heat_capacity_j_k = max(
        100.0,
        areal_heat_capacity_j_m2k * occupied_area_m2,
    )

    room_conductance_w_k = occupied_area_m2 / max(
        resistance_per_area,
        1e-9,
    )

    step_count = int(round(duration_seconds / time_step_seconds))
    if step_count < 1:
        raise ValueError(
            "duration_seconds must be at least one time step."
        )

    temperature = np.empty(step_count + 1, dtype=float)
    temperature[0] = ambient_temperature_c

    for index in range(step_count):
        current_temperature_c = temperature[index]

        heat_in_w = body_contact_conductance_w_k * (
            skin_temperature_c - current_temperature_c
        )

        heat_out_w = room_conductance_w_k * (
            current_temperature_c - ambient_temperature_c
        )

        next_temperature_c = current_temperature_c + (
            (heat_in_w - heat_out_w)
            * time_step_seconds
            / occupied_heat_capacity_j_k
        )

        if not np.isfinite(next_temperature_c):
            raise RuntimeError(
                f"Thermal screening became unstable for "
                f"{candidate.configuration_id!r} at step {index}."
            )

        temperature[index + 1] = next_temperature_c

    comfort_samples = int(
        np.count_nonzero(
            (temperature >= 28.0)
            & (temperature <= 32.0)
        )
    )

    final_window_samples = max(
        1,
        min(
            len(temperature),
            int(round(900.0 / time_step_seconds)) + 1,
        ),
    )

    estimated_final_temperature_c = float(
        np.mean(temperature[-final_window_samples:])
    )

    return SimulationScreeningResult(
        configuration_id=candidate.configuration_id,
        thermal_resistance_m2k_w=round(
            resistance_per_area,
            5,
        ),
        areal_heat_capacity_kj_m2k=round(
            areal_heat_capacity_j_m2k / 1_000.0,
            5,
        ),
        estimated_final_interface_temperature_c=round(
            estimated_final_temperature_c,
            3,
        ),
        comfort_zone_minutes=round(
            comfort_samples * time_step_seconds / 60.0,
            2,
        ),
        peak_interface_temperature_c=round(
            float(np.max(temperature)),
            3,
        ),
        screening_only=True,
    )