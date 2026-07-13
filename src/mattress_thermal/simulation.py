"""Object-oriented thermal simulation for five mattress prototypes.

The model uses a lumped thermal capacitance for the occupied mattress zone.
At every time step, conductive heat flow is calculated with
``q = k * A * (T_hot - T_cold) / dx``. Active devices are represented as
capacity-limited conductive paths so their thermal removal can never exceed
either the physical path or the available device capacity.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Physical constants and numerical settings for one simulation run."""

    duration_seconds: int = 21_600
    time_step_seconds: float = 1.0
    skin_temperature_c: float = 37.0
    ambient_temperature_c: float = 25.0
    mattress_zone_mass_kg: float = 0.6
    foam_specific_heat_j_per_kg_k: float = 1_600.0

    def __post_init__(self) -> None:
        positive_values = {
            "duration_seconds": self.duration_seconds,
            "time_step_seconds": self.time_step_seconds,
            "mattress_zone_mass_kg": self.mattress_zone_mass_kg,
            "foam_specific_heat_j_per_kg_k": self.foam_specific_heat_j_per_kg_k,
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero; received {value}.")

        step_count = self.duration_seconds / self.time_step_seconds
        if not np.isclose(step_count, round(step_count)):
            raise ValueError(
                "duration_seconds must be evenly divisible by time_step_seconds."
            )
        if self.skin_temperature_c <= self.ambient_temperature_c:
            raise ValueError(
                "skin_temperature_c must exceed ambient_temperature_c for this "
                "body-heating model."
            )

    @property
    def heat_capacity_j_per_k(self) -> float:
        return self.mattress_zone_mass_kg * self.foam_specific_heat_j_per_kg_k

    @property
    def number_of_steps(self) -> int:
        return int(round(self.duration_seconds / self.time_step_seconds))


@dataclass(frozen=True, slots=True)
class ThermalPath:
    """One-dimensional conductive heat-transfer path."""

    conductivity_w_per_m_k: float
    area_m2: float
    distance_m: float
    description: str

    def __post_init__(self) -> None:
        for name, value in (
            ("conductivity_w_per_m_k", self.conductivity_w_per_m_k),
            ("area_m2", self.area_m2),
            ("distance_m", self.distance_m),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero; received {value}.")

    @property
    def conductance_w_per_k(self) -> float:
        return (
            self.conductivity_w_per_m_k * self.area_m2 / self.distance_m
        )

    def heat_flow_w(self, hot_temperature_c: float, cold_temperature_c: float) -> float:
        return self.conductance_w_per_k * (
            hot_temperature_c - cold_temperature_c
        )


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Time-series output for one mattress architecture."""

    architecture_name: str
    legend_label: str
    color: str
    line_style: str
    time_seconds: np.ndarray
    temperature_c: np.ndarray
    power_w: np.ndarray
    cumulative_energy_wh: np.ndarray

    @property
    def total_energy_wh(self) -> float:
        return float(self.cumulative_energy_wh[-1])

    def stabilized_temperature_c(self, window_seconds: float = 900.0) -> float:
        """Return the mean interface temperature over the ending time window."""

        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")
        window_start = max(0.0, float(self.time_seconds[-1]) - window_seconds)
        samples = self.temperature_c[self.time_seconds >= window_start]
        return float(np.mean(samples))


class MattressPrototype(ABC):
    """Base class shared by all prototype thermal architectures."""

    architecture_name: str
    legend_label: str
    color: str
    line_style: str = "-"

    def __init__(self, passive_rejection_path: ThermalPath) -> None:
        self.passive_rejection_path = passive_rejection_path
        self.body_contact_path = ThermalPath(
            conductivity_w_per_m_k=0.15,
            area_m2=0.18,
            distance_m=0.015,
            description="skin-to-mattress occupied contact zone",
        )

    def initial_state(self, config: SimulationConfig) -> dict[str, Any]:
        return {}

    @abstractmethod
    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        """Return the electrical power demanded during the current interval."""

    def additional_heat_rejection_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        electrical_power_w: float,
        net_heat_before_device_w: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        """Return heat removed in addition to the passive conductive path."""

        return 0.0

    def simulate(self, config: SimulationConfig) -> SimulationResult:
        number_of_steps = config.number_of_steps
        dt = config.time_step_seconds
        time_seconds = np.linspace(
            0.0,
            float(config.duration_seconds),
            number_of_steps + 1,
            dtype=float,
        )
        temperature_c = np.empty(number_of_steps + 1, dtype=float)
        power_w = np.empty(number_of_steps + 1, dtype=float)
        cumulative_energy_wh = np.zeros(number_of_steps + 1, dtype=float)
        temperature_c[0] = config.ambient_temperature_c
        state = self.initial_state(config)

        for step_index in range(number_of_steps):
            elapsed_seconds = time_seconds[step_index]
            current_temperature_c = temperature_c[step_index]

            body_heat_input_w = self.body_contact_path.heat_flow_w(
                config.skin_temperature_c,
                current_temperature_c,
            )
            passive_heat_rejection_w = self.passive_rejection_path.heat_flow_w(
                current_temperature_c,
                config.ambient_temperature_c,
            )
            net_heat_before_device_w = (
                body_heat_input_w - passive_heat_rejection_w
            )

            current_power_w = float(
                self.electrical_power_w(
                    elapsed_seconds,
                    current_temperature_c,
                    state,
                    config,
                )
            )
            if not np.isfinite(current_power_w) or current_power_w < 0:
                raise RuntimeError(
                    f"{self.architecture_name} produced invalid electrical power "
                    f"at t={elapsed_seconds:.1f} s: {current_power_w}."
                )

            additional_rejection_w = float(
                self.additional_heat_rejection_w(
                    elapsed_seconds,
                    current_temperature_c,
                    current_power_w,
                    net_heat_before_device_w,
                    state,
                    config,
                )
            )
            if not np.isfinite(additional_rejection_w) or additional_rejection_w < 0:
                raise RuntimeError(
                    f"{self.architecture_name} produced invalid heat rejection "
                    f"at t={elapsed_seconds:.1f} s: {additional_rejection_w}."
                )

            net_heat_w = net_heat_before_device_w - additional_rejection_w
            temperature_change_c = (
                net_heat_w * dt / config.heat_capacity_j_per_k
            )
            next_temperature_c = current_temperature_c + temperature_change_c
            if not np.isfinite(next_temperature_c) or not -50.0 <= next_temperature_c <= 100.0:
                raise RuntimeError(
                    f"{self.architecture_name} became numerically unstable at "
                    f"t={elapsed_seconds:.1f} s."
                )

            power_w[step_index] = current_power_w
            temperature_c[step_index + 1] = next_temperature_c
            cumulative_energy_wh[step_index + 1] = (
                cumulative_energy_wh[step_index] + current_power_w * dt / 3_600.0
            )

        power_w[-1] = self.electrical_power_w(
            time_seconds[-1],
            temperature_c[-1],
            state,
            config,
        )

        return SimulationResult(
            architecture_name=self.architecture_name,
            legend_label=self.legend_label,
            color=self.color,
            line_style=self.line_style,
            time_seconds=time_seconds,
            temperature_c=temperature_c,
            power_w=power_w,
            cumulative_energy_wh=cumulative_energy_wh,
        )


class AeroNaturalPrototype(MattressPrototype):
    architecture_name = "P1: Aero-Natural (Passive)"
    legend_label = "P1 Aero-Natural · 0 W"
    color = "#D1495B"
    line_style = "--"

    def __init__(
        self,
        latex_conductivity_w_per_m_k: float = 0.035,
        rejection_area_m2: float = 0.15,
        rejection_distance_m: float = 0.015,
        pcm_phase_temperature_c: float = 28.5,
        pcm_capacity_kj: float = 73.5,
        pcm_max_absorption_w: float = 15.0,
    ) -> None:
        super().__init__(
            ThermalPath(
                conductivity_w_per_m_k=latex_conductivity_w_per_m_k,
                area_m2=rejection_area_m2,
                distance_m=rejection_distance_m,
                description="open-cell latex to room air",
            )
        )
        if pcm_capacity_kj < 0.0 or pcm_max_absorption_w < 0.0:
            raise ValueError("PCM capacity and absorption power cannot be negative.")
        self.pcm_phase_temperature_c = pcm_phase_temperature_c
        self.pcm_capacity_j = pcm_capacity_kj * 1_000.0
        self.pcm_max_absorption_w = pcm_max_absorption_w

    def initial_state(self, config: SimulationConfig) -> dict[str, Any]:
        return {"pcm_energy_remaining_j": self.pcm_capacity_j}

    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        return 0.0

    def additional_heat_rejection_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        electrical_power_w: float,
        net_heat_before_device_w: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        remaining_energy_j = float(state["pcm_energy_remaining_j"])
        if remaining_energy_j <= 0.0:
            state["pcm_energy_remaining_j"] = 0.0
            return 0.0

        # This limit prevents the phase-change sink from pulling the foam below
        # the PCM transition temperature during a one-second update.
        allowable_rejection_w = max(
            0.0,
            net_heat_before_device_w
            + config.heat_capacity_j_per_k
            * (interface_temperature_c - self.pcm_phase_temperature_c)
            / config.time_step_seconds,
        )
        absorbed_heat_w = min(
            self.pcm_max_absorption_w,
            remaining_energy_j / config.time_step_seconds,
            allowable_rejection_w,
        )
        state["pcm_energy_remaining_j"] = max(
            0.0,
            remaining_energy_j - absorbed_heat_w * config.time_step_seconds,
        )
        return absorbed_heat_w


class EcoBatteryPrototype(MattressPrototype):
    architecture_name = "P2: Eco-Battery (5W Low Power)"
    legend_label = "P2 Eco-Battery · 5 W"
    color = "#0077B6"

    def __init__(
        self,
        pump_power_w: float = 5.0,
        loop_conductivity_w_per_m_k: float = 0.70,
        radiator_area_m2: float = 0.12,
        flow_path_distance_m: float = 0.030,
    ) -> None:
        super().__init__(
            ThermalPath(
                conductivity_w_per_m_k=loop_conductivity_w_per_m_k,
                area_m2=radiator_area_m2,
                distance_m=flow_path_distance_m,
                description="circulated ambient-water loop and aluminum radiator",
            )
        )
        if pump_power_w < 0.0:
            raise ValueError("pump_power_w cannot be negative.")
        self.pump_power_w = pump_power_w
        self.legend_label = f"P2 Eco-Battery · {pump_power_w:g} W"

    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        return self.pump_power_w


class CoreChillerPrototype(MattressPrototype):
    architecture_name = "P3: Core-Chiller (60W AC Active)"
    legend_label = "P3 Core-Chiller · 60 W max"
    color = "#6F2DBD"

    def __init__(
        self,
        target_temperature_c: float = 29.5,
        maximum_electrical_power_w: float = 60.0,
        peltier_cop: float = 0.65,
        proportional_gain_w_per_k: float = 8.0,
        cold_side_temperature_c: float = 18.0,
    ) -> None:
        super().__init__(
            ThermalPath(
                conductivity_w_per_m_k=0.12,
                area_m2=0.15,
                distance_m=0.030,
                description="premium foam shell to room air",
            )
        )
        if maximum_electrical_power_w < 0.0:
            raise ValueError("maximum_electrical_power_w cannot be negative.")
        if peltier_cop <= 0.0 or proportional_gain_w_per_k < 0.0:
            raise ValueError("Peltier COP must be positive and controller gain non-negative.")
        self.target_temperature_c = target_temperature_c
        self.maximum_electrical_power_w = maximum_electrical_power_w
        self.peltier_cop = peltier_cop
        self.proportional_gain_w_per_k = proportional_gain_w_per_k
        self.cold_side_temperature_c = cold_side_temperature_c
        self.legend_label = f"P3 Core-Chiller · {maximum_electrical_power_w:g} W max"
        self.cold_side_path = ThermalPath(
            conductivity_w_per_m_k=15.0,
            area_m2=0.015,
            distance_m=0.020,
            description="Peltier water block to chilled loop",
        )

    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        predicted_body_load_w = self.body_contact_path.heat_flow_w(
            config.skin_temperature_c,
            self.target_temperature_c,
        )
        predicted_passive_rejection_w = self.passive_rejection_path.heat_flow_w(
            self.target_temperature_c,
            config.ambient_temperature_c,
        )
        holding_load_w = max(
            0.0,
            predicted_body_load_w - predicted_passive_rejection_w,
        )
        requested_thermal_rejection_w = max(
            0.0,
            holding_load_w
            + self.proportional_gain_w_per_k
            * (interface_temperature_c - self.target_temperature_c),
        )
        requested_electrical_power_w = (
            requested_thermal_rejection_w / self.peltier_cop
        )
        return float(
            np.clip(
                requested_electrical_power_w,
                0.0,
                self.maximum_electrical_power_w,
            )
        )

    def additional_heat_rejection_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        electrical_power_w: float,
        net_heat_before_device_w: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        conductive_capacity_w = max(
            0.0,
            self.cold_side_path.heat_flow_w(
                interface_temperature_c,
                self.cold_side_temperature_c,
            ),
        )
        device_capacity_w = self.peltier_cop * electrical_power_w
        return min(conductive_capacity_w, device_capacity_w)


class HyperConductivePrototype(MattressPrototype):
    architecture_name = "P4: Hyper-Conductive (Zero Power Graphite)"
    legend_label = "P4 Hyper-Conductive · 0 W"
    color = "#E76F51"
    line_style = "-."

    def __init__(
        self,
        graphite_conductivity_w_per_m_k: float = 15.0,
        spreader_area_m2: float = 0.012,
        edge_distance_m: float = 0.080,
    ) -> None:
        super().__init__(
            ThermalPath(
                conductivity_w_per_m_k=graphite_conductivity_w_per_m_k,
                area_m2=spreader_area_m2,
                distance_m=edge_distance_m,
                description="flexible graphite heat spreader to exposed bed edges",
            )
        )

    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        return 0.0


class DualZoneSmartMeshPrototype(MattressPrototype):
    architecture_name = "P5: Dual-Zone Smart Mesh (Hybrid)"
    legend_label = "P5 Dual-Zone · 40 W → 10 W pulse"
    color = "#008B8B"

    def __init__(
        self,
        turbo_duration_minutes: float = 60.0,
        turbo_power_w: float = 40.0,
        eco_pulse_power_w: float = 10.0,
        pulse_period_seconds: float = 60.0,
        pulse_on_seconds: float = 30.0,
        cooling_coupling: float = 0.35,
    ) -> None:
        super().__init__(
            ThermalPath(
                conductivity_w_per_m_k=0.42,
                area_m2=0.12,
                distance_m=0.036,
                description="smart mesh and distributed microtube field",
            )
        )
        if turbo_duration_minutes < 0.0:
            raise ValueError("turbo_duration_minutes cannot be negative.")
        if turbo_power_w < 0.0 or eco_pulse_power_w < 0.0:
            raise ValueError("Hybrid electrical power values cannot be negative.")
        if pulse_period_seconds <= 0.0 or not 0.0 <= pulse_on_seconds <= pulse_period_seconds:
            raise ValueError("Pulse on-time must be between zero and the pulse period.")
        if cooling_coupling < 0.0:
            raise ValueError("cooling_coupling cannot be negative.")
        self.turbo_duration_seconds = turbo_duration_minutes * 60.0
        self.turbo_power_w = turbo_power_w
        self.eco_pulse_power_w = eco_pulse_power_w
        self.pulse_period_seconds = pulse_period_seconds
        self.pulse_on_seconds = pulse_on_seconds
        self.cooling_coupling = cooling_coupling
        self.legend_label = (
            f"P5 Dual-Zone · {turbo_power_w:g} W → {eco_pulse_power_w:g} W pulse"
        )
        self.activation_floor_c = 27.8
        self.full_activation_c = 30.0
        self.loop_temperature_c = 20.0
        self.active_loop_path = ThermalPath(
            conductivity_w_per_m_k=0.62,
            area_m2=0.08,
            distance_m=0.020,
            description="dual-zone liquid microtube cooling loop",
        )

    def electrical_power_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        if elapsed_seconds < self.turbo_duration_seconds:
            return self.turbo_power_w
        eco_elapsed_seconds = elapsed_seconds - self.turbo_duration_seconds
        pulse_phase_seconds = eco_elapsed_seconds % self.pulse_period_seconds
        return self.eco_pulse_power_w if pulse_phase_seconds < self.pulse_on_seconds else 0.0

    def additional_heat_rejection_w(
        self,
        elapsed_seconds: float,
        interface_temperature_c: float,
        electrical_power_w: float,
        net_heat_before_device_w: float,
        state: dict[str, Any],
        config: SimulationConfig,
    ) -> float:
        activation_fraction = float(
            np.clip(
                (interface_temperature_c - self.activation_floor_c)
                / (self.full_activation_c - self.activation_floor_c),
                0.0,
                1.0,
            )
        )
        conductive_capacity_w = max(
            0.0,
            self.active_loop_path.heat_flow_w(
                interface_temperature_c,
                self.loop_temperature_c,
            ),
        )
        coupled_device_capacity_w = self.cooling_coupling * electrical_power_w
        return activation_fraction * min(
            conductive_capacity_w,
            coupled_device_capacity_w,
        )


def build_prototypes() -> tuple[MattressPrototype, ...]:
    """Construct the five required architectures in presentation order."""

    return (
        AeroNaturalPrototype(),
        EcoBatteryPrototype(),
        CoreChillerPrototype(),
        HyperConductivePrototype(),
        DualZoneSmartMeshPrototype(),
    )


def simulate_all(
    config: SimulationConfig,
    prototypes: Iterable[MattressPrototype] | None = None,
) -> tuple[SimulationResult, ...]:
    selected_prototypes = tuple(prototypes) if prototypes is not None else build_prototypes()
    if not selected_prototypes:
        raise ValueError("At least one mattress prototype is required.")
    return tuple(prototype.simulate(config) for prototype in selected_prototypes)


def create_investor_dashboard(
    results: Iterable[SimulationResult],
    config: SimulationConfig | None = None,
) -> Figure:
    """Create the two-panel investor dashboard for scripts or graphical apps."""

    result_set = tuple(results)
    if not result_set:
        raise ValueError("At least one simulation result is required for plotting.")
    simulation_config = config or SimulationConfig()
    duration_hours = result_set[0].time_seconds[-1] / 3_600.0

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (temperature_axis, power_axis) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15.5, 10.0),
        sharex=True,
        gridspec_kw={"height_ratios": (1.35, 1.0), "hspace": 0.10},
    )
    figure.patch.set_facecolor("#F7F9FC")

    for axis in (temperature_axis, power_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(True, color="#D9E1EA", linewidth=0.8, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#778899")
        axis.spines["bottom"].set_color("#778899")
        axis.tick_params(colors="#263238", labelsize=10)

    temperature_axis.axhspan(
        28.0,
        32.0,
        color="#8FD694",
        alpha=0.24,
        zorder=0,
        label="Optimal Sleep Comfort Zone (28°C–32°C)",
    )
    temperature_axis.axhline(28.0, color="#4C956C", linewidth=0.8, alpha=0.7)
    temperature_axis.axhline(32.0, color="#4C956C", linewidth=0.8, alpha=0.7)

    for result in result_set:
        minutes = result.time_seconds / 60.0
        temperature_axis.plot(
            minutes,
            result.temperature_c,
            label=result.legend_label,
            color=result.color,
            linestyle=result.line_style,
            linewidth=2.4,
            zorder=3,
        )
        power_axis.plot(
            minutes,
            result.power_w,
            label=result.legend_label,
            color=result.color,
            linestyle=result.line_style,
            linewidth=2.1,
            markevery=1_800 if result.total_energy_wh == 0.0 else None,
            marker="o" if result.total_energy_wh == 0.0 else None,
            markersize=3.0,
        )

    temperature_axis.set_ylabel("Interface Temperature (°C)", fontsize=11, weight="bold")
    temperature_axis.set_ylim(24.0, 37.0)
    temperature_axis.yaxis.set_major_locator(MultipleLocator(2.0))
    temperature_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D9E1EA",
        framealpha=0.96,
        fontsize=9,
    )

    maximum_power_w = max(float(np.max(result.power_w)) for result in result_set)
    power_axis.set_ylabel("Real-Time Power Demand (W)", fontsize=11, weight="bold")
    power_axis.set_xlabel("Sleep Duration (Minutes)", fontsize=11, weight="bold")
    power_axis.set_xlim(0.0, result_set[0].time_seconds[-1] / 60.0)
    power_axis.set_ylim(-1.5, max(64.0, maximum_power_w * 1.15))
    power_axis.xaxis.set_major_locator(MultipleLocator(30.0))
    power_axis.yaxis.set_major_locator(MultipleLocator(10.0))
    power_axis.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D9E1EA",
        framealpha=0.96,
        fontsize=9,
    )

    figure.suptitle(
        f"{duration_hours:g}-Hour Mattress Thermal Performance & Energy Footprint",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=18,
        weight="bold",
        color="#102A43",
    )
    figure.text(
        0.07,
        0.952,
        (
            f"{simulation_config.skin_temperature_c:g}°C skin load  •  "
            f"{simulation_config.ambient_temperature_c:g}°C room  •  "
            f"{simulation_config.mattress_zone_mass_kg:g} kg occupied zone  •  "
            f"{simulation_config.time_step_seconds:g}-second resolution"
        ),
        ha="left",
        fontsize=10.5,
        color="#52606D",
    )
    figure.subplots_adjust(top=0.88, left=0.08, right=0.98, bottom=0.08)
    return figure


def render_investor_dashboard(
    results: Iterable[SimulationResult],
    output_path: str | Path | None,
    show: bool = False,
    config: SimulationConfig | None = None,
) -> None:
    """Create, optionally save, and close the investor dashboard."""

    figure = create_investor_dashboard(results, config=config)
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight", facecolor=figure.get_facecolor())
    if show:
        plt.show()
    plt.close(figure)


def format_terminal_report(results: Iterable[SimulationResult]) -> str:
    """Build a dependency-free ASCII summary table."""

    result_set = tuple(results)
    if not result_set:
        raise ValueError("At least one simulation result is required for reporting.")
    duration_hours = result_set[0].time_seconds[-1] / 3_600.0
    headers = (
        "Architecture Name",
        "Final Stabilised Temp (°C)",
        f"Energy over {duration_hours:g} h (Wh)",
    )
    rows = [
        (
            result.architecture_name,
            f"{result.stabilized_temperature_c():.2f}",
            f"{result.total_energy_wh:.2f}",
        )
        for result in result_set
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(values: tuple[str, str, str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) if index == 0 else value.rjust(widths[index])
            for index, value in enumerate(values)
        ) + " |"

    table_lines = [
        "",
        f"MATTRESS PROTOTYPE — {duration_hours:g}-HOUR PERFORMANCE SUMMARY",
        border,
        format_row(headers),
        border,
        *(format_row(row) for row in rows),
        border,
    ]
    return "\n".join(table_lines)


def results_to_csv_text(results: Iterable[SimulationResult]) -> str:
    """Serialize aligned simulation time series for files or UI downloads."""

    result_set = tuple(results)
    if not result_set:
        raise ValueError("At least one simulation result is required for CSV export.")

    columns: list[np.ndarray] = [
        result_set[0].time_seconds,
        result_set[0].time_seconds / 60.0,
    ]
    headers = ["time_seconds", "sleep_duration_minutes"]
    for index, result in enumerate(result_set, start=1):
        prefix = f"p{index}"
        columns.extend(
            (result.temperature_c, result.power_w, result.cumulative_energy_wh)
        )
        headers.extend(
            (
                f"{prefix}_temperature_c",
                f"{prefix}_power_w",
                f"{prefix}_cumulative_energy_wh",
            )
        )

    csv_buffer = io.StringIO()
    np.savetxt(
        csv_buffer,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.6f",
    )
    return csv_buffer.getvalue()


def export_results_csv(results: Iterable[SimulationResult], csv_path: str | Path) -> None:
    """Export aligned temperature, power, and cumulative-energy time series."""

    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(results_to_csv_text(results), encoding="utf-8")


def run_mattress_simulation(
    output_path: str | Path | None = Path("outputs/mattress_investor_dashboard.png"),
    csv_path: str | Path | None = None,
    show: bool = False,
    print_report: bool = True,
    config: SimulationConfig | None = None,
) -> tuple[SimulationResult, ...]:
    """Run all prototypes, render outputs, and return the numerical results."""

    simulation_config = config or SimulationConfig()
    results = simulate_all(simulation_config)
    render_investor_dashboard(
        results,
        output_path=output_path,
        show=show,
        config=simulation_config,
    )
    if csv_path is not None:
        export_results_csv(results, csv_path)
    if print_report:
        print(format_terminal_report(results))
    return results


if __name__ == "__main__":
    run_mattress_simulation()
