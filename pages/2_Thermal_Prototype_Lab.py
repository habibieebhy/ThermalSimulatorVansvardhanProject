"""Streamlit control panel for the mattress thermal prototype simulator."""

from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mattress_thermal.simulation import (  # noqa: E402
    AeroNaturalPrototype,
    CoreChillerPrototype,
    DualZoneSmartMeshPrototype,
    EcoBatteryPrototype,
    HyperConductivePrototype,
    SimulationConfig,
    create_investor_dashboard,
    results_to_csv_text,
    simulate_all,
)


st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #d9e1ea;
        border-radius: 0.75rem;
        padding: 0.85rem;
    }
    .small-note { color: #52606d; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Mattress Thermal Prototype Lab")
st.caption(
    "Interactive six-hour thermal and energy comparison • 1-second physics engine • "
    "Investor dashboard and downloadable data"
)


def reveal_dashboard_window(figure, results, end_index: int) -> None:
    """Limit plotted traces to a shared time index for animated playback."""

    for axis, value_attribute in (
        (figure.axes[0], "temperature_c"),
        (figure.axes[1], "power_w"),
    ):
        labelled_lines = {line.get_label(): line for line in axis.get_lines()}
        for result in results:
            trace = labelled_lines.get(result.legend_label)
            if trace is not None:
                trace.set_data(
                    result.time_seconds[:end_index] / 60.0,
                    getattr(result, value_attribute)[:end_index],
                )

with st.sidebar:
    st.header("Simulation controls")
    st.caption("Defaults reproduce the validated baseline model.")

    with st.form("simulation_controls"):
        with st.expander("Room, body & foam", expanded=True):
            duration_hours = st.slider(
                "Sleep duration (hours)", 1.0, 10.0, 6.0, 0.5,
                help="Changes the observation window. The required baseline is six hours.",
            )
            ambient_temperature_c = st.slider(
                "Room temperature (°C)", 18.0, 32.0, 25.0, 0.5,
                help="The passive heat sink for every architecture.",
            )
            skin_temperature_c = st.slider(
                "Skin temperature (°C)", 32.0, 39.0, 37.0, 0.5,
                help="The body-side thermal boundary condition.",
            )
            mattress_mass_kg = st.slider(
                "Occupied foam mass (kg)", 0.2, 2.0, 0.6, 0.05,
                help="More mass changes temperature more slowly under the same heat load.",
            )
            specific_heat_j_per_kg_k = st.slider(
                "Foam specific heat (J/kg·K)", 800, 2_500, 1_600, 50,
                help="Higher values increase thermal inertia, not final steady-state cooling.",
            )
            st.text_input("Physics time step", "1.0 second", disabled=True)
            animate_graph = st.checkbox(
                "Animate graph formation",
                value=True,
                help=(
                    "Reveals the actual calculated traces progressively after each run. "
                    "Disable it when rapidly comparing many parameter combinations."
                ),
            )
            animation_duration_seconds = st.slider(
                "Animation duration (seconds)", 2, 12, 6, 1,
                disabled=not animate_graph,
            )

        with st.expander("P1 • Aero-Natural PCM"):
            p1_pcm_phase_c = st.slider("PCM phase temperature (°C)", 26.0, 32.0, 28.5, 0.1)
            p1_pcm_capacity_kj = st.slider(
                "PCM usable capacity (kJ)", 0.0, 160.0, 73.5, 2.5,
                help="Increasing this primarily extends the pre-saturation comfort duration.",
            )
            p1_pcm_absorption_w = st.slider(
                "PCM maximum absorption (W)", 1.0, 30.0, 15.0, 0.5,
                help="This controls whether the PCM can keep up with instantaneous body heat.",
            )
            p1_latex_k = st.slider(
                "Latex effective conductivity (W/m·K)", 0.020, 0.120, 0.035, 0.005,
            )

        with st.expander("P2 • Eco-Battery radiator"):
            p2_pump_power_w = st.slider("Pump electrical power (W)", 1.0, 15.0, 5.0, 0.5)
            p2_loop_k = st.slider(
                "Water-loop effective conductivity (W/m·K)", 0.10, 1.50, 0.70, 0.05,
            )
            p2_radiator_area_m2 = st.slider(
                "Effective radiator area (m²)", 0.03, 0.30, 0.12, 0.01,
            )
            p2_flow_distance_m = st.slider(
                "Heat-flow distance (m)", 0.010, 0.100, 0.030, 0.005,
                help="Shorter paths reject more heat because q is inversely proportional to distance.",
            )

        with st.expander("P3 • Core-Chiller controller"):
            p3_target_c = st.slider("Control target (°C)", 28.0, 32.0, 29.5, 0.1)
            p3_max_power_w = st.slider("Rated maximum power (W)", 20.0, 120.0, 60.0, 5.0)
            p3_cop = st.slider(
                "Peltier effective COP", 0.20, 1.20, 0.65, 0.05,
                help="Thermal watts removed per electrical watt in this simplified model.",
            )
            p3_gain = st.slider(
                "Controller gain (W/K)", 1.0, 20.0, 8.0, 0.5,
                help="Higher gain corrects temperature error harder but can become unrealistic.",
            )
            p3_cold_side_c = st.slider("Coolant/cold-side temperature (°C)", 10.0, 24.0, 18.0, 0.5)

        with st.expander("P4 • Graphite spreader"):
            p4_graphite_k = st.slider(
                "Effective graphite conductivity (W/m·K)", 5.0, 100.0, 15.0, 2.5,
                help="Use measured assembled-system conductivity, not a supplier's ideal sheet value.",
            )
            p4_area_m2 = st.slider("Effective spreader area (m²)", 0.004, 0.040, 0.012, 0.002)
            p4_edge_distance_m = st.slider(
                "Distance to rejecting edge (m)", 0.030, 0.250, 0.080, 0.010,
            )

        with st.expander("P5 • Hybrid strategy"):
            p5_turbo_minutes = st.slider("Turbo duration (minutes)", 15.0, 120.0, 60.0, 5.0)
            p5_turbo_power_w = st.slider("Turbo power (W)", 10.0, 80.0, 40.0, 2.5)
            p5_eco_power_w = st.slider("Eco pulse power (W)", 2.0, 30.0, 10.0, 1.0)
            p5_pulse_period_s = st.slider("Pulse period (seconds)", 30, 300, 60, 10)
            p5_duty_percent = st.slider(
                "Eco pulse duty cycle (%)", 10, 100, 50, 5,
                help="50% means the eco loop is on for half of every pulse period.",
            )
            p5_cooling_coupling = st.slider(
                "Electrical-to-zone cooling coupling", 0.10, 1.00, 0.35, 0.05,
                help="A calibration factor for losses between device power and the occupied zone.",
            )

        run_clicked = st.form_submit_button(
            "Run simulation",
            type="primary",
            use_container_width=True,
        )

if run_clicked:
    st.toast("Simulation updated", icon="✅")

try:
    config = SimulationConfig(
        duration_seconds=int(round(duration_hours * 3_600)),
        time_step_seconds=1.0,
        skin_temperature_c=skin_temperature_c,
        ambient_temperature_c=ambient_temperature_c,
        mattress_zone_mass_kg=mattress_mass_kg,
        foam_specific_heat_j_per_kg_k=float(specific_heat_j_per_kg_k),
    )
    prototypes = (
        AeroNaturalPrototype(
            latex_conductivity_w_per_m_k=p1_latex_k,
            pcm_phase_temperature_c=p1_pcm_phase_c,
            pcm_capacity_kj=p1_pcm_capacity_kj,
            pcm_max_absorption_w=p1_pcm_absorption_w,
        ),
        EcoBatteryPrototype(
            pump_power_w=p2_pump_power_w,
            loop_conductivity_w_per_m_k=p2_loop_k,
            radiator_area_m2=p2_radiator_area_m2,
            flow_path_distance_m=p2_flow_distance_m,
        ),
        CoreChillerPrototype(
            target_temperature_c=p3_target_c,
            maximum_electrical_power_w=p3_max_power_w,
            peltier_cop=p3_cop,
            proportional_gain_w_per_k=p3_gain,
            cold_side_temperature_c=p3_cold_side_c,
        ),
        HyperConductivePrototype(
            graphite_conductivity_w_per_m_k=p4_graphite_k,
            spreader_area_m2=p4_area_m2,
            edge_distance_m=p4_edge_distance_m,
        ),
        DualZoneSmartMeshPrototype(
            turbo_duration_minutes=p5_turbo_minutes,
            turbo_power_w=p5_turbo_power_w,
            eco_pulse_power_w=p5_eco_power_w,
            pulse_period_seconds=float(p5_pulse_period_s),
            pulse_on_seconds=p5_pulse_period_s * p5_duty_percent / 100.0,
            cooling_coupling=p5_cooling_coupling,
        ),
    )
    with st.spinner("Solving the thermal balance for every second..."):
        results = simulate_all(config, prototypes)
except (ValueError, RuntimeError) as error:
    st.error(f"Simulation stopped safely: {error}")
    st.stop()

metric_columns = st.columns(5)
for column, result in zip(metric_columns, results):
    comfort_fraction = np.mean(
        (result.temperature_c >= 28.0) & (result.temperature_c <= 32.0)
    )
    with column:
        st.metric(
            result.architecture_name.split(":", maxsplit=1)[0],
            f"{result.stabilized_temperature_c():.2f}°C",
            f"{result.total_energy_wh:.1f} Wh • {comfort_fraction:.0%} comfort",
            delta_color="off",
        )

figure = create_investor_dashboard(results, config=config)
chart_placeholder = st.empty()
clock_placeholder = st.empty()
if run_clicked and animate_graph:
    frame_count = max(20, animation_duration_seconds * 6)
    frame_end_indices = np.unique(
        np.linspace(2, len(results[0].time_seconds), frame_count, dtype=int)
    )
    frame_pause_seconds = animation_duration_seconds / len(frame_end_indices)
    for end_index in frame_end_indices:
        reveal_dashboard_window(figure, results, int(end_index))
        elapsed_hours = results[0].time_seconds[end_index - 1] / 3_600.0
        clock_placeholder.caption(
            f"Simulation playback: {elapsed_hours:.2f} of {duration_hours:g} hours"
        )
        chart_placeholder.pyplot(figure, use_container_width=True, clear_figure=False)
        time.sleep(frame_pause_seconds)
    clock_placeholder.caption(
        f"Simulation complete: {duration_hours:g} hours • "
        f"{len(results[0].time_seconds):,} calculated time points"
    )
else:
    chart_placeholder.pyplot(figure, use_container_width=True, clear_figure=False)
    clock_placeholder.caption(
        f"Simulation complete: {duration_hours:g} hours • "
        f"{len(results[0].time_seconds):,} calculated time points"
    )

summary_rows = []
for result in results:
    in_comfort = (result.temperature_c >= 28.0) & (result.temperature_c <= 32.0)
    summary_rows.append(
        {
            "Architecture": result.architecture_name,
            "Stabilised temperature (°C)": round(result.stabilized_temperature_c(), 2),
            "Energy (Wh)": round(result.total_energy_wh, 2),
            "Time in comfort zone (%)": round(float(np.mean(in_comfort) * 100.0), 1),
            "Peak electrical power (W)": round(float(np.max(result.power_w)), 2),
        }
    )

st.subheader("Decision table")
st.dataframe(summary_rows, use_container_width=True, hide_index=True)

png_buffer = BytesIO()
figure.savefig(png_buffer, format="png", dpi=180, bbox_inches="tight")
csv_text = results_to_csv_text(results)
download_column_1, download_column_2, note_column = st.columns((1, 1, 2))
with download_column_1:
    st.download_button(
        "Download dashboard PNG",
        data=png_buffer.getvalue(),
        file_name="mattress_investor_dashboard.png",
        mime="image/png",
        use_container_width=True,
    )
with download_column_2:
    st.download_button(
        "Download simulation CSV",
        data=csv_text.encode("utf-8"),
        file_name="mattress_simulation.csv",
        mime="text/csv",
        use_container_width=True,
    )
with note_column:
    st.markdown(
        '<p class="small-note">“Stabilised” is the final 15-minute mean. '
        "Comfort percentage includes the initial warm-up from room temperature.</p>",
        unsafe_allow_html=True,
    )
plt.close(figure)

with st.expander("How the physics engine works"):
    st.markdown(
        r"""
        The occupied zone is one thermal mass with heat capacity
        $C = m c_p$. During every one-second interval, each path calculates
        $q = kA(T_{hot}-T_{cold})/L$. The solver then applies:

        $$T_{next}=T_{now}+\frac{(q_{body}-q_{passive}-q_{device})\Delta t}{m c_p}$$

        PCM removes heat only while latent capacity remains. The water loop and
        graphite paths reject heat toward the 25°C room. The Peltier system is
        capacity-limited and controlled toward its target. The hybrid follows
        the selected turbo and pulse schedule. Electrical power is integrated
        as $Wh=\sum P\Delta t/3600$.
        """
    )

with st.expander("Control dictionary — what every slider changes"):
    st.dataframe(
        [
            {"Control": "Sleep duration", "Physical meaning": "How long the test is observed", "When increased": "Reveals later saturation; does not alter heat flow"},
            {"Control": "Room temperature", "Physical meaning": "Temperature of the heat-rejection destination", "When increased": "Every room-cooled design rejects less heat"},
            {"Control": "Skin temperature", "Physical meaning": "Hot-side body boundary", "When increased": "More body heat enters the mattress"},
            {"Control": "Foam mass", "Physical meaning": "Material included in the occupied thermal zone", "When increased": "Temperature changes more slowly"},
            {"Control": "Foam specific heat", "Physical meaning": "Energy required to warm 1 kg by 1°C", "When increased": "Temperature changes more slowly"},
            {"Control": "Physics time step", "Physical meaning": "Numerical update interval Δt", "When increased": "Runs coarser and may lose accuracy; locked at 1 s"},
            {"Control": "P1 PCM phase temperature", "Physical meaning": "Temperature where PCM strongly absorbs latent heat", "When increased": "Moves the PCM holding plateau upward"},
            {"Control": "P1 PCM usable capacity", "Physical meaning": "Total heat the PCM can store", "When increased": "Extends time before saturation"},
            {"Control": "P1 PCM absorption", "Physical meaning": "Maximum instantaneous heat acceptance", "When increased": "PCM can keep up with a larger body load"},
            {"Control": "P1 latex conductivity", "Physical meaning": "Passive latex heat-transfer ability", "When increased": "More heat escapes toward the room"},
            {"Control": "P2 pump power", "Physical meaning": "Electrical demand of circulation pump", "When increased": "Raises Wh; thermal benefit changes only if measured loop effectiveness also improves"},
            {"Control": "P2 loop conductivity", "Physical meaning": "Effective heat-transfer ability of water loop plus interfaces", "When increased": "More heat reaches the radiator"},
            {"Control": "P2 radiator area", "Physical meaning": "Effective area participating in rejection", "When increased": "More passive heat can leave"},
            {"Control": "P2 heat-flow distance", "Physical meaning": "Distance heat travels through the modelled path", "When increased": "Less heat leaves because distance is in the denominator"},
            {"Control": "P3 control target", "Physical meaning": "Requested interface temperature", "When increased": "Chiller permits a warmer bed and usually consumes less"},
            {"Control": "P3 rated maximum power", "Physical meaning": "Electrical ceiling available to the Peltier", "When increased": "Allows stronger correction; does not force constant maximum draw"},
            {"Control": "P3 effective COP", "Physical meaning": "Thermal watts removed per electrical watt", "When increased": "Same cooling requires less electricity"},
            {"Control": "P3 controller gain", "Physical meaning": "How aggressively power reacts to temperature error", "When increased": "Returns to target harder and faster"},
            {"Control": "P3 cold-side temperature", "Physical meaning": "Temperature of coolant accepting Peltier heat", "When increased": "Reduces available conductive cooling head"},
            {"Control": "P4 graphite conductivity", "Physical meaning": "Assembled spreader's effective ability to move heat", "When increased": "Heat reaches the edges faster"},
            {"Control": "P4 spreader area", "Physical meaning": "Graphite area effectively carrying heat", "When increased": "More heat can be dispersed"},
            {"Control": "P4 edge distance", "Physical meaning": "Travel length from body zone to rejecting edge", "When increased": "Graphite path becomes less effective"},
            {"Control": "P5 turbo duration", "Physical meaning": "Time spent in first-stage high power", "When increased": "Extends aggressive cooling and directly raises Wh"},
            {"Control": "P5 turbo power", "Physical meaning": "First-stage electrical demand", "When increased": "Adds cooling capability and energy use"},
            {"Control": "P5 eco pulse power", "Physical meaning": "Power while maintenance loop is on", "When increased": "Stronger maintenance pulses and more Wh"},
            {"Control": "P5 pulse period", "Physical meaning": "Length of one on/off cycle", "When increased": "Makes individual on and off blocks longer at the same duty cycle"},
            {"Control": "P5 duty cycle", "Physical meaning": "Fraction of eco time switched on", "When increased": "Raises average power and cooling"},
            {"Control": "P5 cooling coupling", "Physical meaning": "Fraction linking device power to useful occupied-zone removal", "When increased": "More device effort reaches the user zone"},
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Control": st.column_config.TextColumn(width="medium"),
            "Physical meaning": st.column_config.TextColumn(width="large"),
            "When increased": st.column_config.TextColumn(width="large"),
        },
    )

st.warning(
    "Use this for architecture comparison and experiment planning. Calibrate the "
    "effective conductivities, areas, COP, PCM capacity, and coupling against measured "
    "prototype data before using the curves as product-performance claims.",
    icon="⚠️",
)
