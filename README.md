# Mattress Thermal Prototype Simulator

A clean, object-oriented NumPy/Matplotlib model with a Streamlit graphical
control panel for comparing five mattress thermal architectures. It produces a
two-panel investor dashboard, decision table, console report, and CSV export.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m mattress_thermal --csv outputs/mattress_simulation.csv
```

## Graphical interface

Launch the browser-based thermal lab with:

```bash
streamlit run app.py
```

Use the left control panel to change room/body conditions and architecture
parameters, press **Run simulation**, then download the regenerated dashboard or
full time-series CSV. The defaults reproduce the validated six-hour baseline.
With **Animate graph formation** enabled, Run simulation progressively reveals
the real calculated curves and displays a simulation clock. This is playback of
the completed one-second numerical solution; disable it for faster parameter
sweeps.

For the shortest route after installing only `requirements.txt`, run:

```bash
python run_simulation.py
```

The dashboard is saved to `outputs/mattress_investor_dashboard.png`. Add
`--show` to open it interactively. Use `--output PATH` to change the image
destination.

## Physics model

The occupied mattress zone is represented as a lumped thermal mass:

```text
C = mass × specific heat = 0.6 kg × 1600 J/(kg·K) = 960 J/K
q = conductivity × area × temperature difference / path length
ΔT = net heat flow × Δt / C
```

At each one-second time step, the model independently calculates heat entering
from 37°C skin, heat rejected toward the 25°C room, architecture-specific heat
removal, electrical power, and accumulated watt-hours. Active thermal paths are
limited by both conductive capacity and device capacity.

| Prototype | Modelled mechanism | Electrical profile |
| --- | --- | ---: |
| P1 Aero-Natural | Open-cell latex plus a finite 73.5 kJ PCM reservoir that saturates around 90 minutes | 0 W |
| P2 Eco-Battery | Ambient-water microtubes and passive aluminum radiator | Constant 5 W |
| P3 Core-Chiller | Peltier/water-block proportional controller targeting 29.5°C | 0–60 W controlled |
| P4 Hyper-Conductive | Flexible graphite spreading heat to exposed edges | 0 W |
| P5 Dual-Zone | One-hour turbo followed by 30 s on / 30 s off eco pulses | 40 W, then pulsed 10 W |

“Final stabilised temperature” is the mean interface temperature during the
last 15 minutes, not a potentially misleading single sample.

## What to tune—and what it changes

| Control | Primary effect | Important distinction |
| --- | --- | --- |
| Room/skin temperature | Changes the driving temperature difference | A hotter room makes every room-coupled passive system weaker |
| Foam mass or specific heat | Changes warm-up speed | Does not materially change true steady-state temperature |
| PCM capacity | Extends P1 cooling duration | Does not improve P1 after saturation |
| PCM absorption power | Limits how fast PCM can accept body heat | Capacity is “how long”; absorption power is “how hard” |
| Conductivity or area | Increases conductive heat rejection | Test assembled-system effective values, not ideal material datasheets |
| Heat-flow distance | Reduces rejection when increased | It appears in the denominator of `q = kAΔT/L` |
| Peltier COP | Changes cooling obtained per electrical watt | Poor heat-sink performance lowers real COP sharply |
| Controller target/gain | Sets P3 operating point and response strength | Extreme gain can imply control behaviour the hardware cannot deliver |
| Turbo power/duration | Controls P5 first-hour pull-down and energy | Longer turbo directly increases Wh |
| Pulse duty cycle | Controls average eco power | Average eco demand is pulse power × duty cycle |
| Cooling coupling | Represents system-level delivery losses | This must be calibrated from a physical prototype |

Change one family of parameters at a time, keep a named baseline, and compare
temperature **and** Wh together. A curve inside 28–32°C is not automatically a
winning design if the assumed material path, COP, or radiator area cannot be
manufactured.

## Programmatic use

```python
from mattress_thermal import run_mattress_simulation

results = run_mattress_simulation(
    output_path="outputs/pitch_dashboard.png",
    csv_path="outputs/pitch_data.csv",
    show=False,
)
```

The returned tuple contains the temperature, instantaneous power, and
cumulative energy arrays for all five prototypes.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Engineering scope

This is a transparent comparative prototype model, not a certification-grade
finite-element or CFD model. The effective path dimensions, PCM capacity,
Peltier COP, controller gain, and cooling coupling are explicit calibration
parameters in `src/mattress_thermal/simulation.py`. Replace them with measured
coupon, thermal-manikin, or guarded-hot-plate data before making product claims.