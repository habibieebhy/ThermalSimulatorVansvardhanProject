from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mattress_thermal.simulation import (
    SimulationConfig,
    build_prototypes,
    simulate_all,
)


class MattressSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = simulate_all(SimulationConfig())

    def test_required_six_hour_one_second_shape(self) -> None:
        self.assertEqual(len(self.results), 5)
        for result in self.results:
            self.assertAlmostEqual(result.time_seconds[0], 0.0)
            self.assertAlmostEqual(result.time_seconds[-1], 21_600.0)
            self.assertEqual(len(result.time_seconds), 21_601)
            self.assertTrue(np.allclose(np.diff(result.time_seconds), 1.0))
            self.assertTrue(np.all(np.isfinite(result.temperature_c)))
            self.assertTrue(np.all(np.isfinite(result.power_w)))

    def test_energy_accounting_matches_defined_power_profiles(self) -> None:
        energies = [result.total_energy_wh for result in self.results]
        self.assertAlmostEqual(energies[0], 0.0, places=9)
        self.assertAlmostEqual(energies[1], 30.0, places=9)
        self.assertGreater(energies[2], 0.0)
        self.assertLess(energies[2], 360.0)
        self.assertAlmostEqual(energies[3], 0.0, places=9)
        self.assertAlmostEqual(energies[4], 65.0, places=9)

    def test_architecture_thermal_behaviour(self) -> None:
        aero_natural, eco_battery, core_chiller, graphite, dual_zone = self.results
        minute_80_index = 80 * 60

        self.assertGreaterEqual(aero_natural.temperature_c[minute_80_index], 28.0)
        self.assertLessEqual(aero_natural.temperature_c[minute_80_index], 32.0)
        self.assertGreater(aero_natural.stabilized_temperature_c(), 34.0)

        for managed_prototype in (eco_battery, core_chiller, graphite, dual_zone):
            self.assertGreaterEqual(managed_prototype.stabilized_temperature_c(), 28.0)
            self.assertLessEqual(managed_prototype.stabilized_temperature_c(), 32.0)

    def test_power_profiles_and_limits(self) -> None:
        _, eco_battery, core_chiller, _, dual_zone = self.results
        self.assertTrue(np.allclose(eco_battery.power_w, 5.0))
        self.assertLessEqual(float(np.max(core_chiller.power_w)), 60.0)
        self.assertAlmostEqual(dual_zone.power_w[3_599], 40.0)
        self.assertAlmostEqual(dual_zone.power_w[3_600], 10.0)
        self.assertAlmostEqual(dual_zone.power_w[3_630], 0.0)

    def test_invalid_configurations_fail_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_step_seconds"):
            SimulationConfig(time_step_seconds=0.0)
        with self.assertRaisesRegex(ValueError, "evenly divisible"):
            SimulationConfig(duration_seconds=10, time_step_seconds=3.0)
        with self.assertRaisesRegex(ValueError, "skin_temperature_c"):
            SimulationConfig(skin_temperature_c=24.0)

    def test_all_required_prototype_names_are_present(self) -> None:
        names = [prototype.architecture_name for prototype in build_prototypes()]
        self.assertEqual(
            names,
            [
                "P1: Aero-Natural (Passive)",
                "P2: Eco-Battery (5W Low Power)",
                "P3: Core-Chiller (60W AC Active)",
                "P4: Hyper-Conductive (Zero Power Graphite)",
                "P5: Dual-Zone Smart Mesh (Hybrid)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
