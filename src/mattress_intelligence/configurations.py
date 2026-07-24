"""Constraint-based generation of physically feasible mattress stacks."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Iterable

from .materials import MaterialLibrary
from .models import CandidateLayer, ConfigurationCandidate, LayerRecord, ProductRecord, stable_id


@dataclass(frozen=True, slots=True)
class GenerationReport:
    candidates: list[ConfigurationCandidate]
    enumerated_combinations: int
    rejected_thickness: int
    rejected_weight: int
    backend: str
    warnings: list[str]


def _selected_variant_area_m2(product: ProductRecord) -> float | None:
    for variant in product.variants:
        if variant.width_mm and variant.length_mm:
            return variant.width_mm * variant.length_mm / 1_000_000.0
    return None


def _candidate_weight_kg(layers: list[CandidateLayer], area_m2: float | None) -> float | None:
    if area_m2 is None:
        return None
    return sum(
        area_m2 * (layer.thickness_mm / 1_000.0) * layer.density_kg_m3
        for layer in layers
    )


class ConfigurationGenerator:
    """Enumerate discrete candidates and apply hard thickness/mass constraints."""

    def __init__(self, materials: MaterialLibrary) -> None:
        self.materials = materials

    def _layer_blueprint(self, product: ProductRecord) -> tuple[list[LayerRecord], bool]:
        if product.layers:
            return sorted(product.layers, key=lambda layer: layer.position), False
        return (
            [
                LayerRecord(
                    position=1,
                    marketing_name="Quilted comfort layer",
                    normalized_material="quilt_foam",
                ),
                LayerRecord(
                    position=2,
                    marketing_name="Comfort foam",
                    normalized_material="memory_foam",
                ),
                LayerRecord(
                    position=3,
                    marketing_name="Support core",
                    normalized_material="hr_foam",
                ),
            ],
            True,
        )

    def _thickness_options(
        self, layer: LayerRecord, total_mm: int, layer_count: int
    ) -> tuple[int, ...]:
        if layer.thickness_mm is not None:
            return (max(1, int(round(layer.thickness_mm))),)
        upper = max(10, total_mm - 10 * (layer_count - 1))
        values = tuple(value for value in self.materials.thickness_grid_mm if value <= upper)
        return values or (max(10, total_mm // layer_count),)

    @staticmethod
    def _python_thickness_solutions(
        options: list[tuple[int, ...]], total_mm: int, limit: int
    ) -> list[tuple[int, ...]]:
        solutions: list[tuple[int, ...]] = []
        for values in itertools.product(*options):
            if sum(values) == total_mm:
                solutions.append(tuple(int(value) for value in values))
                if len(solutions) >= limit:
                    break
        return solutions

    @staticmethod
    def _ortools_thickness_solutions(
        options: list[tuple[int, ...]], total_mm: int, limit: int
    ) -> list[tuple[int, ...]] | None:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            return None

        model = cp_model.CpModel()
        variables = [
            model.new_int_var_from_domain(cp_model.Domain.from_values(list(values)), f"t_{index}")
            for index, values in enumerate(options)
        ]
        model.add(sum(variables) == total_mm)

        class Collector(cp_model.CpSolverSolutionCallback):
            def __init__(self) -> None:
                super().__init__()
                self.solutions: list[tuple[int, ...]] = []

            def on_solution_callback(self) -> None:
                self.solutions.append(tuple(int(self.value(variable)) for variable in variables))
                if len(self.solutions) >= limit:
                    self.stop_search()

        collector = Collector()
        solver = cp_model.CpSolver()
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.max_time_in_seconds = 5.0
        solver.solve(model, collector)
        return collector.solutions

    def generate(
        self,
        product: ProductRecord,
        max_candidates: int = 50,
        max_enumerations: int = 50_000,
    ) -> GenerationReport:
        warnings: list[str] = []
        blueprint, synthesized_pattern = self._layer_blueprint(product)
        total_mm = int(round(product.total_thickness_mm or 200.0))
        if product.total_thickness_mm is None:
            warnings.append("Total thickness was undisclosed; 200 mm research baseline used.")
        if synthesized_pattern:
            warnings.append("Layer order was undisclosed; generic three-layer foam pattern used.")

        thickness_options = [
            self._thickness_options(layer, total_mm, len(blueprint)) for layer in blueprint
        ]
        solution_limit = max(500, max_candidates * 20)
        thickness_solutions = self._ortools_thickness_solutions(
            thickness_options, total_mm, solution_limit
        )
        backend = "ortools-cp-sat" if thickness_solutions is not None else "python-enumerator"
        if thickness_solutions is None:
            thickness_solutions = self._python_thickness_solutions(
                thickness_options, total_mm, solution_limit
            )

        if not thickness_solutions:
            warnings.append(
                "No exact grid solution; positive proportional layer thicknesses were used."
            )

            layer_count = len(blueprint)
            if total_mm < layer_count:
                warnings.append(
                    f"Total thickness {total_mm} mm cannot provide at least 1 mm "
                    f"for each of {layer_count} layers; 200 mm research baseline used."
                )
                total_mm = max(200, layer_count)

            # Reserve 1 mm for every layer first. Then distribute the remaining
            # thickness proportionally. Known thicknesses are used as weights;
            # undisclosed or invalid values receive a neutral 1.0 weight.
            weights = [
                max(float(layer.thickness_mm or 1.0), 1.0)
                for layer in blueprint
            ]
            weight_total = sum(weights)
            remaining_mm = total_mm - layer_count
            raw_extras = [
                remaining_mm * weight / weight_total
                for weight in weights
            ]
            adjusted = [
                1 + math.floor(extra)
                for extra in raw_extras
            ]

            # Largest-remainder allocation guarantees integer values that are
            # all positive and sum exactly to the required mattress thickness.
            remainder = total_mm - sum(adjusted)
            allocation_order = sorted(
                range(layer_count),
                key=lambda index: raw_extras[index] - math.floor(raw_extras[index]),
                reverse=True,
            )
            for offset in range(remainder):
                adjusted[allocation_order[offset % layer_count]] += 1

            if any(value <= 0 for value in adjusted) or sum(adjusted) != total_mm:
                raise RuntimeError(
                    "Unable to allocate positive layer thicknesses that match "
                    f"the required total of {total_mm} mm."
                )

            thickness_solutions = [tuple(adjusted)]

        density_options: list[tuple[int, ...]] = []
        for layer in blueprint:
            if layer.density_kg_m3 is not None:
                density_options.append((int(round(layer.density_kg_m3)),))
            else:
                density_options.append(self.materials.get(layer.normalized_material).densities_kg_m3)

        area_m2 = _selected_variant_area_m2(product)
        observed_weight = product.product_weight_kg
        if observed_weight is None:
            observed_weight = next(
                (variant.weight_kg for variant in product.variants if variant.weight_kg), None
            )
        if observed_weight is not None and area_m2 is None:
            warnings.append("Weight was known but no width/length variant was available; mass constraint skipped.")

        candidates: list[ConfigurationCandidate] = []
        enumerated = 0
        rejected_weight = 0
        seen: set[str] = set()
        for thicknesses in thickness_solutions:
            for densities in itertools.product(*density_options):
                enumerated += 1
                if enumerated > max_enumerations:
                    warnings.append(f"Enumeration stopped at safety limit {max_enumerations}.")
                    break
                candidate_layers: list[CandidateLayer] = []
                for layer, thickness, density in zip(blueprint, thicknesses, densities, strict=True):
                    material_id = self.materials.normalize(layer.normalized_material)
                    spec = self.materials.get(material_id)
                    candidate_layers.append(
                        CandidateLayer(
                            position=layer.position,
                            material=material_id,
                            marketing_name=layer.marketing_name,
                            thickness_mm=int(thickness),
                            density_kg_m3=int(density),
                            conductivity_w_mk=spec.conductivity_w_mk,
                            specific_heat_j_kgk=spec.specific_heat_j_kgk,
                        )
                    )
                estimated_weight = _candidate_weight_kg(candidate_layers, area_m2)
                if observed_weight is not None and estimated_weight is not None:
                    tolerance = max(2.0, observed_weight * 0.25)
                    if abs(estimated_weight - observed_weight) > tolerance:
                        rejected_weight += 1
                        continue
                signature = ";".join(
                    f"{layer.material}:{layer.thickness_mm}:{layer.density_kg_m3}"
                    for layer in candidate_layers
                )
                if signature in seen:
                    continue
                seen.add(signature)
                candidates.append(
                    ConfigurationCandidate(
                        configuration_id=stable_id("cfg", product.product_id, signature),
                        product_id=str(product.product_id),
                        layers=candidate_layers,
                        total_thickness_mm=total_mm,
                        estimated_weight_kg=(
                            round(estimated_weight, 3) if estimated_weight is not None else None
                        ),
                    )
                )
            if enumerated > max_enumerations:
                break

        if not candidates:
            warnings.append("All enumerated candidates failed the mass constraint.")
        return GenerationReport(
            candidates=candidates[: max(max_candidates * 20, max_candidates)],
            enumerated_combinations=enumerated,
            rejected_thickness=0,
            rejected_weight=rejected_weight,
            backend=backend,
            warnings=warnings,
        )
