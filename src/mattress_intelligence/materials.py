"""Material taxonomy and engineering property catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
import re
from typing import Iterable

import yaml


@dataclass(frozen=True, slots=True)
class MaterialSpec:
    material_id: str
    aliases: tuple[str, ...]
    densities_kg_m3: tuple[int, ...]
    density_prior: tuple[float, ...]
    conductivity_w_mk: float
    specific_heat_j_kgk: float
    firmness_index: float

    def __post_init__(self) -> None:
        if len(self.densities_kg_m3) != len(self.density_prior):
            raise ValueError(f"Density prior length mismatch for {self.material_id}.")
        if not 0.99 <= sum(self.density_prior) <= 1.01:
            raise ValueError(f"Density priors must sum to one for {self.material_id}.")


class MaterialLibrary:
    """Load and normalize a deliberately small, editable material taxonomy."""

    def __init__(
        self,
        materials: dict[str, MaterialSpec],
        fallback_material: str,
        thickness_grid_mm: tuple[int, ...],
    ) -> None:
        if fallback_material not in materials:
            raise ValueError("fallback_material must be present in materials.")
        self.materials = materials
        self.fallback_material = fallback_material
        self.thickness_grid_mm = thickness_grid_mm

        aliases: list[tuple[str, str]] = []
        for material_id, spec in materials.items():
            aliases.append((material_id.replace("_", " "), material_id))
            aliases.extend((alias.casefold(), material_id) for alias in spec.aliases)
        self._aliases = sorted(aliases, key=lambda pair: len(pair[0]), reverse=True)

    @classmethod
    def load(cls, path: Path | None = None) -> "MaterialLibrary":
        resource = path or Path(str(files("mattress_intelligence").joinpath("data/materials.yaml")))
        with resource.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)

        materials = {
            material_id: MaterialSpec(
                material_id=material_id,
                aliases=tuple(values.get("aliases", [])),
                densities_kg_m3=tuple(int(value) for value in values["densities_kg_m3"]),
                density_prior=tuple(float(value) for value in values["density_prior"]),
                conductivity_w_mk=float(values["conductivity_w_mk"]),
                specific_heat_j_kgk=float(values["specific_heat_j_kgk"]),
                firmness_index=float(values["firmness_index"]),
            )
            for material_id, values in raw["materials"].items()
        }
        return cls(
            materials=materials,
            fallback_material=raw["fallback_material"],
            thickness_grid_mm=tuple(int(value) for value in raw["thickness_grid_mm"]),
        )

    def normalize(self, text: str | None) -> str:
        if not text:
            return self.fallback_material
        normalized = " ".join(text.casefold().replace("-", " ").split())
        for alias, material_id in self._aliases:
            if alias in normalized:
                return material_id
        return self.fallback_material

    def find_material_mentions(self, text: str) -> list[tuple[str, str]]:
        normalized = " ".join(text.casefold().replace("-", " ").split())
        matches: list[tuple[str, str]] = []
        seen: set[str] = set()
        for alias, material_id in self._aliases:
            if alias in normalized and material_id not in seen:
                matches.append((alias, material_id))
                seen.add(material_id)
        return matches

    def iter_material_mentions(
        self,
        text: str,
        *,
        max_matches: int = 500,
    ) -> list[tuple[str, str, int, int]]:
        """Return ordered material-alias occurrences with offsets in the original text."""

        lowered = text.casefold()
        matches: list[tuple[str, str, int, int]] = []
        seen_spans: set[tuple[int, int, str]] = set()
        for alias, material_id in self._aliases:
            tokens = [re.escape(token) for token in alias.split()]
            if not tokens:
                continue
            pattern = r"(?<![a-z0-9])" + r"[\s_-]+".join(tokens) + r"(?![a-z0-9])"
            for match in re.finditer(pattern, lowered):
                key = (match.start(), match.end(), material_id)
                if key in seen_spans:
                    continue
                seen_spans.add(key)
                matches.append((text[match.start():match.end()], material_id, match.start(), match.end()))
                if len(matches) >= max_matches:
                    return sorted(matches, key=lambda item: (item[2], -(item[3] - item[2])))
        return sorted(matches, key=lambda item: (item[2], -(item[3] - item[2])))

    def get(self, material_id: str) -> MaterialSpec:
        return self.materials.get(material_id, self.materials[self.fallback_material])

    def known_ids(self) -> Iterable[str]:
        return self.materials.keys()

