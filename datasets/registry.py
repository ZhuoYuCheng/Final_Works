import os
from typing import Dict, List

from data.adapters.aegis import load_aegis_dir
from data.adapters.who_when import load_who_when_dir
from data.schema import UnifiedTrajectory


def load_dataset(data_dir: str, subset: str, base_mode: str = "auto") -> List[UnifiedTrajectory]:
    if subset in ("Hand-Crafted", "Algorithm-Generated"):
        return load_who_when_dir(data_dir, subset, base_mode=base_mode)
    if subset.lower() == "aegis":
        return load_aegis_dir(os.path.join(data_dir, "aegis"))
    raise ValueError(f"Unknown subset: {subset}")


def parse_subset_list(value: str, default: List[str]) -> List[str]:
    if not value:
        return default
    parts = [v.strip() for v in value.split(",") if v.strip()]
    return parts or default
