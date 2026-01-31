from typing import Dict, List

from data.schema import UnifiedTrajectory


def validate_trajectory(traj: UnifiedTrajectory) -> Dict[str, List[str]]:
    warnings: List[str] = []
    errors: List[str] = []

    if not traj.steps:
        errors.append("trajectory has no steps")
        return {"errors": errors, "warnings": warnings}

    for idx, step in enumerate(traj.steps):
        if step.t != idx:
            warnings.append(f"step index mismatch at {idx} -> {step.t}")
            break

    if traj.labels.when is not None:
        if traj.labels.when < 0 or traj.labels.when >= len(traj.steps):
            warnings.append("when label is out of range")

    if traj.labels.who:
        known = {a.name.lower() for a in traj.agents if a.name} | {a.role.lower() for a in traj.agents if a.role}
        if traj.labels.who.lower() not in known:
            warnings.append("who label not found in agents list")

    return {"errors": errors, "warnings": warnings}
