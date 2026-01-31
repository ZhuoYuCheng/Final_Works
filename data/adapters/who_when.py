import json
import os
from typing import Dict, List, Optional

from data.adapters.base import BaseAdapter
from data.schema import Agent, Labels, Step, UnifiedTrajectory


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def infer_mistake_index(mistake_step, history_len: int, base_mode: str = "auto") -> Optional[int]:
    ms = _safe_int(mistake_step)
    if ms is None:
        return None

    if base_mode == "0":
        return ms
    if base_mode == "1":
        return ms - 1 if ms > 0 else None

    if ms < 0:
        return None
    if ms >= history_len and history_len > 0:
        # likely 1-based
        if ms - 1 < history_len:
            return ms - 1
        return history_len - 1
    return ms


class WhoWhenAdapter(BaseAdapter):
    def can_handle(self, raw_obj: Dict) -> bool:
        return "history" in raw_obj and "mistake_step" in raw_obj

    def convert(self, raw_obj: Dict, meta: dict) -> UnifiedTrajectory:
        history = raw_obj.get("history", []) or []

        agent_map: Dict[str, int] = {}
        agents: List[Agent] = []
        steps: List[Step] = []

        for idx, step in enumerate(history):
            role = step.get("role", "") or ""
            name = step.get("name", "") or ""
            agent_key = name.strip() or role.strip()
            if agent_key not in agent_map:
                agent_id = len(agents)
                agent_map[agent_key] = agent_id
                agents.append(Agent(agent_id=agent_id, role=role, name=name))
            agent_id = agent_map[agent_key]
            steps.append(
                Step(
                    t=idx,
                    agent_id=agent_id,
                    role=role,
                    name=name,
                    output=step.get("content", "") or "",
                    events=[],
                    raw=step,
                )
            )

        mistake_idx = infer_mistake_index(
            raw_obj.get("mistake_step"), len(steps), base_mode=meta.get("base_mode", "auto")
        )
        labels = Labels(when=mistake_idx, who=raw_obj.get("mistake_agent", ""))

        return UnifiedTrajectory(
            trace_id=raw_obj.get("question_ID") or raw_obj.get("id") or meta.get("fname", ""),
            query=raw_obj.get("question", ""),
            agents=agents,
            steps=steps,
            labels=labels,
            meta=meta,
        )


def load_who_when_dir(base_dir: str, subset: str, base_mode: str = "auto") -> List[UnifiedTrajectory]:
    subset_dir = os.path.join(base_dir, subset)
    items: List[UnifiedTrajectory] = []
    adapter = WhoWhenAdapter()
    for fname in sorted(os.listdir(subset_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(subset_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = {"dataset": "who_when", "subset": subset, "base_mode": base_mode, "fname": fname}
        if adapter.can_handle(data):
            items.append(adapter.convert(data, meta))
    return items
