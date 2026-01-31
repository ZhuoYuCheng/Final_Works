import json
import os
from typing import Dict, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema import Agent, Labels, Step, UnifiedTrajectory


class AegisAdapter(BaseAdapter):
    def _normalize_name(self, name: str) -> str:
        return (name or "").strip().lower()

    def _pick_faulty(
        self, history: List[Dict], faulty_names: List[str]
    ) -> Tuple[Optional[int], str]:
        faulty_set = {self._normalize_name(x) for x in faulty_names if x}
        for idx, step in enumerate(history):
            name = self._normalize_name(step.get("agent_name", ""))
            if name in faulty_set:
                return idx, step.get("agent_name", "")
        if faulty_names:
            return None, faulty_names[0] or ""
        return None, ""

    def can_handle(self, raw_obj: Dict) -> bool:
        return "input" in raw_obj and "output" in raw_obj and "conversation_history" in raw_obj.get("input", {})

    def convert(self, raw_obj: Dict, meta: dict) -> UnifiedTrajectory:
        history = raw_obj.get("input", {}).get("conversation_history", []) or []
        faulty = raw_obj.get("output", {}).get("faulty_agents", []) or []
        faulty_names = [
            f.get("agent_name", "") for f in faulty if isinstance(f, dict) and f.get("agent_name")
        ]
        earliest_idx, earliest_agent = self._pick_faulty(history, faulty_names)

        agent_map: Dict[str, int] = {}
        agents: List[Agent] = []
        steps: List[Step] = []
        mistake_idx: Optional[int] = earliest_idx

        for idx, step in enumerate(history):
            name = step.get("agent_name", "") or ""
            agent_key = name
            if agent_key not in agent_map:
                agent_id = len(agents)
                agent_map[agent_key] = agent_id
                agents.append(Agent(agent_id=agent_id, role=name, name=name))
            agent_id = agent_map[agent_key]
            steps.append(
                Step(
                    t=idx,
                    agent_id=agent_id,
                    role=name,
                    name=name,
                    output=step.get("content", "") or "",
                    events=[],
                    raw=step,
                )
            )

        labels = Labels(when=mistake_idx, who=earliest_agent or "")

        query = raw_obj.get("input", {}).get("query", "")
        trace_id = raw_obj.get("id") or meta.get("fname", "")

        return UnifiedTrajectory(
            trace_id=trace_id,
            query=query,
            agents=agents,
            steps=steps,
            labels=labels,
            meta=meta,
        )


def load_aegis_dir(base_dir: str) -> List[UnifiedTrajectory]:
    items: List[UnifiedTrajectory] = []
    adapter = AegisAdapter()
    for fname in sorted(os.listdir(base_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(base_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = {
            "dataset": "aegis",
            "fname": fname,
            "framework": data.get("metadata", {}).get("framework", ""),
            "benchmark": data.get("metadata", {}).get("benchmark", ""),
            "model": data.get("metadata", {}).get("model", ""),
        }
        if adapter.can_handle(data):
            items.append(adapter.convert(data, meta))
    return items
