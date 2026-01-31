from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Agent:
    agent_id: int
    role: str = ""
    name: str = ""


@dataclass
class Step:
    t: int
    agent_id: int
    role: str = ""
    name: str = ""
    output: str = ""
    events: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Labels:
    when: Optional[int] = None
    who: Optional[str] = None


@dataclass
class UnifiedTrajectory:
    trace_id: str
    query: str
    agents: List[Agent]
    steps: List[Step]
    labels: Labels = field(default_factory=Labels)
    meta: Dict[str, Any] = field(default_factory=dict)
