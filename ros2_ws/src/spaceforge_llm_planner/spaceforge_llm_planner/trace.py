"""Complete, serializable planner decision traces."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import json

@dataclass
class DecisionTrace:
    trace_id: str
    prompt_version: str
    schema_version: str
    provider: str
    request: Dict[str, Any]
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    raw_output: Any = None
    parsed_output: Any = None
    parse_status: str = "not_attempted"
    mission_status: str = "not_sent"
    mission_reason: str = ""
    token_count: int = 0
    cost_usd: float = 0.0
    failure: str = ""

    def to_dict(self): return asdict(self)
    def to_json(self): return json.dumps(self.to_dict(), sort_keys=True)
