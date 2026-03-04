import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------
# Utilities
# ---------------------------

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def tokens(s: str) -> List[str]:
    s = norm(s)
    return [t for t in re.split(r"[^a-z0-9]+", s) if t]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def parse_goal_intent(goal: str) -> str:
    g = norm(goal)
    if any(w in g for w in ["date", "calendar", "month", "year", "choose-date", "pick a date"]):
        return "choose_date"
    if any(w in g for w in ["slider", "drag", "set the value", "use-slider"]):
        return "use_slider"
    if any(w in g for w in ["inbox", "email", "message", "reply", "archive", "delete"]):
        return "email_inbox"
    if any(w in g for w in ["type", "enter", "fill", "input"]):
        return "type"
    return "click"

# ---------------------------
# Candidate representation
# ---------------------------

@dataclass
class Node:
    bid: str
    role: str = ""
    name: str = ""
    value: str = ""
    disabled: bool = False
    clickable: bool = True
    typeable: bool = False
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # optional

# You need to implement this for your obs format:
def extract_nodes_from_obs(obs: Dict[str, Any]) -> List[Node]:
    """
    Adapt this to BrowserGym's obs you are using.
    If you use click-by-bid style obs["extra_element_properties"], map each element to Node.
    If you use obs["axtree"], parse nodes and attach bid/name/role.
    """
    nodes: List[Node] = []
    props = obs.get("extra_element_properties")
    if isinstance(props, list):
        for e in props:
            bid = str(e.get("bid", ""))
            if not bid:
                continue
            nodes.append(Node(
                bid=bid,
                role=str(e.get("role", "")),
                name=str(e.get("text", e.get("name", ""))),
                value=str(e.get("value", "")),
                disabled=bool(e.get("disabled", False)),
                clickable=bool(e.get("clickable", True)),
                typeable=bool(e.get("typeable", False)),
                bbox=tuple(e.get("bbox", (0,0,0,0))) if e.get("bbox") else (0,0,0,0),
            ))
    return nodes

# ---------------------------
# Ranker
# ---------------------------

def rank_nodes(goal: str, nodes: List[Node], want: str) -> List[Tuple[float, Node]]:
    gt = tokens(goal)
    scored = []
    for n in nodes:
        if n.disabled:
            continue
        if want == "type" and not n.typeable:
            continue
        if want == "click" and not n.clickable:
            continue

        nt = tokens(n.name) + tokens(n.value)
        s = 0.0
        s += 2.0 * jaccard(gt, nt)

        # role priors
        r = norm(n.role)
        if want == "type" and any(k in r for k in ["textbox", "input", "textarea"]):
            s += 0.5
        if want == "click" and any(k in r for k in ["button", "link", "menuitem", "checkbox", "radio", "gridcell"]):
            s += 0.3

        scored.append((s, n))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored

# ---------------------------
# Memory + Options
# ---------------------------

@dataclass
class AgentMem:
    intent: str = ""
    phase: str = ""
    tried: set = field(default_factory=set)
    target: Dict[str, Any] = field(default_factory=dict)
    selected: Dict[str, Any] = field(default_factory=dict)
    step: int = 0

class BaseOption:
    def step(self, obs: Dict[str, Any], nodes: List[Node], mem: AgentMem) -> Dict[str, Any]:
        raise NotImplementedError

class GenericOption(BaseOption):
    def step(self, obs, nodes, mem):
        goal = obs.get("goal", "")
        want = "type" if mem.intent == "type" else "click"
        ranked = rank_nodes(goal, nodes, want=want)
        if not ranked:
            return {"type": "noop"}  # depends on your env
        _, n = ranked[0]
        if want == "click":
            return {"type": "click", "bid": n.bid}
        else:
            # naive: type goal's quoted string or last token; customize for your tasks
            text = obs.get("target_text") or goal
            return {"type": "type", "bid": n.bid, "text": text}

# Placeholders: for your three tasks you can implement the FSM we discussed.
class SliderOption(BaseOption):
    def step(self, obs, nodes, mem):
        # Find slider node by role/value fields, focus it, then press arrows until target met.
        # Return {"type":"press","key":"ArrowRight"} style actions or drag actions.
        return {"type": "noop"}

class ChooseDateOption(BaseOption):
    def step(self, obs, nodes, mem):
        # Parse target date from goal -> open picker -> navigate -> select day
        return {"type": "noop"}

class EmailInboxOption(BaseOption):
    def step(self, obs, nodes, mem):
        # scan rows -> open -> verify -> act -> back
        return {"type": "noop"}

# ---------------------------
# Agent
# ---------------------------

class BrowserGymAgent:
    def __init__(self):
        self.mem = AgentMem()
        self.opts = {
            "click": GenericOption(),
            "type": GenericOption(),
            "use_slider": SliderOption(),
            "choose_date": ChooseDateOption(),
            "email_inbox": EmailInboxOption(),
        }

    def reset(self):
        self.mem = AgentMem()

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.mem.step += 1
        goal = obs.get("goal", "")
        if not self.mem.intent:
            self.mem.intent = parse_goal_intent(goal)

        nodes = extract_nodes_from_obs(obs)
        opt = self.opts.get(self.mem.intent, self.opts["click"])
        return opt.step(obs, nodes, self.mem)
