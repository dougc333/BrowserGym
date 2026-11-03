"""
Wire protocol between the browser front end and this server.

Every message is a JSON object with a `type` field. The front end validates
everything it receives with Zod (web/src/shared/schemas.ts); this file is the
Python mirror, validated with Pydantic. Keep the two in sync.

Vocabulary (RL terms in brackets):
  observation  [o_t]  what the policy sees at step t (goal + AXTree + screenshot)
  proposal     [a_t ~ pi(.|o_t)]  what the policy wants to do, with its reasoning
  step         [(o_t, a_t, r_t, done)]  one recorded transition
  episode      [tau]  a full trajectory, the unit we save for behavior cloning
"""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

PolicyName = Literal["claude", "random", "human"]
Phase = Literal["idle", "resetting", "observing", "thinking", "awaiting_approval", "acting", "done"]


# ---------------------------------------------------------------- records


class Observation(BaseModel):
    t: int
    goal: str
    url: str
    open_tabs: list[str] = []
    axtree: str  # flattened accessibility tree, every interactive node prefixed by its [bid]
    screenshot: Optional[str] = None  # data:image/jpeg;base64,...
    last_action: str = ""
    last_action_error: str = ""


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Prompt(BaseModel):
    system: str
    user: str


class Proposal(BaseModel):
    t: int
    policy: PolicyName
    model: Optional[str] = None
    thought: str
    action: str
    raw: str = ""
    prompt: Optional[Prompt] = None
    latency_ms: int = 0
    usage: Usage = Usage()


class Step(BaseModel):
    t: int
    obs: Observation
    proposal: Optional[Proposal] = None  # what the policy suggested (may differ from `action`)
    action: str  # what was actually executed
    action_source: PolicyName  # who decided the executed action
    prompt: Optional[Prompt] = None  # exact policy input at o_t (the BC "x"), even for human steps
    thought: str = ""
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    action_error: str = ""  # error raised while executing `action`, if any
    env_ms: int = 0


class EpisodeSummary(BaseModel):
    id: str
    task: str
    seed: int
    policy: PolicyName
    model: Optional[str] = None
    started_at: float
    ended_at: Optional[float] = None
    n_steps: int = 0
    total_reward: float = 0.0
    success: bool = False
    terminated: bool = False
    truncated: bool = False
    human_interventions: int = 0


class Episode(EpisodeSummary):
    goal: str = ""
    steps: list[Step] = []


# ---------------------------------------------------------------- client -> server


class ResetCmd(BaseModel):
    type: Literal["reset"]
    task: str
    seed: int = 0
    max_steps: int = 10
    headless: bool = True
    use_screenshot: bool = False  # also send the screenshot to Claude (multimodal)


class ThinkCmd(BaseModel):
    type: Literal["think"]
    policy: PolicyName = "claude"
    model: Optional[str] = None
    temperature: Optional[float] = None


class ActCmd(BaseModel):
    type: Literal["act"]
    action: str
    source: PolicyName = "human"
    thought: str = ""


class RunCmd(BaseModel):
    type: Literal["run"]
    policy: PolicyName = "claude"
    model: Optional[str] = None
    temperature: Optional[float] = None
    supervised: bool = False  # pause for approval before every action


class PauseCmd(BaseModel):
    type: Literal["pause"]


class BatchCmd(BaseModel):
    type: Literal["batch"]
    tasks: list[str]
    seeds: list[int]
    policy: PolicyName = "claude"
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_steps: int = 10
    use_screenshot: bool = False


class CancelCmd(BaseModel):
    type: Literal["cancel"]


ClientCommand = Union[ResetCmd, ThinkCmd, ActCmd, RunCmd, PauseCmd, BatchCmd, CancelCmd]


class Envelope(BaseModel):
    cmd: ClientCommand = Field(discriminator="type")
