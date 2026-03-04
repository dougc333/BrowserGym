# this has nothing to do wtih agentlab 

import os, re
import gymnasium as gym
import browsergym.miniwob  # noqa
from google import genai
from google.genai import types
from browsergym.core.action.highlevel import HighLevelActionSet

MODEL_ID = "gemini-2.5-flash"
ENV_ID = "browsergym/miniwob.click-button"

ACTION_PAT = re.compile(r"(?im)^\s*ACTION:\s*(.+?)\s*$")

def extract_action(text: str) -> str | None:
    if not text:
        return None
    m = ACTION_PAT.search(text)
    if m:
        return m.group(1).strip()
    # fallback: if model outputs just "click('14')" with no prefix
    t = text.strip().splitlines()[0].strip().strip("`")
    if t.startswith(("click(", "hover(", "fill(", "press(", "scroll(", "noop(")):
        return t
    return None

def get_clickables(obs):
    ax = obs["axtree_object"]
    clickables = []
    for n in ax.get("nodes", []):
        if n.get("ignored", False):
            continue
        bid = n.get("browsergym_id")
        role = str((n.get("role") or {}).get("value", "")).lower()
        name = str(((n.get("name") or {}).get("value")) or "")
        if bid is None:
            continue
        if role in {"button","link","menuitem","tab","option","listitem","checkbox","radio"}:
            clickables.append((str(bid), role, name))
    return clickables

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

env = gym.make(ENV_ID,action_mapping=HighLevelActionSet().to_python_code)
obs, info = env.reset()

NUM_STEPS=2 #this is num env.steps() different than num times we call env.reset(). 
# how many episodes do we need to create trajectories for SFT and RL. These are single step trajecctoies
# {a1,s1,a2,s2,a3,s3} 
for step in range(NUM_STEPS):
    goal = obs.get("goal", "")
    print(f"iteration:{step} goal:{goal}")
    clickables = get_clickables(obs)

    # compact list for the model
    clickable_lines = "\n".join([f"- bid='{bid}' role={role} name={name!r}"
                                 for bid, role, name in clickables[:60]])

    system_instruction = (
        "You are a BrowserGym agent.\n"
        "You MUST output exactly one line and nothing else.\n"
        "Format: ACTION: <action>\n"
        "Valid actions (BrowserGym):\n"
        "  click(bid)\n"
        "  hover(bid)\n"
        "  fill(bid, value)\n"
        "  press(bid, key_comb)\n"
        "  scroll(delta_x, delta_y)\n"
        "  noop(wait_ms)\n"
        "Use single quotes around string args, e.g. click('14').\n"
        "No explanation. No markdown. No extra text."
    )

    user_text = (
        f"GOAL:\n{goal}\n\n"
        f"CLICKABLES:\n{clickable_lines}\n\n"
        "Return exactly one line:\n"
        "ACTION: click('<bid>')\n"
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
        config=types.GenerateContentConfig(
            temperature=0,
            system_instruction=system_instruction,
        ),
    )

    raw = (response.text or "").strip()
    action = extract_action(raw) or "noop(200)"

    print(f"\n[step {step}] model_raw: {raw!r}")
    print(f"[step {step}] executing: {action}")

    obs, reward, terminated, truncated, info = env.step(action)
    print("reward:", reward, "terminated:", terminated, "last_err:", obs.get("last_action_error"))

    if terminated or truncated:
        break

env.close()