# run as colab cli python program
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io


def get_cleaned_axtree(obs):
    """
    Filters the AXTree to only include elements that BrowserGym 
    identifies as 'visible' or 'clickable' in the extra properties.
    """
    axtree_text = obs.get('axtree_object', "")
    extra_props = obs.get('extra_element_properties', {})
    
    # Identify BIDs that are actually useful
    # We keep elements that are visible, even if not clickable (for context)
    valid_bids = {
        bid for bid, props in extra_props.items() 
        if props.get("visible", False)
    }

    lines = axtree_text.split('\n')
    cleaned_lines = []

    for line in lines:
        # Extract the [bid] from the line (e.g., "[12] button 'Submit'")
        import re
        match = re.search(r'\[(\d+)\]', line)
        
        if match:
            bid = match.group(1)
            # Only keep the line if the BID is in our valid/visible set
            if bid in valid_bids:
                cleaned_lines.append(line)
        else:
            # Keep lines that don't have a BID (like "RootWebArea") for structure
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

env = gym.make(
    "browsergym/miniwob.click-button",
    headless=True,
    disable_env_checker=True,
)

obs, info = env.reset(seed=0)


print(f"obs: {type(obs)}")
print(f"obs keys: {obs.keys()}")
print(f"obs['chat_messages']:{obs['chat_messages']}")
print(f"obs['goal_object']:{obs['goal_object']}")
print("--------------------------------")
# print(f"obs['open_pages_urls']:{obs['open_pages_urls']}")
# print("--------------------------------")
# print(f"obs['open_pages_titles']:{obs['open_pages_titles']}")
# print("--------------------------------")
# print(f"obs['active_page_index']:{obs['active_page_index']}")
# print("--------------------------------")
# print(f"obs['url']:{obs['url']}")   
# print("--------------------------------")
# print(f"obs['screenshot'] shape:{obs['screenshot'].shape}")
# print("--------------------------------")
# print(f"obs['dom_object']:{obs['dom_object']}")
# print("--------------------------------")
print(f"obs['axtree_object']:{obs['axtree_object']}")
print("--------------------------------")
print(f"cleaned axtree: {get_cleaned_axtree(obs)}")
# print("--------------------------------")
# print(f"obs['extra_element_properties']:{obs['extra_element_properties']}")
# print("--------------------------------")
# print(f"obs['focused_element_bid']:{obs['focused_element_bid']}")
# print("--------------------------------")
print(f"obs['last_action']:{obs['last_action']}")
print("--------------------------------")
print(f"obs['last_action_error']:{obs['last_action_error']}")
print("--------------------------------")
print(f"obs['elapsed_time']:{obs['elapsed_time']}")
print("--------------------------------")
print(f"info: {type(info)}")
print(f"info keys: {info.keys()}")
print(f"type of info['task_info']:{type(info['task_info'])}")
print("--------------------------------")
goal = obs.get("goal", "")
print("GOAL:", goal)
print("--------------------------------")
axtree = obs.get('axtree_object')
print(f"axtree: {type(axtree)}")
print(f"axtree keys: {axtree.keys()}")
print(f"axtree: {axtree}")
print("--------------------------------")
#print(f"screenshot shape: {obs.get('screenshot').shape}")
#img = Image.fromarray(obs['screenshot'])
#img_byte_arr = io.BytesIO()
#img.save(img_byte_arr, format='PNG')


print("goal:", obs.get("goal"))
print("keys:", sorted(obs.keys()))
print("screenshot shape:", None if obs.get("screenshot") is None else obs["screenshot"].shape)

env.close()