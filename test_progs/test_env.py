# run as colab cli python program
import gymnasium as gym
import browsergym.miniwob
import numpy as np
import ollama
from PIL import Image
import io

def filter_axtree_json(node, valid_bids):
    """
    Recursively filters a JSON-style AXTree to only include 
    nodes that have a BID in the valid_bids set.
    """
    # 1. Check if this node itself has a BID and if that BID is valid
    # BrowserGym nodes usually store the BID in a 'bid' or 'browsergym_id' key
    node_bid = str(node.get('bid', ''))
    
    # 2. Recursively filter children first
    children = node.get('children', [])
    filtered_children = [filter_axtree_json(child, valid_bids) for child in children]
    
    # Remove None values from the children list
    filtered_children = [c for c in filtered_children if c is not None]
    
    # 3. Decision Logic: Keep this node if...
    # - It has a valid BID
    # - OR it has children that are valid (to keep the structure alive)
    # - OR it's a structural node (like RootWebArea) with no BID
    if node_bid in valid_bids or filtered_children or not node_bid:
        # Create a copy of the node with only the filtered children
        new_node = node.copy()
        new_node['children'] = filtered_children
        return new_node
    
    return None # Prune this branch


import re

def get_cleaned_axtree(obs):
    axtree_obj = obs.get('axtree_object', {})
    extra_props = obs.get('extra_element_properties', {})
    
    # 1. Identify all BIDs that are actually visible/clickable
    # Use strings for keys to avoid type-mismatch errors
    valid_bids = {
        str(bid) for bid, props in extra_props.items() 
        if props.get("visible", False) or props.get("clickable", False)
    }

    # 2. Recursive helper to build a cleaned string from the JSON tree
    def build_tree_string(node, indent=0):
        # Extract BID from the node (BrowserGym usually uses 'bid' as the key)
        node_bid = node.get('bid')
        str_bid = str(node_bid) if node_bid is not None else None
        
        # Determine if we should keep this node
        # We keep it if it has a valid BID or if it's a structural parent
        has_children = len(node.get('children', [])) > 0
        
        # Prepare the current line text
        role = node.get('role', 'Generic')
        name = node.get('name', '').strip()
        name_str = f' "{name}"' if name else ""
        bid_str = f' [{str_bid}]' if str_bid else ""
        
        current_line = f"{'  ' * indent}{role}{name_str}{bid_str}"

        # Recursively process children
        child_lines = []
        for child in node.get('children', []):
            child_text = build_tree_string(child, indent + 1)
            if child_text: # Only add if the child wasn't pruned
                child_lines.append(child_text)

        # PRUNING LOGIC: 
        # Keep node if it's a valid BID, OR it has valid children
        if str_bid in valid_bids or child_lines:
            return "\n".join([current_line] + child_lines)
        return None

    # 3. Start the process from the root
    cleaned_string = build_tree_string(axtree_obj)
    return cleaned_string if cleaned_string else "No visible elements found."


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
print("--------------------------------")
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