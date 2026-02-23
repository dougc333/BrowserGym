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
    axtree_data = obs.get('axtree_object', {})
    nodes = axtree_data.get('nodes', [])
    extra_props = obs.get('extra_element_properties', {})

    # 1. Identify valid BIDs from extra properties
    valid_bids = {
        str(bid) for bid, props in extra_props.items() 
        if props.get("visible", False) or props.get("clickable", False)
    }

    # 2. Create a lookup table for quick node access
    node_map = {node['nodeId']: node for node in nodes}

    def format_node(node_id, indent=0):
        node = node_map.get(node_id)
        if not node or node.get('ignored', False):
            return None

        # Extract semantics
        role = node.get('role', {}).get('value', 'generic')
        name = node.get('name', {}).get('value', '').strip()
        bg_id = node.get('browsergym_id')
        
        # Format the line
        bid_str = f" [{bg_id}]" if bg_id else ""
        name_str = f" '{name}'" if name else ""
        current_line = f"{'  ' * indent}{role}{name_str}{bid_str}"

        # Recursively handle children
        child_lines = []
        for c_id in node.get('childIds', []):
            child_text = format_node(c_id, indent + 1)
            if child_text:
                child_lines.append(child_text)

        # PRUNING: Only keep the node if it has a BID we care about OR has valid children
        if str(bg_id) in valid_bids or child_lines:
            return "\n".join([current_line] + child_lines)
        
        return None

    # 3. Find the Root node (usually nodeId '2' or first node) to start
    root_id = nodes[0]['nodeId'] if nodes else None
    if not root_id:
        return "No nodes found."

    result = format_node(root_id)
    return result if result else "No visible elements found."
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
print("--------------------------------")

print(f"cleaned axtree: {get_cleaned_axtree(obs)}")
print("--------------------------------")
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