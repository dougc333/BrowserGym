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
    """
    Parses a flat Chromium AXTree into a readable, indented string.
    Specifically handles 'ignored' nodes and retrieves placeholders as labels.
    """
    axtree_data = obs.get('axtree_object', {})
    nodes = axtree_data.get('nodes', [])
    extra_props = obs.get('extra_element_properties', {})
    
    # 1. Convert BIDs to strings for reliable matching with node data
    valid_bids = {str(bid) for bid, p in extra_props.items() if p.get("visible", False)}
    
    # 2. Create a lookup map for the flat list of nodes
    node_map = {node['nodeId']: node for node in nodes}

    def build_string(node_id, indent=0):
        node = node_map.get(node_id)
        if not node:
            return None
        
        # Pull BID safely
        bg_id = node.get('browsergym_id')
        str_bg_id = str(bg_id) if bg_id is not None else None
        
        # Recursive step: get children's strings first
        child_lines = []
        for c_id in node.get('childIds', []):
            # If current node is 'ignored', don't increase indent for children
            next_indent = indent if node.get('ignored', False) else indent + 1
            res = build_string(c_id, next_indent)
            if res:
                child_lines.append(res)

        # PRUNING LOGIC
        # If node is 'ignored', we act as a transparent bridge to its children
        if node.get('ignored', False):
            return "\n".join(child_lines) if child_lines else None

        # Determine the name (Primary: Name field | Fallback: Placeholder)
        name = node.get('name', {}).get('value', '').strip()
        if not name:
            for prop in node.get('properties', []):
                if prop.get('name') in ['placeholder', 'aria-placeholder']:
                    name = prop.get('value', {}).get('value', '').strip()
                    break

        # KEEP CRITERIA: Keep if it has a valid BID or has valid children
        if str_bg_id in valid_bids or child_lines:
            role = node.get('role', {}).get('value', 'generic')
            
            # Formatting
            name_part = f" '{name}'" if name else ""
            bid_part = f" [{str_bg_id}]" if str_bg_id else ""
            current_line = f"{'  ' * indent}{role}{name_part}{bid_part}"
            
            return "\n".join([current_line] + child_lines)
        
        return None

    if not nodes:
        return "Empty AXTree"

    # Start recursion from the root node
    return build_string(nodes[0]['nodeId'])

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