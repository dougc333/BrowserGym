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

# def get_cleaned_axtree(obs):
#     axtree_data = obs.get('axtree_object', {})
#     nodes = axtree_data.get('nodes', [])
#     extra_props = obs.get('extra_element_properties', {})

#     # 1. Create a set of valid BIDs for fast lookup
#     valid_bids = {
#         str(bid) for bid, props in extra_props.items() 
#         if props.get("visible", False) or props.get("clickable", False)
#     }

#     # 2. Build a lookup map of {nodeId: node_dict}
#     node_map = {node['nodeId']: node for node in nodes}

#     # 3. Recursive helper to build the string
#     def build_string(node_id, indent=0):
#         node = node_map.get(node_id)
#         if not node:
#             return None
        
#         # Skip nodes that are explicitly ignored by the browser
#         if node.get('ignored', False):
#             # But we still want to check their children!
#             # (Node 5 in your data is ignored but leads to the actual content)
#             child_results = []
#             for c_id in node.get('childIds', []):
#                 res = build_string(c_id, indent) # Don't increase indent for ignored wrappers
#                 if res: child_results.append(res)
#             return "\n".join(child_results) if child_results else None

#         # Extract values from the nested 'value' keys in your data
#         role = node.get('role', {}).get('value', 'generic')
#         name = node.get('name', {}).get('value', '').strip()
#         bg_id = node.get('browsergym_id')
        
#         # Create the line text
#         bid_str = f" [{bg_id}]" if bg_id else ""
#         name_str = f" '{name}'" if name else ""
#         current_line = f"{'  ' * indent}{role}{name_str}{bid_str}"

#         # Process children
#         child_results = []
#         for c_id in node.get('childIds', []):
#             res = build_string(c_id, indent + 1)
#             if res:
#                 child_results.append(res)

#         # KEEP LOGIC: Keep this node if:
#         # It has a valid BID OR it has valid children
#         if (bg_id and str(bg_id) in valid_bids) or child_results:
#             return "\n".join([current_line] + child_results)
        
#         return None

#     # Start from the first node (usually RootWebArea)
#     if not nodes:
#         return "Empty AXTree"
    
#     final_tree = build_string(nodes[0]['nodeId'])
#     return final_tree if final_tree else "No visible actionable elements."


def get_cleaned_axtree(obs):
    axtree_data = obs.get('axtree_object', {})
    nodes = axtree_data.get('nodes', [])
    extra_props = obs.get('extra_element_properties', {})

    # 1. Map BIDs to strings and keep track of valid ones
    valid_bids = {str(bid) for bid, p in extra_props.items() if p.get("visible", False)}
    node_map = {node['nodeId']: node for node in nodes}

    actionable_elements = []

    def build_tree(node_id, level=0):
        node = node_map.get(node_id)
        if not node:
            return False
        
        # Tunnel through 'ignored' nodes without increasing level
        if node.get('ignored', False):
            has_valid_child = False
            for c_id in node.get('childIds', []):
                if build_tree(c_id, level):
                    has_valid_child = True
            return has_valid_child

        # --- Data Extraction ---
        role = node.get('role', {}).get('value', 'generic')
        name = node.get('name', {}).get('value', '').strip()
        bg_id = node.get('browsergym_id')
        str_bg_id = str(bg_id) if bg_id is not None else None

        # Fallback to placeholder/labels if name is empty
        if not name:
            for prop in node.get('properties', []):
                if prop.get('name') in ['placeholder', 'aria-placeholder', 'aria-label']:
                    name = prop.get('value', {}).get('value', '').strip()
                    break

        # Calculate Center Point (if bbox exists)
        center_point = None
        bbox = extra_props.get(str_bg_id, {}).get('bounding_box')
        if bbox:
            center_point = {
                "x": round(bbox['x'] + (bbox['width'] / 2), 2),
                "y": round(bbox['y'] + (bbox['height'] / 2), 2)
            }

        # Create the element record
        element_entry = {
            "role": role,
            "name": name,
            "bid": str_bg_id,
            "level": level,
            "center": center_point,
            "bbox": bbox
        }

        # --- Recursive Descent ---
        start_count = len(actionable_elements)
        for c_id in node.get('childIds', []):
            build_tree(c_id, level + 1)
        
        has_valid_children = len(actionable_elements) > start_count

        # --- Pruning Logic ---
        if str_bg_id in valid_bids or has_valid_children:
            # Insert parent before its children to maintain top-down order
            actionable_elements.insert(start_count, element_entry)
            return True
        
        return False

    if nodes:
        build_tree(nodes[0]['nodeId'])
    
    return actionable_elements


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
foo = get_cleaned_axtree(obs)
print(f"type cleaned axtree: {type(foo)}")

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