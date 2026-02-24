
vec_ppo_video_debug
 - n_env=1, multiple envs dont work. not worth debugging 
 - her, should get 80%~ hindsight experience replay, uses bad trajectories. 
 

tricky: headless browsers default to 800x600 you can set the default resolution in env.reset()
can set on action. axtree changes after each action.  
# The action string format:
action = "set_window_size(width=1024, height=768)"

the nodeIDs are in axtree
for each nodeId there is a browsergym_id 
use browsergym_id for getting bounding boxes in obs[extra_element_properties]
