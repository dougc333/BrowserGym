import os
import sys
import browsergym.miniwob

from agentlab.experiments.study import make_study
from agentlab.agents.generic_agent import GenericAgentArgs

# 1. Configure Environment
os.environ["MINIWOB_URL"] = "http://localhost:8080/miniwob/"

# 2. Define your Agent using the 2026 Attribute Pattern
# Instead of passing arguments to __init__, we set them as attributes 
# or use the provided factory structure.
gemini_agent = GenericAgentArgs()
gemini_agent.model_name = "gemini-2.5-flash"
gemini_agent.temperature = 0.0

# 3. Initialize the Study
# make_study will now correctly pick up the attributes from the object
study = make_study(
    benchmark="miniwob",
    agent_args=[gemini_agent],
)

# 4. Run on CPU
print("Starting MiniWoB benchmark on CPU...")
study.run(n_jobs=2)