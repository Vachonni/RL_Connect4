import random
import numpy as np

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs
from src.ModelToAgent import model_to_agent
from config import PATH_ADVERSARY_MODEL, PATH_MEW_MODEL

# Load adversary agent (the one we want to learn beat)
adv_agent = model_to_agent(PATH_ADVERSARY_MODEL)

# Set environment that will reply with adversary agent
env_adv = ConnectFourGym(agent2=adv_agent)
model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

# Train new agent
model.learn(total_timesteps=60000)

# Save agent
model.save(PATH_MEW_MODEL)
