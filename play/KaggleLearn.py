import random
import numpy as np

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages

# Where to save new model model
NEW_MODEL_PATH = "./models/new_agent"

# Adversarial model
ADVERSARY_MODEL_PATH = "./models/first_agent" 

# Set environment basic
env = ConnectFourGym(agent2="random")

# Initialize adversary model
model_adv = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
model_adv = PPO.load(ADVERSARY_MODEL_PATH)

# Build agent from model
def agent_adv(obs, config):
    # Use the best model to select a column
    col, _ = model_adv.predict(np.array(obs['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
   
# Set environment with adversary agent
env_adv = ConnectFourGym(agent2=agent_adv)

model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

# Train agent
model.learn(total_timesteps=60000)


# Save agent
model.save(NEW_MODEL_PATH)