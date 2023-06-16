#%%
import random
import numpy as np

from kaggle_environments import make
from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages


# Choose model
MODEL_PATH = "../models/first_agent"

#%%
# Set environment
env = ConnectFourGym(agent2="random")

# Load model used by agent
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
model = PPO.load(MODEL_PATH)

# Build agent from model
def agent(obs, config):
    # Use the best model to select a column
    col, _ = model.predict(np.array(obs['board']).reshape(1, 6,7))
    # Check if selected column is valid
    is_valid = (obs['board'][int(col)] == 0)
    # If not valid, select random move. 
    if is_valid:
        return int(col)
    else:
        return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
    

#%%
# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([agent, "random"])

#%%
# Visual of the game
env.render(mode="ipython")

#%%
# Play 100 games
print(get_win_percentages(agent1=agent, agent2="random", n_rounds=100))

