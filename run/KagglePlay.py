#%%
from kaggle_environments import make
from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import model_to_agent

from config import PATH_BASE_MODEL, PATH_ADVERSARY_MODEL, N_ROUNDS

#%%
# Load agents
base_agent = model_to_agent(PATH_BASE_MODEL)
adv_agent = model_to_agent(PATH_ADVERSARY_MODEL)

#%%
# Create the game environment
env = make("connectx")

# Two random agents play one game round
env.run([base_agent, adv_agent])

#%%
# Visual of the game
env.render(mode="ipython")

#%%
# Play games
print(get_win_percentages(agent1=base_agent, 
                          agent2=adv_agent, 
                          n_rounds=N_ROUNDS))


# %%
