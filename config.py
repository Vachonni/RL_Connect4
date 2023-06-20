
#%%
import os

# Get path of current directory
path_config = os.getcwd()
# Get path parent RL_Connect4
path_parent = path_config.split("RL_Connect4")[0]
# Path to models
path_models = os.path.join(path_parent, "RL_Connect4", "models")

#%%
# PATH_BASE_MODEL = "random"
PATH_BASE_MODEL = os.path.join(path_models, "third_agent")

PATH_ADVERSARY_MODEL = os.path.join(path_models, "fourth_agent")

PATH_MEW_MODEL = os.path.join(path_models, "fifth_agent")


#%%
N_ROUNDS = 100

N_IT = 60000