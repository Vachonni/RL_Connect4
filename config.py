
#%%
import os

#%%
N_ROUNDS = 100

N_IT = 5000

PATIENCE = 3


#%% PATH CONFIG
# Get path of current directory
path_config = os.getcwd()
# Get path parent RL_Connect4
path_parent = path_config.split("RL_Connect4")[0]
# Path to models
path_models = os.path.join(path_parent, "RL_Connect4", "artifacts", "models")
# Path to learning curves
path_learning_curves = os.path.join(path_parent, "RL_Connect4", "artifacts", "learning_curves")



#%% MODELS PATHS
PATH_BASE_MODEL = "random"
# PATH_BASE_MODEL = os.path.join(path_models, "third_agent")

PATH_ADVERSARY_MODEL = os.path.join(path_models, "fourth_agent")

PATH_MEW_MODEL = os.path.join(path_models, f"stop_agent_{N_IT}_{N_ROUNDS}")


#%% LEARNING CURVES PATHS
PATH_ALL_WINS = os.path.join(path_learning_curves, f"all_wins_{N_IT}_{N_ROUNDS}.csv")

