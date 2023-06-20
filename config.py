
#%%
import os

#%%
N_ROUNDS = 100

N_IT = 1000

PATIENCE = 20


#%% PATH CONFIG
# Get path of current directory
path_config = os.getcwd()
# Get path parent RL_Connect4
path_parent = path_config.split("RL_Connect4")[0]
# Path to models
path_models = os.path.join(path_parent, "RL_Connect4", "artifacts", "models")
# Path to learning curves
path_learning_curves = os.path.join(path_parent, "RL_Connect4", "artifacts", "learning_curves")
# Path to plots
path_plots = os.path.join(path_parent, "RL_Connect4", "artifacts", "plots")



#%% MODELS PATHS
# PATH_BASE_MODEL = "random"
PATH_BASE_MODEL = os.path.join(path_models, "stop_agent_BR0_1000_100")

PATH_ADVERSARY_MODEL = os.path.join(path_models, "fourth_agent")

PATH_MEW_MODEL = os.path.join(path_models, f"stop_agent_AR1_{N_IT}_{N_ROUNDS}")


#%% LEARNING CURVES PATHS
PATH_LEARNING_CURVES = os.path.join(path_learning_curves, f"df_wins_AR1_{N_IT}_{N_ROUNDS}")


#%% PLOTS PATHS
PATH_PLOT = os.path.join(path_plots, f"graph_AR1_{N_IT}_{N_ROUNDS}")

