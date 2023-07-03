
#%%
import os

#%%
N_ROUNDS = 100

N_IT = 1000     # Size of the buffer, i.e. number of turns we train on

N_LEARNING = 10

PATIENCE = 4

MODEL_NAME= f"MULTI_MODELS_{N_IT}_{N_ROUNDS}_"
MODEL_NAME_I = MODEL_NAME + "i_" 


#%% PATH CONFIG
# Get the directory of the config script, that is at the root of the project
base_dir = os.path.dirname(os.path.abspath(__file__))
# Path to models
path_models = os.path.join(base_dir, "outputs", "models")
# Path to learning curves
path_learning_curves = os.path.join(base_dir, "outputs", "learning_curves")
# Path to plots
path_plots = os.path.join(base_dir, "outputs", "plots")



#%% MODELS PATHS
# PATH_BASE_MODEL = "random"
# PATH_BASE_MODEL = os.path.join(path_models, "fourth_agent")
# PATH_BASE_MODEL = os.path.join(path_models, f"stop_agent_5000_100_2_")
# PATH_BASE_MODEL = os.path.join(path_models, f"stop_agent_{N_IT}_{N_ROUNDS}_i_")
# PATH_BASE_MODEL = os.path.join(path_models, "Kernel4_5000_100_2_")
PATH_BASE_MODEL = os.path.join(path_models, MODEL_NAME_I)
# LIST_PATH_BASE_MODEL = [MODEL_NAME_I + str(i) for i in range(N_LEARNING)]

# PATH_ADVERSARY_MODEL = "random"
# PATH_ADVERSARY_MODEL = os.path.join(path_models, "stop_agent_6_1000_100")
# PATH_ADVERSARY_MODEL = os.path.join(path_models, f"stop_agent_5000_100_3_")
# PATH_ADVERSARY_MODEL = os.path.join(path_models, "fourth_agent")
PATH_ADVERSARY_MODEL = os.path.join(path_models, MODEL_NAME + "3_")


PATH_MEW_MODEL = os.path.join(path_models, MODEL_NAME_I)


#%% LEARNING CURVES PATHS
PATH_LEARNING_CURVES = os.path.join(path_learning_curves, MODEL_NAME_I)


#%% PLOTS PATHS
PATH_PLOT = os.path.join(path_plots, MODEL_NAME_I)

