
#%%
import os

import pandas as pd

from kaggle_environments import make
from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent

from config import PATH_BASE_MODEL, PATH_ADVERSARY_MODEL, PATH_MEW_MODEL, N_ROUNDS, LIST_OF_MODELS

#%%
def play_once(path_base_model, path_adv_model, render=True):
    # Load agents
    base_agent = modelpath_to_agent(path_base_model)
    adv_agent = modelpath_to_agent(path_adv_model)

    # Create the game environment
    env = make("connectx")

    # Two random agents play one game round
    details = env.run([base_agent, adv_agent])
    print(details)
    if details[-1][0]['reward'] == 1:
        print("==> Winner: base_agent (1)")
    else:
        print("==> Winner: adv_agent (2)")

    # Visual of the game
    if render:
        env.render(mode="ipython")

    # # Run interactive game
    # env.play([base_agent, None])

    
def play_multi(path_base_model, path_adv_model, n_rounds=1000):
    # Load agents
    base_agent = modelpath_to_agent(path_base_model)
    adv_agent = modelpath_to_agent(path_adv_model)

    # Play games
    wins_base_agent, wins_new_agent  = get_win_percentages(agent1=base_agent, 
                                                           agent2=adv_agent, 
                                                           n_rounds=n_rounds)
    
    return wins_base_agent, wins_new_agent 

    
def play_list(list_of_models, path_adv_model, n_rounds=1000):

    results = {"new_model_wins": [], "base_model": []}

    for path_base_model in list_of_models:
        _, wins_new_agent = play_multi(path_base_model=path_base_model,
                                       path_adv_model=path_adv_model,
                                       n_rounds=n_rounds)
        results["new_model_wins"].append(wins_new_agent)
        results["base_model"].append(os.path.basename(path_base_model))
    
    return results



if __name__ == "__main__":
    #%%
    PATH_BASE_MODEL = LIST_OF_MODELS[5]
    play_once(path_base_model=PATH_BASE_MODEL, path_adv_model=PATH_MEW_MODEL, render=True)
    #%%
    _, _ = play_multi(path_base_model=PATH_BASE_MODEL, path_adv_model=PATH_ADVERSARY_MODEL, n_rounds=N_ROUNDS)

    #%%
    results = play_list(list_of_models=LIST_OF_MODELS, path_adv_model=PATH_MEW_MODEL, n_rounds=N_ROUNDS)
    df_results = pd.DataFrame(results)
    print(df_results)
    print(f'==> Mean: {df_results["new_model_wins"].mean()}')




# %%
