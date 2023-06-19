#%%
from kaggle_environments import make
from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent

from config import PATH_BASE_MODEL, PATH_ADVERSARY_MODEL, N_ROUNDS


def play_once(path_base_model, path_adv_model, render=True):
    # Load agents
    base_agent = modelpath_to_agent(path_base_model)
    adv_agent = modelpath_to_agent(path_adv_model)

    # Create the game environment
    env = make("connectx")

    # Two random agents play one game round
    env.run([base_agent, adv_agent])

    # Visual of the game
    if render:
        env.render(mode="ipython")

    
def play_multi(path_base_model, path_adv_model, n_rounds=1000):
    # Load agents
    base_agent = modelpath_to_agent(path_base_model)
    adv_agent = modelpath_to_agent(path_adv_model)

    # Play games
    _, _ = get_win_percentages(agent1=base_agent, 
                               agent2=adv_agent, 
                               n_rounds=n_rounds)
    


if __name__ == "__main__":
    play_once(path_base_model=PATH_BASE_MODEL, path_adv_model=PATH_ADVERSARY_MODEL, render=True)
    play_multi(path_base_model=PATH_BASE_MODEL, path_adv_model=PATH_ADVERSARY_MODEL, n_rounds=N_ROUNDS)


