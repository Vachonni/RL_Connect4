# This file is used to train a new agent that will learn to beat the adversary agent

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, model_to_agent
from config import PATH_BASE_MODEL, PATH_MEW_MODEL, N_ROUNDS, N_IT


def learn_once(path_base_model, path_new_model, n_it=60000, save=True):

    # Load adversary agent (the one we want to learn beat)
    adv_agent = modelpath_to_agent(path_base_model)

    # Set environment that will reply with adversary agent
    env_adv = ConnectFourGym(agent2=adv_agent)
    model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

    # Train new agent
    model.learn(total_timesteps=n_it)

    if save:
        # Save agent
        model.save(path_new_model)
    else:
        return model


def learn_multi(path_base_model, path_new_model, n_it=60000, test_rounds=1000):

    # Initial setup
    count = 0
    base_agent = modelpath_to_agent(path_base_model)

    # Train once
    new_model = learn_once(path_base_model, path_new_model, n_it, save=False)
    new_agent = model_to_agent(new_model)
    wins_base_agent, wins_new_agent = get_win_percentages(agent1=base_agent,
                                                          agent2=new_agent,
                                                          n_rounds=test_rounds)
    count += 1

    while wins_base_agent < wins_new_agent:
        print(f"\n\nRound: {count}.")
        print(f"New agent is better ({wins_new_agent} > {wins_base_agent}), saving it.")
        # Save agent
        new_model.save(path_new_model)
        # New model becones the base model
        path_base_model = path_new_model

        # Train once
        new_model = learn_once(path_base_model, path_new_model, n_it, save=False)
        new_agent = model_to_agent(new_model)
        wins_base_agent, wins_new_agent = get_win_percentages(agent1=base_agent,
                                                              agent2=new_agent,
                                                              n_rounds=test_rounds)
        count += 1

if __name__ == "__main__":

    # learn_once(path_base_model=PATH_BASE_MODEL, 
    #            path_new_model=PATH_MEW_MODEL,
    #           n_it=N_IT)
    learn_multi(path_base_model=PATH_BASE_MODEL, 
                path_new_model=PATH_MEW_MODEL, 
                n_it=N_IT,
                test_rounds=N_ROUNDS)
