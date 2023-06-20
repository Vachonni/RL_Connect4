# Learning stop criteria

#%%
import pandas as pd

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, model_to_agent
from config import N_ROUNDS, N_IT, PATIENCE, PATH_BASE_MODEL, PATH_MEW_MODEL, PATH_ALL_WINS



def learn_to_stop(path_base_model, path_new_model, n_it=1000, n_rounds=100):

    # AGENT SETUP
    # Load adversary agent (the one we want to learn to beat)
    adv_agent = modelpath_to_agent(path_base_model)
    # Set environment that will reply as adversary agent
    env_adv = ConnectFourGym(agent2=adv_agent)
    model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

    # LOOP SETUP
    all_wins = []
    new_win = 0.5
    old_win = 0
    not_improving = []
    patience = PATIENCE
    stop_criteria = False


    while stop_criteria is False:
        print(f"\n\nRound: {len(all_wins)}.")
        # Save scores
        all_wins.append([new_win])

        # Update loop variables
        if new_win > old_win:
            print("New agent is better than old agent, saving new model.")
            # save new model
            model.save(PATH_MEW_MODEL)
            old_win = new_win
            not_improving = []
        else:
            not_improving.append(new_win)
            if len(not_improving) == patience:
                stop_criteria = True
                print("Stop criteria met. This will be the last round.")
            else:
                print(f"New agent is not better than old agent for {len(not_improving)} rounds.")
                print(f"Stop criteria will be met after {patience - len(not_improving)} more rounds.")

        # Train new agent
        print("Training new agent...")
        model.learn(total_timesteps=n_it)
        # Turn model into agent
        new_agent = model_to_agent(model)

        # Get win percentage
        print("Getting win percentage...")
        _, new_win = get_win_percentages(agent1=adv_agent,
                                         agent2=new_agent,
                                         n_rounds=n_rounds)

    # All wins as dataframe
    all_wins = pd.DataFrame(all_wins, columns=["pourcentage_wins"])

    # Add column with number of iterations
    all_wins["n_it"] = all_wins.index * n_it

    return all_wins


#%%
if __name__ == "__main__":
    all_wins = learn_to_stop(path_base_model=PATH_BASE_MODEL,
                             path_new_model=PATH_MEW_MODEL,
                             n_it=N_IT,
                             n_rounds=N_ROUNDS)

    print(all_wins)

    # Plot all wins with pandas plot
    all_wins.plot(x="n_it", y="pourcentage_wins")
    # Save all wins
    all_wins.to_csv(PATH_ALL_WINS, index=False)



# %%
