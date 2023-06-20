# Learning stop criteria

#%%
import pandas as pd

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, model_to_agent
from config import N_ROUNDS, N_IT, PATIENCE, PATH_BASE_MODEL, PATH_MEW_MODEL, PATH_LEARNING_CURVES, PATH_PLOT


def learn_to_stop(path_base_model, path_new_model, n_it=1000, n_rounds=100):

    # AGENT SETUP
    # Load adversary agent (the one we want to learn to beat)
    base_agent = modelpath_to_agent(path_base_model)
    # Set environment that will reply as adversary agent
    env_adv = ConnectFourGym(agent2=base_agent)
    model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

    # LOOP SETUP
    all_wins = {"wins_vs_base": [], "wins_vs_random": []}
    new_win = 0.5
    max_win = 0.5
    random_win = 0.5
    not_improving = []
    patience = PATIENCE
    stop_criteria = False


    while stop_criteria is False:
        print(f"\n\nRound: {len(all_wins['wins_vs_base'])} -- Score to beat: {max_win}")
        # Save scores
        all_wins["wins_vs_base"].append(new_win)
        all_wins["wins_vs_random"].append(random_win)

        # Update loop variables
        if new_win > max_win:
            print("New agent is better than old agent, saving new model.")
            # save new model
            max_win = new_win
            model.save(path_new_model)
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
        _, new_win = get_win_percentages(agent1=base_agent,
                                         agent2=new_agent,
                                         n_rounds=n_rounds)
        _, random_win = get_win_percentages(agent1="random",
                                         agent2=new_agent,
                                         n_rounds=n_rounds)        

    # All wins as dataframe
    df_wins = pd.DataFrame(all_wins)

    # Add column with number of iterations
    df_wins["n_it"] = df_wins.index * n_it

    return df_wins, max_win


#%%
if __name__ == "__main__":
    df_wins, max_win = learn_to_stop(path_base_model=PATH_BASE_MODEL,
                             path_new_model=PATH_MEW_MODEL,
                             n_it=N_IT,
                             n_rounds=N_ROUNDS)

    print(df_wins)

    # Save all wins
    df_wins.to_csv(PATH_LEARNING_CURVES+"_"+str(int(max_win*100))+".csv", index=False)
   
    new_model_name = PATH_MEW_MODEL.split("/")[-1]
    # Plot all wins with pandas plot
    fig = df_wins.plot(x="n_it", 
                       y=["wins_vs_base", "wins_vs_random"], 
                       title=f"Win ratio {new_model_name}",
                       ylabel="ratio").get_figure()
    # Save the plot with panqdas plot
    fig.savefig(PATH_PLOT+"_"+str(int(max_win*100))+".png")




# %%
