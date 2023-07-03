# Learning stop criteria

#%%
import os 

import pandas as pd

from azureml.core.run import Run

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, modelpath_to_model, model_to_agent
from config import N_ROUNDS, N_IT, N_LEARNING, PATIENCE, PATH_BASE_MODEL, PATH_MEW_MODEL, PATH_LEARNING_CURVES, PATH_PLOT

run = Run.get_context()


def learn_to_stop(path_base_model, path_new_model, i, n_it=1000, n_rounds=100):
    """The base model is the one we want to learn to beat.
    Hence, it is the adversary agent and the model we train."""

    # AGENT SETUP
    # Load adversary agent (the one we want to learn to beat)
    # adv_agent is random. Not a problem, its environment will be replaced by the adversary agent's environment
    adv_agent = modelpath_to_agent(path_base_model, adv_agent='random')

    # MODEL SETUP
    model = modelpath_to_model(path_base_model, adv_agent=adv_agent)
    
    # LOOP SETUP
    all_wins = {"wins_vs_base": [], "wins_vs_random": []}
    new_win = 0.5
    max_win = 0.5
    random_win = 0.5
    not_improving = []
    patience = PATIENCE
    stop_criteria = False

    # TRAINING LOOP
    while stop_criteria is False:
        print(f"\n\nRound: {i}.{len(all_wins['wins_vs_base'])} -- Score to beat: {max_win}")
        # Save scores
        all_wins["wins_vs_base"].append(new_win)
        all_wins["wins_vs_random"].append(random_win)

        # Update loop variables
        if new_win > max_win:
            print("New agent is better than old agent, saving new model.")
            # save new model
            max_win = new_win
            # Save improved model locally
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
        _, new_win = get_win_percentages(agent1=adv_agent,
                                         agent2=new_agent,
                                         n_rounds=n_rounds)
        _, random_win = get_win_percentages(agent1="random",
                                         agent2=new_agent,
                                         n_rounds=n_rounds)        

    # LOGGING RESULTS   
    run.log("max_win", max_win)

    # All wins as dataframe
    df_wins = pd.DataFrame(all_wins)

    # Add column with number of iterations
    df_wins["n_it"] = df_wins.index * n_it

    # Save best model to Azure (by loading it from local)
    new_model_name = os.path.basename(path_new_model)
    az_path_new_model = path_new_model + ".zip"
    run.upload_file(name=new_model_name, path_or_stream=az_path_new_model)

    return df_wins, max_win

    
def learn_to_stop_multi(path_base_model,
                        path_new_model,
                        path_learning_curves,
                        path_plot,
                        n_it,
                        n_rounds,
                        n_learning):
    
    for i in range(n_learning):
        # Adjust paths for ith iteration
        if i == 0:
            path_base_model_i = "random"
        else:
            path_base_model_i = path_base_model.replace("_i_", f"_{i-1}_")
        path_new_model_i = path_new_model.replace("_i_", f"_{i}_")
        path_learning_curves_i = path_learning_curves.replace("_i_", f"_{i}_")
        path_plot_i = path_plot.replace("_i_", f"_{i}_")

        df_wins, max_win = learn_to_stop(path_base_model_i,
                                         path_new_model_i,
                                         i,
                                         n_it,
                                         n_rounds)

        print(df_wins)

        # Save all wins
        df_wins.to_csv(path_learning_curves_i+str(int(max_win*100))+".csv", index=False)
    
        new_model_name = path_new_model_i.split("/")[-1]
        # Plot all wins with pandas plot
        fig = df_wins.plot(x="n_it", 
                        y=["wins_vs_base", "wins_vs_random"], 
                        title=f"Win ratio {new_model_name}",
                        ylabel="ratio").get_figure()
        # Save the plot with pandas plot
        fig.savefig(path_plot_i+str(int(max_win*100))+".png")

        # Save image log_image to Azure
        run.log_image(name="Learning curves",
                      plot=fig)

    

#%%
if __name__ == "__main__":

    learn_to_stop_multi(path_base_model=PATH_BASE_MODEL,
                        path_new_model=PATH_MEW_MODEL,
                        path_learning_curves=PATH_LEARNING_CURVES,
                        path_plot=PATH_PLOT,
                        n_it=N_IT,
                        n_rounds=N_ROUNDS,
                        n_learning=N_LEARNING)
    




# %%
