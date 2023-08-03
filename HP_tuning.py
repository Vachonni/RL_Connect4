#%%
import optuna
import joblib

from stable_baselines3 import DQN
from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, model_to_agent

#%%

def learn_once(path_base_model, n_it=1000, params=None):

    # Load adversary agent (the one we want to learn beat)
    adv_agent = modelpath_to_agent(path_base_model)

    # Set environment that will reply with adversary agent
    env_adv = ConnectFourGym(agent2=adv_agent)
    # model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0, **params)
    model = DQN("MlpPolicy", env_adv, verbose=0, **params)

    # Train new agent
    model.learn(total_timesteps=n_it, progress_bar=True)

    return model, adv_agent


def objective(trial):
    # Define parameters to tune
    params = {
        # "n_steps": 2 ** trial.suggest_int("exponent_n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "buffer_size": trial.suggest_categorical("buffer_size", [1000000, 2000000]),
        # "ent_coef": trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True),
    }

    n_it = 100000# trial.suggest_int("n_it", 100000, 500000, 400000)

    # Train model
    new_model, adv_agent = learn_once(path_base_model, n_it=n_it, params=params)

    # Save model with trial number
    print("Saving model...")
    new_model.save(f"HP_tuning_DQN_trail_{trial.number}")
    print("Model saved!")

    # Evaluate model
    new_agent = model_to_agent(new_model)
    _, wins_new_agent = get_win_percentages(agent1=adv_agent,
                                            agent2=new_agent,
                                            n_rounds=100)

    return wins_new_agent

#%%
if __name__ == "__main__":

    path_base_model = "random"

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)


    # %%
    best_trial = study.best_trial
    best_trial

    #%%
    dfi = study.trials_dataframe()
    dfi.plot.scatter(y='value', x='params_n_it')


    #%%
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()

    #%%
    fig = optuna.visualization.plot_slice(study)
    fig.show()

    #%%
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.show()

    #%%
    study.optimize(objective, n_trials=10)








    #%%
    import joblib
    # Save study with joblib
    joblib.dump(study, "HP_study_DQN.pkl")
    #%%
    # Load study
    study_load = joblib.load("HP_study.pkl")
    print("Best trial until now:")
    print(" Value: ", study_load.best_trial.value)
    print(" Params: ")
    for key, value in study_load.best_trial.params.items():
        print(f"    {key}: {value}")
# %%
