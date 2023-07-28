import optuna

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_agent, model_to_agent



def learn_once(path_base_model, n_it=1000, params=None):

    # Load adversary agent (the one we want to learn beat)
    adv_agent = modelpath_to_agent(path_base_model)

    # Set environment that will reply with adversary agent
    env_adv = ConnectFourGym(agent2=adv_agent)
    model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0, **params)

    # Train new agent
    model.learn(total_timesteps=n_it, progress_bar=True)

    return model, adv_agent


def objective(trial):
    # Define parameters to tune
    params = {
        # "n_steps": 2 ** trial.suggest_int("exponent_n_steps", 3, 10),
        "gamma": trial.suggest_float("gamma", 0.9, 0.99999, log=True),
        "learning_rate": trial.suggest_float("lr", 1e-5, 1, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True),
    }

    n_it = trial.suggest_int("n_it", 1000, 5000)

    # Train model
    new_model, adv_agent = learn_once(path_base_model, n_it=n_it, params=params)

    # Evaluate model

    new_agent = model_to_agent(new_model)
    _, wins_new_agent = get_win_percentages(agent1=adv_agent,
                                            agent2=new_agent,
                                            n_rounds=100)

    return wins_new_agent


if __name__ == "__main__":

    path_base_model = "random"

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
