"""
File to train models to play against other models.
Includes a function to train a model against models from all previous levels.
"""
import os
import random

import pandas as pd

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs, get_win_percentages
from src.ModelToAgent import modelpath_to_model, modelpath_to_agent, model_to_agent
from run.KagglePlay import play_list_score
from config import PATH_MEW_MODEL, N_ROUNDS, N_IT, N_LEARNING, EVAL_TRESHOLD, LIST_OF_MODELS


def learn_once(path_base_model, path_new_model, n_it=60000, save=True):
    """
    Function to train a model once against another model.
    """

    # Base agent will be adversary (the one we want to learn beat)
    adv_agent = modelpath_to_agent(path_base_model)

    # Set environment that will reply with adversary agent
    env_adv = ConnectFourGym(agent2=adv_agent)
    new_model = PPO("CnnPolicy", env_adv, policy_kwargs=policy_kwargs, verbose=0)

    # Train new model
    new_model.learn(total_timesteps=n_it)

    if save:
        # Save model
        new_model.save(path_new_model)

    return new_model, adv_agent


def eval_once(new_model, adv_agent, n_rounds=100):
    """
    Function to evaluate a model once against another model.
    """

    # Create agent for new model
    new_agent = model_to_agent(new_model)

    # Evaluate agents
    wins_base_agent, wins_new_agent = get_win_percentages(agent1=adv_agent,
                                                          agent2=new_agent,
                                                          n_rounds=n_rounds)

    return wins_base_agent, wins_new_agent


def learn_n_eval_once(path_base_model, path_new_model, n_it=60000, n_rounds=100, save=True):
    """
    Function to train and evaluate a model once against another model.
    """

    # Train new model
    new_model, adv_agent = learn_once(path_base_model=path_base_model,
                                      path_new_model=path_new_model,
                                      n_it=n_it, 
                                      save=save)

    # Evaluate model
    wins_base_agent, wins_new_agent = eval_once(new_model=new_model,
                                                adv_agent=adv_agent,
                                                n_rounds=n_rounds)

    return new_model, wins_base_agent, wins_new_agent


def learn_against_list_models(list_models, path_new_model, n_it=60000, n_rounds=100):
    """
    Function to train a model against a list of models.
    """

    eval = {"new_model_wins": [], "base_model": [], "eval_avrg": []}
    qt_models = len(list_models)

    # Train once against each model, random order
    shuff_list_models = list_models.copy()
    random.shuffle(shuff_list_models)
    for path_base_model in shuff_list_models:
        print(f"\n\nTraining against {os.path.basename(path_base_model)}.")
        # Train and evaluate new model against selected model
        _, _, wins_new_agent = learn_n_eval_once(path_base_model=path_base_model,
                                                 path_new_model=path_new_model,
                                                 n_it=n_it)
        print(f"New agent win ratio {wins_new_agent}.")
        eval["new_model_wins"].append(wins_new_agent)
        eval["base_model"].append(os.path.basename(path_base_model))
        eval["eval_avrg"].append(0.5)
    
    print("Initial shuffle training againsts all models COMPLETED")

    # Until new model average win ratio is lower than 50% against last qt_models wins, continue training it agains models selected randomly
    eval_avrg = sum(eval["new_model_wins"][-qt_models:]) / qt_models
    print(f"\n--> Evaluation average is at: {eval_avrg}, threshold at {EVAL_TRESHOLD}.")
    eval["eval_avrg"][-1] = eval_avrg
    while eval_avrg < EVAL_TRESHOLD:
        # Select a model randomly
        path_base_model = random.choice(list_models)
        print(f"\n\nTraining against {os.path.basename(path_base_model)}.")
        # Train and evaluate new model against selected model
        _, _, wins_new_agent = learn_n_eval_once(path_base_model=path_base_model,
                                                 path_new_model=path_new_model,
                                                 n_it=n_it)
        print(f"New agent win ratio {wins_new_agent}.")
        eval_avrg = sum(eval["new_model_wins"][-qt_models:]) / qt_models
        print(f"\n--> Evaluation average is at: {eval_avrg}, threshold at {EVAL_TRESHOLD}.")
        eval["new_model_wins"].append(wins_new_agent)
        eval["base_model"].append(os.path.basename(path_base_model))
        eval["eval_avrg"].append(eval_avrg)
    
    # Save evaluation
    outputs_dir = os.path.dirname(os.path.dirname((path_new_model)))
    eval_path = outputs_dir + "/learning_curves/" + os.path.basename(path_new_model)
    print(f"\n\nSaving evaluation to {eval_path}.")
    df_eval = pd.DataFrame(eval)
    df_eval.to_csv(eval_path, index=False)


def learn_against_list_models_by_worst(list_models, path_new_model, n_it=60000, n_rounds=100):
    
    # If new model does not exist, create it as random model
    if not os.path.exists(path_new_model+".zip"):
        print(f"\n\nNew model {os.path.basename(path_new_model)} does not exist. Creating random model.\n")
        new_model = modelpath_to_model('random')
        new_model.save(path_new_model)

    # Initial evaluation of new model against all models
    df_eval, avrg_eval = play_list_score(list_of_models=list_models,
                                         path_adv_model=path_new_model,
                                         n_rounds=n_rounds)
    print(f"\n\nInitial evaluation average is at: {avrg_eval}, threshold at {EVAL_TRESHOLD}.")
    print(f"\n{df_eval}")

    # Keep training until new model average win ratio is higher than EAVAL_TRESHOLD
    while avrg_eval < EVAL_TRESHOLD:
        # Select worst model
        worst_model = df_eval[df_eval["new_model_wins"] == df_eval["new_model_wins"].min()]["base_model"].values[0]
        print(f"\n\nTraining against {worst_model}.")
        path_worst_model = os.path.join(os.path.dirname(path_new_model), worst_model)
        # Train new model against wosrt model
        _, _ = learn_once(path_base_model=path_worst_model,
                          path_new_model=path_new_model,
                          n_it=n_it, save=True)
        # Evaluate new model against all models
        df_eval, avrg_eval = play_list_score(list_of_models=list_models,
                                             path_adv_model=path_new_model,
                                             n_rounds=n_rounds)
        print(f"\n--> Evaluation average is at: {avrg_eval}, threshold at {EVAL_TRESHOLD}.")
        print(f"\n{df_eval}")

    # Save evaluation
    outputs_dir = os.path.dirname(os.path.dirname((path_new_model)))
    eval_path = outputs_dir + "/learning_curves/" + os.path.basename(path_new_model)
    print(f"\nTraining of model {os.path.basename(path_new_model)} COMPLETED.")
    df_eval.to_csv(eval_path, index=False)


def learn_against_list_models_and_new_ones_by_worst(list_of_models, 
                                                    path_new_model,
                                                    n_learning,
                                                    n_it,
                                                    n_rounds):
    """
    Function to train a model against a list of models by worst. 
    If you want ot learn from scratch, set list_of_models to [].
    NOTE: path_new_model must contain _i_ to be replaced by the iteration number ********
    """

    # Learn against each model in the list, plus the ones newly created in previous levels, including random
    for i in range(n_learning):
        if i == 0:
            aug_list_models = ["random"] + list_of_models
        else:
            aug_list_models = ["random"] + list_of_models + [path_new_model.replace("_i_", f"_{i}_") for i in range(i)]

        learn_against_list_models_by_worst(list_models=aug_list_models,
                                           path_new_model=path_new_model.replace("_i_", f"_{i}_"),
                                           n_it=n_it,
                                           n_rounds=n_rounds)
                                  

    


if __name__=="__main__":

    # learn_against_list_models(list_models=LIST_OF_MODELS,
    #                           path_new_model=PATH_MEW_MODEL,
    #                           n_it=N_IT,
    #                           n_rounds=N_ROUNDS)
    
    learn_against_list_models_and_new_ones_by_worst(list_of_models=LIST_OF_MODELS,
                                                    path_new_model=PATH_MEW_MODEL,
                                                    n_learning=N_LEARNING,
                                                    n_it=N_IT,
                                                    n_rounds=N_ROUNDS)
    

    
    
