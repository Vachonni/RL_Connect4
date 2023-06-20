import random
import numpy as np

from src.KaggleTest import ConnectFourGym, PPO, policy_kwargs


def modelpath_to_model(model_path, adv_agent='random'):

    # Set environment to reply as adversary agent
    adv_env = ConnectFourGym(agent2=adv_agent)

    # Get model
    if model_path == 'random':
        model = PPO("CnnPolicy", adv_env, policy_kwargs=policy_kwargs, verbose=0)
    else:
        model = PPO.load(model_path, env=adv_env, verbose=0)
    
    return model



def model_to_agent(model):

    if model == 'random':
        return 'random'

    # Build agent from model
    def agent(obs, config):
        # Use the best model to select a column
        col, _ = model.predict(np.array(obs['board']).reshape(1, 6,7))
        # Check if selected column is valid
        is_valid = (obs['board'][int(col)] == 0)
        # If not valid, select random move. 
        if is_valid:
            return int(col)
        else:
            return random.choice([col for col in range(config.columns) if obs.board[int(col)] == 0])
        
    return agent


def modelpath_to_agent(model_path, adv_agent='random'):

    model = modelpath_to_model(model_path, adv_agent)
    agent = model_to_agent(model)
    
    return agent
