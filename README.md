# RL_Connect4

Possible to train with Google Colab, but models can't to transfered (serialization issus known by StableBaselines)

TODO: 
- Probably get rid of Kaggle and use prediction with derterministic=True (to avoid exploration when evaluating)
- Try vecEnv


## HP_tuning

Too muych variance in the evaluation to be working. Probably missing a predict with deterministic=True.

## Connection to Azure ML

Connection information to Worspace are in devcontainer.env

## Creation of Azure Ennvironement 'RLCondaEnvFromExisting'
<i> no success with Dockerfile  </i> :(

1- Start an Azure ML Computer Instance 

2- Connect to it through VSCode AzureML extension (right click)

3- Create a conda ennvironement `conda create --name RLCondaEnv python=3.10`

4- Activate it `source activate RLCondaEnv`

5- After insuring you are in the proper folder, with requirements fileInstall requirements `pip install -r requirements.txt`

6- Install local folder `pip install -e .`

7- Register it as AML Environment called "RLCondaEnvFromExisting". See script in AML_CreateENV.py

<i>NOTE: If you want to use this conda file to have the same environment from a new workspace, need to run `pip install -e .` </i>


<br><br><br>

#### Observations:
- LearnToStop implies forgetting old adversaries. Better than last, but less when compare to inital models
- 1000 can beat 5000. Is 1000 iterations enough?
- First models on a serie are easier to beat, then more complicated, closer to equal even if continue to train. Is this because model has not enough capacity?
- Rewards in Play are only 1 or -1 at the end. Is this the case while traiing or 1, -10 and 1/42?

#### Next steps:
- Keep learning from all (?) previous agents (to avoid catastrophic forgetting) (SEE OBSERVATIONS)
- Delete KaggleLearn (not matching ModelpathToAgent anymore)


