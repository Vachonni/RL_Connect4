import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig


# Get an azure workspace
ws = Workspace.get(name=os.environ.get('WORKSPACE_NAME'), 
                   subscription_id=os.environ.get('SUBSCRIPTION_ID'), 
                   resource_group=os.environ.get('RESOURCE_GROUP'))

# Set an Azure environment with Dockerfile
env = Environment.from_dockerfile(name='RLConnect4Package',
                                  dockerfile='Dockerfile')

# Create Azure ScriptRunConfig
script_run_config = ScriptRunConfig(source_directory='./run',
                                    script='KaggleLearnStop.py',
                                    environment=env)                                    

# Create a new experiment
experiment = Experiment(workspace=ws, name='kaggle-learn-stop')

# Submit the experiment
run = experiment.submit(config=script_run_config)

run.log('tryingLOG', "is it working?")

# Run the experiment
run.wait_for_completion(show_output=True)
                        








 

