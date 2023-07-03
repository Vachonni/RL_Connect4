import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication


# Try Azure Service Principal authentification.
# If fails, use interactive authentification
try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=os.environ['TENANT_ID'],
        service_principal_id=os.environ['CLIENT_ID'],
        service_principal_password=os.environ['CLIENT_SECRET'])
except:
    svc_pr = None


# Get Azure workspace
ws = Workspace.get(name=os.environ.get('WORKSPACE_NAME'), 
                   subscription_id=os.environ.get('SUBSCRIPTION_ID'), 
                   resource_group=os.environ.get('RESOURCE_GROUP'),
                   auth=svc_pr)


# Try to get existing environment. 
# If fails, create a new one and register it
try:
    env = Environment.get(workspace=ws, name='RLConnect4Package')
except:
    # Set an Azure environment with Dockerfile
    env = Environment.from_dockerfile(name='RLConnect4Package',
                                      dockerfile='Dockerfile')
    # Register the environment
    env.register(workspace=ws)

    

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
                        








 

