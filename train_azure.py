import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.authentication import ServicePrincipalAuthentication


# Try Azure Service Principal authentification.
# If fails, use interactive authentification
try:
    svc_pr = ServicePrincipalAuthentication(
        tenant_id=os.environ['AZURE_TENANT_ID'],
        service_principal_id=os.environ['AZURE_CLIENT_ID'],
        service_principal_password=os.environ['AZURE_CLIENT_SECRET'])
except:
    svc_pr = None


# Get Azure workspace
ws = Workspace.get(name=os.environ.get('AML_WORKSPACE_NAME'), 
                   subscription_id=os.environ.get('AML_SUBSCRIPTION_ID'), 
                   resource_group=os.environ.get('AML_RESOURCE_GROUP'),
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

# Run the experiment
run.wait_for_completion(show_output=True)
                        








 

