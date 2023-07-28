import os

from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Datastore
from azureml.core.compute import ComputeTarget
from azureml.core.authentication import ServicePrincipalAuthentication


# # Try Azure Service Principal authentification.
# # If fails, use interactive authentification
# try:
#     svc_pr = ServicePrincipalAuthentication(
#         tenant_id=os.environ['AZURE_TENANT_ID'],
#         service_principal_id=os.environ['AZURE_CLIENT_ID'],
#         service_principal_password=os.environ['AZURE_CLIENT_SECRET'])
# except:
#     svc_pr = None


# # Get Azure workspace
# ws = Workspace.get(name=os.environ.get('AML_WORKSPACE_NAME'), 
#                    subscription_id=os.environ.get('AML_SUBSCRIPTION_ID'), 
#                    resource_group=os.environ.get('AML_RESOURCE_GROUP'),
#                    auth=svc_pr)

# # Get compute target
# compute_target = ComputeTarget(workspace=ws, name='SmallClusterCompute')

# # Move code to compute target
# # Get the default datastore
# datastore = ws.get_default_datastore()
# # Upload the code to the datastore
# datastore.upload(src_dir='../RL_Connect4/', target_path='RL_Connect4')

ws = Workspace.from_config()

# # Get Azure Environment
env = Environment.get(workspace=ws, name='RLCondaEnvFromExisting')

# Create Azure ScriptRunConfig
script_run_config = ScriptRunConfig(source_directory='./run',
                                    script='KaggleLearnStop.py',
                                    environment=env,
                                    compute_target='SmallClusterCompute')                                    

# Create a new experiment
experiment = Experiment(workspace=ws, name='kaggle-learn-stop')

# Submit the experiment
run = experiment.submit(config=script_run_config)

# Run the experiment
run.wait_for_completion(show_output=True)
                        








 

