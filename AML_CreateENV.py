from azureml.core import Environment, Workspace

# Load the workspace
ws = Workspace.from_config()

# Create a new environment
myenv = Environment.from_existing_conda_environment(name="RLCondaEnvFromExisting",
                                                    conda_environment_name="RLCondaEnv")


# Register the environment
myenv.register(workspace=ws)