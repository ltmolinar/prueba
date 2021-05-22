
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import ScriptRunConfig

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml', _file_name='config.json')
    experiment = Experiment(workspace=ws, name='day1-experiment')
    config = ScriptRunConfig(source_directory='./src',
                             script='model.py',
                             compute_target='cpu-cluster')

    # set up pytorch environment
    env = Environment.from_conda_specification(
        name='remote-env',
        file_path='./.azureml/remote-env.yml'
    )
    config.run_config.environment = env

    run = experiment.submit(config)

    aml_url = run.get_portal_url()
    print(aml_url)