from azureml.core import Workspace
from azureml.core import Model

if __name__ == "__main__":
    ws = Workspace.from_config(path='./.azureml',_file_name='config.json')

    model = Model.register(model_name='model',
                           tags={'area': 'trabajo_cluod'},
                           model_path='outputs/model.pkl',
                           workspace = ws)
    print(model.name, model.id, model.version, sep='\t')