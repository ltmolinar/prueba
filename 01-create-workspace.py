from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace

interactive_auth = InteractiveLoginAuthentication(tenant_id="7cb476fd-c725-4e5b-bb9b-c2639e1004c6")
ws = Workspace.create(name='azure-ml',
            subscription_id='cd12f72f-a4e5-4cfe-b776-0e78cf8c0394',
            resource_group='cloud-ml',
            create_resource_group=True,
            location='eastus2',
            auth=interactive_auth
            )
            
# write out the workspace details to a configuration file: .azureml/config.json
ws.write_config(path='.azureml')