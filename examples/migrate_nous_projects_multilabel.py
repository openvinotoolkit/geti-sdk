from dotenv import dotenv_values

from sc_api_tools import SCRESTClient

from sc_api_tools.nous import migrate_nous_project


if __name__ == '__main__':

    # Get credentials from .env file
    env_variables = dotenv_values(dotenv_path=".env")

    if not env_variables:
        raise ValueError(
            "Unable to load login details from .env file, please make sure the file "
            "exists at the root of the `examples` directory."
        )

    # Set up REST client with server address and login details
    client = SCRESTClient(
        host=env_variables.get("HOST"),
        username=env_variables.get("USERNAME"),
        password=env_variables.get("PASSWORD")
    )

    # For multilabel projects, it is required to specify the labels as a list of
    # dictionaries. Each dictionary holds the properties of one label. All labels
    # belonging to the same group are exclusive with each other. Labels from
    # different groups are non-exclusive
    labels = [
        {
            'name': 'person',
            'group': 'label group 1'
        },
        {
            'name': 'people',
            'group': 'label group 2'
        }
    ]

    migrate_nous_project(
        rest_client=client,
        export_path='C:/Projects/SC/API/multi-label.zip',
        project_name='Multi-label migration',
        project_type='classification',
        labels=labels
    )
