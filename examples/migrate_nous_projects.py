from dotenv import dotenv_values

from sc_api_tools import SCRESTClient

from sc_api_tools.nous import migrate_nous_chain, migrate_nous_project


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

    migrate_nous_project(
        rest_client=client,
        export_path='C:/Projects/SC/API/hat-segmentation.zip',
        project_name='hat seg',
        project_type='segmentation'
    )

    migrate_nous_chain(
        rest_client=client,
        export_path='C:/Projects/SC/API/hat-det-seg-chain.zip',
        task_types=['detection', 'segmentation'],
        labels_per_task=[['hard hat'], ['hard hat shape']],
        project_name='dummy_pipeline_project'
    )
