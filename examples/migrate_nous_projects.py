from sc_api_tools import SCRESTClient

from sc_api_tools.nous import migrate_nous_chain, migrate_nous_project


if __name__ == '__main__':

    client = SCRESTClient(
        host="https://10.211.120.60",
        username="a@a.com",
        password="Password1"
    )

    migrate_nous_project(
        rest_client=client,
        export_path='C:/Projects/SC/API/hat-segmentation.zip',
        project_name='hat seg',
        project_type='segmentation'
    )

    migrate_nous_chain(
        rest_client=client,
        export_path='C:/Projects/SC/API/hat-segmentation.zip',
        task_types=['detection', 'segmentation'],
        labels_per_task=[['hard hat'], ['hard hat shape']],
        project_name='dummy_pipeline_project'
    )
