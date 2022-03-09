from sc_api_tools import SCRESTClient

from sc_api_tools.nous import migrate_nous_chain, migrate_nous_project


if __name__ == '__main__':

    client = SCRESTClient(
        host="https://10.211.120.60",
        username="a@a.com",
        password="Password1"
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
