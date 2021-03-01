from distributed import LocalCluster, Client, Nanny


def create_cpu_gpu_cluster(cpu_lim=10e9, gpu_lim=50e9):
    """
    Create a Dask cluster with 1 CPU and 1 GPU worker.

    :param cpu_lim: Max total CPU to be used (all cores will be used by ffmpeg)
    :param gpu_lim: Max total GPU to be used
    :return cluster, client: dask.LocalCluster and dask.Client running this cluster

    """
    # TODO: find gpu and cpu limit from machine
    # Make a local cluster without workers
    cluster = LocalCluster(n_workers=0, name='DLC')

    # Define new workers
    cpu_worker = {'CPU': {'cls': Nanny,
                          'options': {'nthreads': 1,
                                      'services': {},
                                      'dashboard_address': None,
                                      'dashboard': False,
                                      'interface': None,
                                      'protocol': 'tcp://',
                                      'silence_logs': 30,
                                      'memory_limit': cpu_lim}}}

    gpu_worker = {'GPU': {'cls': Nanny,
                          'options': {'nthreads': 1,
                                      'services': {},
                                      'dashboard_address': None,
                                      'dashboard': False,
                                      'interface': None,
                                      'protocol': 'tcp://',
                                      'silence_logs': 30,
                                      'memory_limit': gpu_lim}}}

    # Add the worker specs to the cluster
    cluster.worker_spec.update(cpu_worker)
    cluster.worker_spec.update(gpu_worker)

    # Scale the cluster to inlcude the workers
    cluster.scale(2)
    client = Client(cluster)

    return cluster, client
