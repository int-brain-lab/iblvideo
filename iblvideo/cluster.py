from distributed import LocalCluster, Client, Nanny


def create_cpu_gpu_cluster(cpu_lim=10e9, gpu_lim=50e9):
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
