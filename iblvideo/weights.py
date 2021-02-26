"""Functions to handle DLC weights."""
import shutil
from pathlib import Path
from ibllib.io import params
from oneibl.webclient import http_download_file
from . import __version__


def download_weights_flatiron(version=__version__):
    """Download the DLC weights from FlatIron."""
    # Read one_params file
    par = params.read('one_params')
    weights_dir = 'resources/DLC'

    # Create target directory if it doesn't exist
    weights_path = Path(par.CACHE_DIR).joinpath(weights_dir)
    weights_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/{}/DLC_weights_v{}.zip'.format(par.HTTP_DATA_SERVER,
                                             weights_dir,
                                             version)
    file_name = http_download_file(url,
                                   cache_dir=weights_path,
                                   username=par.HTTP_DATA_SERVER_LOGIN,
                                   password=par.HTTP_DATA_SERVER_PWD)

    # unzip file
    shutil.unpack_archive(file_name, weights_path)

    return Path(file_name.split('.')[0])
