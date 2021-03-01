"""Functions to handle DLC weights."""
import shutil
from pathlib import Path
from ibllib.io import params
from oneibl.webclient import http_download_file
from . import __version__


def download_weights(version=__version__):
    """Download the DLC weights associated with current version from FlatIron."""
    # Read one_params file
    par = params.read('one_params')
    weights_dir = Path('resources', 'DLC')

    # Create target directory if it doesn't exist
    weights_path = Path(par.CACHE_DIR).joinpath(weights_dir)
    weights_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/{}/DLC_weights_v{}.zip'.format(par.HTTP_DATA_SERVER, str(weights_dir), version)
    file_name = Path(http_download_file(url,
                                        cache_dir=weights_path,
                                        username=par.HTTP_DATA_SERVER_LOGIN,
                                        password=par.HTTP_DATA_SERVER_PWD))
    weights_dir = file_name.parent.joinpath(Path(file_name).stem)
    # we assume that user side, any change will be labeled by a version bump
    if not weights_dir.exists():
        shutil.unpack_archive(file_name, weights_path)  # unzip file
    return weights_dir
