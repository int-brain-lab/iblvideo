"""Functions to handle DLC weights."""
import shutil
from pathlib import Path
from datetime import date
from ibllib.io import params
from oneibl.webclient import http_download_file


def download_weights_flatiron(version_date=None):
    """Download the DLC weights from FlatIron."""
    # Read one_params file
    par = params.read('one_params')

    # Create target directory if it doesn't exist
    weights_path = Path(par.CACHE_DIR).joinpath('resources', 'DLC')
    weights_path.mkdir(exist_ok=True, parents=True)

    # Check if version date was given, if not use current day
    if version_date is None:
        version_date = date.today().strftime("%Y-%m-%d")
    # Here we still need a piece of code that compares the version date with
    # the available version dates and chooses the one closest before the given
    # date for download.

    # Construct URL and call download
    url = '{}/resources/DLC/DLC_weights_{}.zip'.format(par.HTTP_DATA_SERVER,
                                                       version_date)
    file_name = http_download_file(url,
                                   cache_dir=weights_path,
                                   username=par.HTTP_DATA_SERVER_LOGIN,
                                   password=par.HTTP_DATA_SERVER_PWD)

    # unzip file
    shutil.unpack_archive(file_name, weights_path)

    return weights_path
