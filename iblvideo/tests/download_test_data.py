import shutil
from pathlib import Path
from ibllib.io import params
from oneibl.webclient import http_download_file
from .. import __version__


def _download_test_data(version=__version__):
    """Download test data from FlatIron."""
    # Read one_params file
    par = params.read('one_params')
    data_dir = Path('resources', 'DLC', 'test_data')

    # Create target directory if it doesn't exist
    local_path = Path(par.CACHE_DIR).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/{}/v{}.zip'.format(par.HTTP_DATA_SERVER, str(data_dir), version)
    file_name = http_download_file(url,
                                   cache_dir=local_path,
                                   username=par.HTTP_DATA_SERVER_LOGIN,
                                   password=par.HTTP_DATA_SERVER_PWD)

    # unzip file
    shutil.unpack_archive(file_name, local_path)

    return Path(file_name[:-4])
