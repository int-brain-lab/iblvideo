import shutil
from pathlib import Path
from ibllib.io import params
from oneibl.webclient import http_download_file
from iblvideo import __version__


def _download_dlc_test_data(version=__version__,):
    """Download test data from FlatIron."""
    # Read one_params file
    par = params.read('one_params')
    data_dir = Path('integration', 'dlc', 'test_data')

    # Create target directory if it doesn't exist
    local_path = Path(par.CACHE_DIR).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/{}/dlc_test_data_v{}.zip'.format(par.HTTP_DATA_SERVER, str(data_dir),
                                               '.'.join(version.split('.')[:-1]))
    file_name = Path(http_download_file(url,
                                        cache_dir=local_path,
                                        username=par.HTTP_DATA_SERVER_LOGIN,
                                        password=par.HTTP_DATA_SERVER_PWD))

    # unzip file
    test_dir = file_name.parent.joinpath(Path(file_name).stem)
    if not test_dir.exists():
        shutil.unpack_archive(file_name, local_path)

    return Path(test_dir)


def _download_me_test_data():
    """Download test data from FlatIron."""
    # eid: cde63527-7f5a-4cc3-8ac2-215d82e7da26
    # Read one_params file
    par = params.read('one_params')
    data_dir = Path('integration', 'dlc', 'test_data')

    # Create target directory if it doesn't exist
    local_path = Path(par.CACHE_DIR).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/{}/me_test_data.zip'.format(par.HTTP_DATA_SERVER, str(data_dir))
    file_name = Path(http_download_file(url,
                                        cache_dir=local_path,
                                        username=par.HTTP_DATA_SERVER_LOGIN,
                                        password=par.HTTP_DATA_SERVER_PWD))

    # unzip file
    test_dir = file_name.parent.joinpath(Path(file_name).stem)
    if not test_dir.exists():
        shutil.unpack_archive(file_name, local_path)

    return Path(test_dir)
