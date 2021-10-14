import logging
import shutil
from pathlib import Path

from one.api import ONE
from iblvideo import __version__

_logger = logging.getLogger('ibllib')


def _download_dlc_test_data(version=__version__, one=None):
    """Download test data from FlatIron."""
    # Read one_params file

    # if there is a test dir in the current path, use this one. Useful for Docker deployment
    local_test_dir = Path(f"dlc_test_data_v{'.'.join(version.split('.')[:-1])}").absolute()
    if local_test_dir.exists():
        _logger.warning(f'Using cached directory at {local_test_dir}')
        return local_test_dir

    # otherwise get it from the SDSC server
    one = one or ONE()
    data_dir = Path('integration', 'dlc', 'test_data')

    # Create target directory if it doesn't exist
    local_path = Path(ONE().cache_dir).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/dlc_test_data_v{}.zip'.format(str(data_dir), '.'.join(version.split('.')[:-1]))
    file_name, hash = one.alyx.download_file(url, cache_dir=local_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    _logger.info(f"Downloaded test data hash: {hash}, {file_name}")
    # unzip file
    test_dir = file_name.parent.joinpath(file_name.stem)
    if not test_dir.exists():
        shutil.unpack_archive(str(file_name), local_path)

    return Path(test_dir)


def _download_me_test_data(one=None):
    """Download test data from FlatIron."""
    # eid: cde63527-7f5a-4cc3-8ac2-215d82e7da26
    # if there is a test dir in the current path, use this one. Useful for Docker deployment
    local_test_dir = Path("me_test_data").absolute()
    if local_test_dir.exists():
        _logger.warning(f'Using cached directory at {local_test_dir}')
        return local_test_dir

    # Read one_params file
    one = one or ONE()
    data_dir = Path('integration', 'dlc', 'test_data')

    # Create target directory if it doesn't exist
    local_path = Path(ONE().cache_dir).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/me_test_data.zip'.format(str(data_dir))
    file_name, hash = one.alyx.download_file(url, cache_dir=local_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    # unzip file
    test_dir = file_name.parent.joinpath(file_name.stem)
    if not test_dir.exists():
        shutil.unpack_archive(file_name, local_path)

    return Path(test_dir)
