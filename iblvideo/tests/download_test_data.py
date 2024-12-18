import logging
import shutil
from pathlib import Path

from one.api import ONE
from iblvideo import __version__
from iblvideo.utils import download_and_unzip_file_from_aws

_logger = logging.getLogger('ibllib')


def _download_dlc_test_data(version=__version__, one=None):
    """Download test data from FlatIron."""
    # Read one_params file

    # if there is a test dir in the current path, use this one. Useful for Docker deployment
    local_test_dir = Path(f"dlc_test_data_v{'.'.join(version.split('_')[-1].split('.')[:-1])}").absolute()
    if local_test_dir.exists():
        _logger.warning(f'Using cached directory at {local_test_dir}')
        return local_test_dir

    # otherwise get it from the SDSC server
    one = one or ONE()
    data_dir = Path('resources', 'dlc')

    # Create target directory if it doesn't exist
    local_path = Path(ONE().cache_dir).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/dlc_test_data_v{}.zip'.format(str(data_dir), '.'.join(version.split('_')[-1].split('.')[:-1]))
    file_name, hash = one.alyx.download_file(url, target_dir=local_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    _logger.info(f"Downloaded test data hash: {hash}, {file_name}")
    # unzip file
    test_dir = file_name.parent.joinpath(file_name.stem)
    if not test_dir.exists():
        shutil.unpack_archive(str(file_name), local_path)

    return Path(test_dir)


def _download_me_test_data(one=None, target_path=None, overwrite=False):
    """Download test data from AWS, unzips it, and returns file name.

    eid: cde63527-7f5a-4cc3-8ac2-215d82e7da26

    Parameters
    ----------
    one : ONE
        An instance of ONE to use for downloading.
        Defaults is None, in which case a new instance pointing to the internal IBL database is
        instantiated.
    target_path : Path
        Path to download test data to. If None, the default cache directory is used.
        Defaults to None.
    overwrite : bool
        If True, will re-download test data even if they exist locally and file sizes match.
        Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the test data

    """

    filename = 'me_test_data'
    return download_and_unzip_file_from_aws(filename, one, target_path, overwrite)


def _download_lp_test_data(version='v2.1', one=None, target_path=None, overwrite=False):
    """Downloads test data from AWS, unzips it, and returns file name.

    Parameters
    ----------
    version : str
        Version of the test data to download.
        Should be the same as the version in weights.download_lit_model
    one : ONE
        An instance of ONE to use for downloading.
        Defaults is None, in which case a new instance pointing to the internal IBL database is
        instantiated.
    target_path : Path
        Path to download test data to. If None, the default cache directory is used.
        Defaults to None.
    overwrite : bool
        If True, will re-download test data even if they exist locally and file sizes match.
        Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the test data

    """

    filename = f'lp_test_data_{version}'
    return download_and_unzip_file_from_aws(filename, one, target_path, overwrite)
