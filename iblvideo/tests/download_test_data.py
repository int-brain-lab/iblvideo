import logging
import shutil
from pathlib import Path

from one.api import ONE
from one.remote import aws
from iblvideo import __version__

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
    data_dir = Path('resources', 'dlc')

    # Create target directory if it doesn't exist
    local_path = Path(ONE().cache_dir).joinpath(data_dir)
    local_path.mkdir(exist_ok=True, parents=True)

    # Construct URL and call download
    url = '{}/me_test_data.zip'.format(str(data_dir))
    file_name, hash = one.alyx.download_file(url, target_dir=local_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    # unzip file
    test_dir = file_name.parent.joinpath(file_name.stem)
    if not test_dir.exists():
        shutil.unpack_archive(file_name, local_path)

    return Path(test_dir)


def _download_lp_test_data(version='v1.0', one=None, target_path=None, overwrite=False):
    """
    Downloads test data from AWS and unzips it

    Parameters
    ----------
    version : str
        Version of the test data to download. Should be the same as the version in weights.download_lit_model
    one : ONE
        An instance of ONE to use for downloading. Defaults is None, in which case a new instance pointing to the
        internal IBL database is instantiated.
    target_path : Path
        Path to download test data to. If None, the default cache directory is used. Defaults to None.
    overwrite : bool
        If True, will re-download test data even if they exist locally and file sizes match. Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the test data
    """

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_data_dir = Path(f"lp_test_data_{version}").absolute()
    if local_data_dir.exists():
        _logger.warning(f'Using cached directory at {local_data_dir}')
        return local_data_dir

    one = one or ONE(base_url='https://alyx.internationalbrainlab.org')

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('resources', 'lightning_pose')
        target_path.mkdir(exist_ok=True, parents=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    full_path = target_path.joinpath(f'lp_test_data_{version}.zip')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(f"resources/lightning_pose/lp_test_data_{version}.zip", full_path, s3=s3,
                         bucket_name=bucket_name, overwrite=overwrite)

    if not full_path.exists():
        print(f'Downloading of lp_test_data_{version} failed.')
        return

    # Unpack
    unzipped = target_path.joinpath(f'lp_test_data_{version}')
    if not unzipped.exists() or overwrite:
        shutil.unpack_archive(str(full_path), target_path)  # unzip file

    if not unzipped.exists():
        print(f'Unzipping of {full_path} failed.')
        return

    return unzipped
