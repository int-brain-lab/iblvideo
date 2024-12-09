"""Functions to handle DLC weights."""
import logging
import shutil
from pathlib import Path
import packaging
from packaging.version import InvalidVersion

from one.api import ONE
from one.remote import aws
from iblvideo import __version__

_logger = logging.getLogger('ibllib')


def download_weights(version=__version__, one=None):
    """Download the DLC weights associated with current version from FlatIron."""

    try:
        vers = packaging.version.parse(version)
        weight_vers = f'{vers.major}.{vers.minor}'
    except InvalidVersion:
        weight_vers = '.'.join(version.split('_')[-1].split('.')[:-1])

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_weight_dir = Path(f"weights_v{weight_vers}").absolute()
    if local_weight_dir.exists():
        _logger.warning(f'Using cached directory at {local_weight_dir}')
        return local_weight_dir

    one = one or ONE(base_url='https://alyx.internationalbrainlab.org')
    weights_dir = Path('resources', 'dlc')

    # Create target directory if it doesn't exist
    weights_path = Path(ONE().cache_dir).joinpath(weights_dir)
    weights_path.mkdir(exist_ok=True, parents=True)
    # Construct URL and call download
    # Weights versions are synchronized with minor versions of iblvideo
    # Therefore they are named only by major.minor excluding the patch
    url = '{}/weights_v{}.zip'.format(str(weights_dir), weight_vers)
    file_name, hash = one.alyx.download_file(url, target_dir=weights_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    _logger.info(f"Downloaded weights: {hash}, {file_name}")
    weights_dir = file_name.parent.joinpath(file_name.stem)
    # we assume that user side, any change will be labeled by a version bump
    if not weights_dir.exists():
        shutil.unpack_archive(str(file_name), weights_path)  # unzip file
    return weights_dir


def download_lit_model(version='v1.0', one=None, target_path=None, overwrite=False):
    """
    Downloads a specific network from AWS and returns file name.

    Parameters
    ----------
    version : str
        Version of the network models to download.
    one : ONE
        An instance of ONE to use for downloading. Defaults is None, in which case a new instance pointing to the
        internal IBL database is instantiated.
    target_path : Path
        Path to download the network models to. If None, the default cache directory is used. Defaults to None.
    overwrite : bool
        If True, will re-download networks even if they exist locally and file sizes match. Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the networks models

    """

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_network_dir = Path(f"networks_{version}").absolute()
    if local_network_dir.exists():
        _logger.warning(f'Using cached directory at {local_network_dir}')
        return local_network_dir

    one = one or ONE(base_url='https://alyx.internationalbrainlab.org')

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('resources', 'lightning_pose')
        target_path.mkdir(exist_ok=True, parents=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    full_path = target_path.joinpath(f'networks_{version}.zip')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(f"resources/lightning_pose/networks_{version}.zip", full_path, s3=s3,
                         bucket_name=bucket_name, overwrite=overwrite)

    if not full_path.exists():
        print(f'Downloading of networks_{version} failed.')
        return

    # Unpack
    unzipped = target_path.joinpath(f'networks_{version}')
    if not unzipped.exists() or overwrite:
        shutil.unpack_archive(str(full_path), target_path)  # unzip file

    if not unzipped.exists():
        print(f'Unzipping of {full_path} failed.')
        return

    return unzipped
