"""Functions to handle DLC weights."""
import logging
import shutil
from pathlib import Path
import packaging
from packaging.version import InvalidVersion

from one.api import ONE
from iblvideo import __version__
from iblvideo.utils import download_and_unzip_file_from_aws

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


def download_lit_model(version='v2.1', one=None, target_path=None, overwrite=False):
    """Downloads specific LP networks version from AWS, unzips it, and returns file name.

    Parameters
    ----------
    version : str
        Version of the network models to download.
    one : ONE
        An instance of ONE to use for downloading.
        Defaults is None, in which case a new instance pointing to the internal IBL database is
        instantiated.
    target_path : Path
        Path to download the network models to. If None, the default cache directory is used.
        Defaults to None.
    overwrite : bool
        If True, will re-download networks even if they exist locally and file sizes match.
        Defaults to False.

    Returns
    -------
    pathlib.Path
        Path to the directory containing the networks models

    """

    filename = f'networks_{version}'
    return download_and_unzip_file_from_aws(filename, one, target_path, overwrite)
