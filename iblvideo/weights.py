"""Functions to handle DLC weights."""
import logging
import shutil
from pathlib import Path

from one.api import ONE
from iblvideo import __version__

_logger = logging.getLogger('ibllib')


def download_weights(version=__version__, one=None):
    """Download the DLC weights associated with current version from FlatIron."""

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_weight_dir = Path(f"weights_v{'.'.join(version.split('_')[-1].split('.')[:-1])}").absolute()
    if local_weight_dir.exists():
        _logger.warning(f'Using cached directory at {local_weight_dir}')
        return local_weight_dir

    one = one or ONE()
    weights_dir = Path('resources', 'dlc')

    # Create target directory if it doesn't exist
    weights_path = Path(ONE().cache_dir).joinpath(weights_dir)
    weights_path.mkdir(exist_ok=True, parents=True)
    f"weights_v{'.'.join(version.split('.')[:-1])}"
    # Construct URL and call download
    # Weights versions are synchronized with minor versions of iblvideo
    # Therefore they are named only by major.minor excluding the patch
    url = '{}/weights_v{}.zip'.format(str(weights_dir), '.'.join(version.split('.')[:-1]))
    file_name, hash = one.alyx.download_file(url, cache_dir=weights_path, return_md5=True, silent=True)
    file_name = Path(file_name)
    _logger.info(f"Downloaded weights: {hash}, {file_name}")
    weights_dir = file_name.parent.joinpath(file_name.stem)
    # we assume that user side, any change will be labeled by a version bump
    if not weights_dir.exists():
        shutil.unpack_archive(str(file_name), weights_path)  # unzip file
    return weights_dir
