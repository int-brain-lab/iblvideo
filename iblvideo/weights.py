"""Functions to handle model weights."""
from __future__ import annotations

import logging
from pathlib import Path

from one.api import ONE

from iblvideo import __dlc_version__, __la_version__, __lp_version__
from iblvideo.utils import download_and_unzip_file_from_aws

_logger = logging.getLogger('ibllib')


def download_weights(
    version: str = __dlc_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download the DLC weights associated with current version from AWS.

    Args:
        version: version of the network models to download
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download models to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the network models, or None if download failed
    """

    directory = 'dlc'
    filename = f'weights_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def download_lp_models(
    version: str = __lp_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download Lightning Pose networks from AWS, unzip, and return directory path.

    Args:
        version: version of the network models to download
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download models to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the network models, or None if download failed
    """

    directory = 'lightning_pose'
    filename = f'networks_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def download_la_models(
    version: str = __la_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download Lightning Action networks from AWS, unzip, and return directory path.

    Args:
        version: version of the network models to download
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download models to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the network models, or None if download failed
    """

    directory = 'lightning_action'
    filename = f'networks_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)
