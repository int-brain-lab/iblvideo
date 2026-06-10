"""Functions to download test data from AWS for the iblvideo test suite."""

from __future__ import annotations

import logging
from pathlib import Path

from one.api import ONE

from iblvideo import __dlc_version__, __la_version__, __lp_version__
from iblvideo.utils import download_and_unzip_file_from_aws

_logger = logging.getLogger('ibllib')


def _download_dlc_test_data(
    version: str = __dlc_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download DLC test data from AWS, unzip it, and return the directory path.

    Args:
        version: version of the test data to download; should match weights.download_weights
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download test data to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the test data, or None if download failed
    """

    directory = 'dlc'
    filename = f'dlc_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_me_test_data(
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
    tracker: str = 'dlc',
) -> Path | None:
    """Download motion energy test data from AWS, unzip it, and return the directory path.

    Args:
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download test data to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match
        tracker: pose tracker used for the test data; 'dlc' or 'lightning_pose'

    Returns:
        path to the directory containing the test data, or None if download failed
    """

    directory = tracker
    filename = 'me_test_data'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_lp_test_data(
    version: str = __lp_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download Lightning Pose test data from AWS, unzip it, and return the directory path.

    Args:
        version: version of the test data to download; should match weights.download_lp_models
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download test data to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the test data, or None if download failed
    """

    directory = 'lightning_pose'
    filename = f'lp_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_la_test_data(
    version: str = __la_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download Lightning Action test data from AWS, unzip it, and return the directory path.

    Args:
        version: version of the test data to download; should match weights.download_la_models
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download test data to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the test data, or None if download failed
    """

    directory = 'lightning_action'
    filename = f'la_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)
