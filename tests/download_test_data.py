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
    """Download DLC test data from AWS, unzip it, and return file name.

    Parameters
    ----------
    version : str
        Version of the test data to download.
        Should be the same as the version in weights.download_weights
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

    directory = 'dlc'
    filename = f'dlc_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_me_test_data(
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
    tracker: str = 'dlc',
) -> Path | None:
    """Download motion energy test data from AWS, unzip it, and return file name.

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
    tracker : str
        'dlc' or 'lightning_pose'

    Returns
    -------
    pathlib.Path
        Path to the directory containing the test data

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
    """Download LP test data from AWS, unzip it, and return file name.

    Parameters
    ----------
    version : str
        Version of the test data to download.
        Should be the same as the version in weights.download_lp_models
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

    directory = 'lightning_pose'
    filename = f'lp_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_la_test_data(
    version: str = __la_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download LA test data from AWS, unzip it, and return file name.

    Parameters
    ----------
    version : str
        Version of the test data to download.
        Should be the same as the version in weights.download_lp_models
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

    directory = 'lightning_action'
    filename = f'la_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)
