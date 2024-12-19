import logging
import shutil
from pathlib import Path

from one.api import ONE
from iblvideo import __dlc_version__, __lp_version__
from iblvideo.utils import download_and_unzip_file_from_aws

_logger = logging.getLogger('ibllib')


def _download_dlc_test_data(version=__dlc_version__, one=None, target_path=None, overwrite=False):
    """Download DLC test data from AWS, unzip it, and return file name.

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

    directory = 'dlc'
    filename = f'dlc_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_me_test_data(one=None, target_path=None, overwrite=False):
    """Download motion energy test data from AWS, unzip it, and returns file name.

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

    directory = 'lightning_pose'
    filename = 'me_test_data'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def _download_lp_test_data(version=__lp_version__, one=None, target_path=None, overwrite=False):
    """Download LP test data from AWS, unzip it, and return file name.

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

    directory = 'lightning_pose'
    filename = f'lp_test_data_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)
