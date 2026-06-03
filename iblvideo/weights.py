"""Functions to handle model weights."""
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

    directory = 'dlc'
    filename = f'weights_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def download_lp_models(
    version: str = __lp_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
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

    directory = 'lightning_pose'
    filename = f'networks_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)


def download_la_models(
    version: str = __la_version__,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Downloads specific LA networks version from AWS, unzips it, and returns file name.

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

    directory = 'lightning_action'
    filename = f'networks_{version}'
    return download_and_unzip_file_from_aws(directory, filename, one, target_path, overwrite)
