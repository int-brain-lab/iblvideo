"""Utility functions for downloading and unpacking files from AWS."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from one.api import ONE
from one.remote import aws

_logger = logging.getLogger('ibllib')


def download_and_unzip_file_from_aws(
    directory: str,
    filename: str,
    one: ONE | None = None,
    target_path: Path | None = None,
    overwrite: bool = False,
) -> Path | None:
    """Download a zipfile from AWS, unzip it, and return the directory path.

    Args:
        directory: name of directory under 'resources' in AWS
        filename: name of zipped file to download, without the '.zip' extension
        one: an instance of ONE to use for downloading; if None a new instance pointing to
            the internal IBL database is instantiated
        target_path: path to download data to; if None the default cache directory is used
        overwrite: if True re-download even if files exist locally and sizes match

    Returns:
        path to the directory containing the unzipped file, or None if download failed
    """

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_data_dir = Path(filename).absolute()
    if local_data_dir.exists():
        _logger.warning(f'Using cached directory at {local_data_dir}')
        return local_data_dir

    one = one or ONE(base_url='https://openalyx.internationalbrainlab.org')

    if target_path is None:
        target_path = Path(one.cache_dir).joinpath('resources', directory)
        target_path.mkdir(exist_ok=True, parents=True)
    else:
        assert target_path.exists(), 'The target_path you passed does not exist.'

    full_path = target_path.joinpath(f'{filename}.zip')
    s3, bucket_name = aws.get_s3_from_alyx(alyx=one.alyx)
    aws.s3_download_file(
        f'resources/{directory}/{filename}.zip', full_path, s3=s3,
        bucket_name=bucket_name, overwrite=overwrite,
    )

    if not full_path.exists():
        print(f'Downloading of {filename} failed.')
        return

    # Unpack
    unzipped = target_path.joinpath(filename)
    if not unzipped.exists() or overwrite:
        shutil.unpack_archive(str(full_path), target_path)  # unzip file

    if not unzipped.exists():
        print(f'Unzipping of {full_path} failed.')
        return

    return unzipped
