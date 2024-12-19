import logging
import shutil
import subprocess
from pathlib import Path

from one.api import ONE
from one.remote import aws

_logger = logging.getLogger('ibllib')


def _run_command(command):
    """
    Run a shell command using subprocess.

    :param command: command to run
    :return: dictionary with keys: process, stdout, stderr
    """
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    info, error = process.communicate()
    return {
        'process': process,
        'stdout': info.decode(),
        'stderr': error.decode()}


def download_and_unzip_file_from_aws(
    directory, filename, one=None, target_path=None, overwrite=False
):
    """Download zipfile from AWS `resources/lightning_pose` bucket, unzip, return directory name.

    Parameters
    ----------
    directory : str
        name of directory under 'resources' in AWS
    filename : str
        name of zipped file to download from AWS, without '.zip' extension
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
        Path to the directory containing the unzipped file

    """

    # if there is a weight dir in the current path, use this one. Useful for Docker deployment
    local_data_dir = Path(filename).absolute()
    if local_data_dir.exists():
        _logger.warning(f'Using cached directory at {local_data_dir}')
        return local_data_dir

    one = one or ONE(base_url='https://alyx.internationalbrainlab.org')

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
