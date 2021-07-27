import subprocess
import logging

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


def _check_nvidia_status():
    process = subprocess.Popen('nvidia-smi', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, executable="/bin/bash")
    info, error = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"Nvida drivers not ready. \n {error.decode('utf-8')}")
    _logger.info("nvidia-smi command successful")
