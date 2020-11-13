from dateutil.parser import parse
from pathlib import Path
import re
import subprocess
import yaml


import ibllib.io


def create_flags(root_path, dry=False):
    """
    Create flag files for the training DLC process
    """
    ibllib.io.flags.create_dlc_flags(root_path.joinpath('dlc_training.flag'),
                                     clobber=True, dry=dry)


def order_glob_by_session_date(flag_files):
    """
    Given a list/generator of PurePaths below an ALF session folder, output a list of of PurePaths
    sorted by date in reverse order.
    :param flag_files: list/generator of PurePaths
    :return: list of PurePaths
    """
    flag_files = list(flag_files)

    def _fdate(fl):
        dat = [parse(fp)
               for fp in fl.parts if re.match(r'\d{4}-\d{2}-\d{2}', fp)]
        if dat:
            return dat[0]
        else:
            return parse('1999-12-12')

    t = [_fdate(fil) for fil in flag_files]
    return [f for _, f in sorted(zip(t, flag_files), reverse=True)]


def run_command(command):
    """
    Runs a shell command using subprocess.

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


def set_dlc_paths(path_dlc):
    """
    OMG! Hard-coded paths everywhere ! Replace hard-coded paths in the config.yaml file.
    """
    for yaml_file in path_dlc.rglob('config.yaml'):
        # read the yaml config file
        with open(yaml_file) as fid:
            yaml_data = yaml.safe_load(fid)
        # if the path is correct skip to next
        if Path(yaml_data['project_path']) == yaml_file.parent:
            continue
        # else read the whole file
        with open(yaml_file) as fid:
            yaml_raw = fid.read()
        # patch the offending line and rewrite properly
        with open(yaml_file, 'w+') as fid:
            fid.writelines(
                yaml_raw.replace(
                    yaml_data['project_path'], str(yaml_file.parent)))




