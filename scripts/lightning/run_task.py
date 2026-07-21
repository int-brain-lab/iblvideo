"""
Batch process multiple sessions by calling process_single_session.py for each EID.

Sessions will run sequentially on the current node:
    python run_task.py --eid_file eids.txt --list 0,2,5,8 --run_lp --run_post_lp --cameras left right body
    python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --cameras left right body

Each session will run on a new studio, concurrently:
    python run_task.py --eid_file eids.txt --list 0,2,5,8 --run_lp --run_post_lp --run_parallel --cameras left right body
    python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --run_parallel --cameras left right body
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROCESS_SESSION_SCRIPT = Path(__file__).parent / 'process_single_session.py'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Batch process multiple sessions')
    parser.add_argument(
        '--eid_file',
        type=str,
        required=True,
        help='Path to text file containing EIDs (one per line)'
    )
    parser.add_argument(
        '--list',
        type=str,
        required=True,
        help='Comma-separated list of indices (e.g., "0,2,5,8") or range (e.g., "0-14" inclusive)'
    )
    parser.add_argument('--run_lp', action='store_true', help='Run Lightning Pose')
    parser.add_argument('--run_post_lp', action='store_true', help='Run Post-LP')
    parser.add_argument('--run_parallel', action='store_true', help='Run in parallel using Jobs')
    parser.add_argument(
        '--cameras',
        nargs='+',
        choices=['left', 'right', 'body'],
        default=['left', 'right', 'body'],
        help='Cameras to process (one or more)',
    )

    args = parser.parse_args()

    return args


def parse_list_arg(list_str: str) -> list[int]:
    """Parse the --list argument.

    If it contains a comma, treat as comma-separated indices.
    If it contains a single dash, treat as a range (inclusive).

    Args:
        list_str: raw --list argument, e.g. '0,2,5' or '0-14'

    Returns:
        list of indices
    """
    if ',' in list_str:
        # comma-separated list
        indices = [int(idx.strip()) for idx in list_str.split(',')]
    elif '-' in list_str:
        # range format
        parts = list_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {list_str}. Expected format: 'start-end'")
        start, end = int(parts[0].strip()), int(parts[1].strip())
        indices = list(range(start, end + 1))  # +1 to make it inclusive
    else:
        # single index
        indices = [int(list_str.strip())]

    return indices


def load_eids(eid_file: str) -> list[str]:
    """Load EIDs from a text file."""
    eids = []
    with open(eid_file, 'r') as file:
        for line in file:
            eid = line.strip().replace('\n', '')
            if eid:  # skip empty lines
                eids.append(eid)
    return eids


def build_command(eid: str, run_lp: bool, run_post_lp: bool, cameras: list[str]) -> list[str]:
    """Build the command to call process_single_session.py."""
    cmd = [sys.executable, str(PROCESS_SESSION_SCRIPT), eid]

    if run_lp:
        cmd.append('--run_lp')
    if run_post_lp:
        cmd.append('--run_post_lp')
    if cameras:
        cmd.extend(['--cameras'] + cameras)

    return cmd


def main() -> None:
    args = parse_args()

    # load EIDs from file
    print(f'Loading EIDs from {args.eid_file}')
    all_eids = load_eids(args.eid_file)

    # parse the list argument
    indices = parse_list_arg(args.list)

    # select EIDs based on the provided indices
    eids = [all_eids[idx] for idx in indices]

    print(f'Processing {len(eids)} eids (indices: {args.list} -> {indices})')
    print(f'  Run LP: {args.run_lp}')
    print(f'  Run Post-LP: {args.run_post_lp}')
    print(f'  Cameras: {args.cameras}')
    print()

    # track errors
    errors = {}

    # import Lightning objects if running in parallel mode
    if args.run_parallel:
        from lightning_sdk import Studio, Machine, Job

        # use currently running studio
        studio = Studio()

        # use other studio
        # studio = Studio(name="iblvideo-inference", teamspace="video")
        # studio.start()

        # track jobs
        jobs = {}

    # process each EID
    for i, eid in enumerate(eids):
        print(f'=' * 80)
        print(f'Processing {i + 1}/{len(eids)}: {eid}')
        print(f'=' * 80)

        cmd = build_command(eid, args.run_lp, args.run_post_lp, args.cameras)

        if args.run_parallel:
            job_name = eid
            job = Job.run(
                command=' '.join(cmd),
                name=job_name,
                machine=Machine.T4_SMALL,
                studio=studio,
            )
            jobs[job_name] = job
            print(job.status)  # Pending, Running, Completed, Stopped, Failed
            time.sleep(1800)  # sleep to prevent simultaneous updates to AWS managed prefix list
        else:
            try:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            except subprocess.CalledProcessError as e:
                print(f'\nERROR processing {eid}')
                errors[eid] = f'Exit code: {e.returncode}'
            except Exception as e:
                print(f'\nERROR processing {eid}: {e}')
                errors[eid] = str(e)

    # manage error handling differently in parallel mode
    if args.run_parallel:
        # this only handles overall job failures, there is error-handling within the video pipeline
        # not picked up here
        while any([str(job.status) in ['Running', 'Pending'] for _, job in jobs.items()]):
            time.sleep(120)

        errors = {
            job_name: job.status for job_name, job in jobs.items() if str(job.status) == 'Failed'
        }

    # print summary
    print('=' * 80)
    print('SUMMARY')
    print('=' * 80)
    print(f'Total sessions: {len(eids)}')
    print(f'Successful: {len(eids) - len(errors)}')
    print(f'Errors: {len(errors)}')

    if errors:
        print('\nSessions with errors:')
        for eid, error in errors.items():
            print(f'  {eid}: {error}')


if __name__ == '__main__':
    main()
