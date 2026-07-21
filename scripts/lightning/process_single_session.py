"""
Process a single session with Lightning Pose and Post-LP.

Usage:
    python process_single_session.py <eid> --run_lp --run_post_lp --cameras left right body
"""
import argparse
import sys
import time
from pathlib import Path
import traceback

from one.api import ONE
from ibllib.pipes.video_tasks import LightningPose, PostLP
from iblaws.compute import manage_firewall_access

one = ONE()


def parse_args():
    parser = argparse.ArgumentParser(description='Process a single session with Lightning Pose')
    parser.add_argument('eid', type=str, help='Experiment ID to process')
    parser.add_argument('--run_lp', action='store_true', help='Run Lightning Pose')
    parser.add_argument('--worker_id', type=int, help='Worker # (int)', default=-1)
    parser.add_argument('--dry', action='store_true',
                        help='Dry run - will just connect to ONE and exit', default=False)
    parser.add_argument('--run_post_lp', action='store_true', help='Run Post-LP')
    parser.add_argument(
        '--cameras',
        nargs='+',
        choices=['left', 'right', 'body'],
        default=['left', 'right', 'body'],
        help='Cameras to process (one or more)',
    )
    return parser.parse_args()


def process_lightning_pose(eid, cameras, session_path):
    """Process Lightning Pose for given cameras."""
    error_task = {}

    for camera in cameras:
        try:
            print(f'{camera} camera')
            t_beg = time.time()
            mp4_file = session_path.joinpath(f'raw_video_data/_iblrig_{camera}Camera.raw.mp4')
            task = LightningPose(session_path, cameras=[camera], one=one, location='ec2')
            task.force = True
            task.lpenv = Path('/home/zeus/miniconda3/envs/cloudspace/bin/activate')
            status = task.run(overwrite=False)
            t_end = time.time()
            if status == 0:
                task.register_datasets(force=True)
            else:
                error_task[camera] = 'task.status=-1'
            print(f'Processing time: {((t_end - t_beg) / 3600):.2f} hours\n')
        except Exception as err:
            error_task[camera] = traceback.format_exc(err)
        finally:
            task.cleanUp()
            mp4_file.unlink(missing_ok=True)

    return error_task


def process_post_lp(eid, cameras, session_path):
    """Process Post-LP."""
    try:
        task = PostLP(session_path, cameras=cameras, one=one, location='ec2')
        task.force = True
        status = task.run(overwrite=True, plot_qc=True)
        if status == 0:
            task.register_datasets(force=True)
            return None
        else:
            return 'task.status=-1'
    except Exception as err:
        return err
    finally:
        task.cleanUp()


# @manage_firewall_access()
def main(args, **kwargs):
    print(f'ONE Version {ONE.version}')
    print(f'Processing EID: {args.eid}')
    print(f'  Cameras: {args.cameras}')
    print(f'  Run LP: {args.run_lp}')
    print(f'  Run Post-LP: {args.run_post_lp}')

    try:
        session_path = one.eid2path(args.eid)
    except Exception as err:
        print(f'ERROR: Could not get session path for {args.eid}: {err}')
        sys.exit(1)

    error_task = {}
    if args.run_lp and not args.dry:
        print(f'\nRunning Lightning Pose...')
        error_task = process_lightning_pose(args.eid, args.cameras, session_path)

        if error_task:
            print(f'Lightning Pose errors:')
            for camera, error in error_task.items():
                print(f'  {camera}: {error}')
        else:
            print('Lightning Pose completed successfully')

    # Run Post-LP
    error_post = None
    if args.run_post_lp and not args.dry:
        print(f'\nRunning Post-LP...')
        error_post = process_post_lp(args.eid, args.cameras, session_path)

        if error_post:
            print(f'Post-LP error: {error_post}')
        else:
            print('Post-LP completed successfully')

    if error_task is None and not error_post:
        print(f'\nFinished processing {args.eid}')
    # else:
    #     sys.exit(1)


if __name__ == '__main__':
    args = parse_args()
    worker_id = args.worker_id if args.worker_id >= 0 else args.eid[:6]
    main(args, worker_id=worker_id)
