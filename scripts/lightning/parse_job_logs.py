"""
Parse logs of Lightning Jobs for specific success/transfer strings.

Usage:
    python parse_job_logs.py --eid_file eids.txt --list 0,2,5,8 --lp --lp_qc --cameras left right body
    python parse_job_logs.py --eid_file eids.txt --list 0-14 --lp --cameras left
"""

import argparse

from lightning_sdk import Job


# ── string definitions ────────────────────────────────────────────────────────

LP_SUCCESS_STRING = 'Lightning Pose completed successfully'
LP_QC_SUCCESS_STRING = 'Post-LP completed successfully'

# per-camera transfer strings for --lp
LP_TRANSFER_PATTERNS = {
    'left': [
        'Transferring file */Subjects/*/*/*/alf/_ibl_leftCamera.lightningPose.pqt',
        'Transferring file */Subjects/*/*/*/alf/leftCamera.ROIMotionEnergy.npy',
        'Transferring file */Subjects/*/*/*/alf/leftROIMotionEnergy.position.npy',
    ],
    'right': [
        'Transferring file */Subjects/*/*/*/alf/_ibl_rightCamera.lightningPose.pqt',
        'Transferring file */Subjects/*/*/*/alf/rightCamera.ROIMotionEnergy.npy',
        'Transferring file */Subjects/*/*/*/alf/rightROIMotionEnergy.position.npy',
    ],
    'body': [
        'Transferring file */Subjects/*/*/*/alf/_ibl_bodyCamera.lightningPose.pqt',
        'Transferring file */Subjects/*/*/*/alf/bodyCamera.ROIMotionEnergy.npy',
        'Transferring file */Subjects/*/*/*/alf/bodyROIMotionEnergy.position.npy',
    ],
}

# per-camera transfer strings for --lp_qc
LP_QC_TRANSFER_PATTERNS = {
    'left':  ['Transferring file */Subjects/*/*/*/alf/_ibl_leftCamera.features.pqt'],
    'right': ['Transferring file */Subjects/*/*/*/alf/_ibl_rightCamera.features.pqt'],
    'body':  ['Transferring file */Subjects/*/*/*/alf/_ibl_bodyCamera.features.pqt'],
}

# shared licks transfer string (checked once, not per-camera)
LP_QC_LICKS_PATTERN = 'Transferring file */Subjects/*/*/*/alf/licks.times.npy'


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse Lightning Job logs for session success strings')
    parser.add_argument('--eid_file', type=str, required=True,
                        help='Path to text file containing EIDs (one per line)')
    parser.add_argument('--list', type=str, required=True,
                        help='Comma-separated indices ("0,2,5") or inclusive range ("0-14")')
    parser.add_argument('--lp', action='store_true', help='Check Lightning Pose strings')
    parser.add_argument('--lp_qc', action='store_true', help='Check Post-LP / QC strings')
    parser.add_argument(
        '--cameras', nargs='+', choices=['left', 'right', 'body'],
        default=['left', 'right', 'body'],
        help='Cameras to check (one or more)',
    )
    return parser.parse_args()


def parse_list_arg(list_str: str) -> list[int]:
    if ',' in list_str:
        return [int(i.strip()) for i in list_str.split(',')]
    elif '-' in list_str:
        parts = list_str.split('-')
        if len(parts) != 2:
            raise ValueError(f'Invalid range format: {list_str}')
        start, end = int(parts[0].strip()), int(parts[1].strip())
        return list(range(start, end + 1))
    else:
        return [int(list_str.strip())]


def load_eids(eid_file: str) -> list[str]:
    eids = []
    with open(eid_file, 'r') as f:
        for line in f:
            eid = line.strip()
            if eid:
                eids.append(eid)
    return eids


def pattern_found_in_log(pattern: str, log_content: str) -> bool:
    """
    Check whether any line of log_content contains the key fragments from the pattern.

    Transfer log lines look like:
      Transferring file /teamspace/.../Subjects/X/date/num/alf/_ibl_leftCamera.lightningPose.pqt to patcher/...

    The pattern looks like:
      Transferring file */Subjects/*/*/*/alf/_ibl_leftCamera.lightningPose.pqt

    Strategy: split on '*', take the fixed prefix (before first '*') and fixed suffix
    (after last '*'), and check both appear in the line.
    """
    parts = [p for p in pattern.split('*') if p]
    prefix = parts[0] if parts else ''   # e.g. 'Transferring file '
    suffix = parts[-1] if parts else ''  # e.g. '/alf/_ibl_leftCamera.lightningPose.pqt'

    for line in log_content.splitlines():
        if prefix and prefix not in line:
            continue
        if suffix and suffix not in line:
            continue
        return True
    return False


def build_checks(run_lp: bool, run_lp_qc: bool, cameras: list[str]) -> list[dict]:
    """
    Build a list of check descriptors.
    Each descriptor: {'label': str, 'pattern': str, 'is_glob': bool}
    """
    checks = []

    if run_lp:
        checks.append({'label': 'LP success', 'pattern': LP_SUCCESS_STRING, 'is_glob': False})
        for cam in cameras:
            for pat in LP_TRANSFER_PATTERNS[cam]:
                checks.append({'label': f'LP transfer [{cam}]: {pat}', 'pattern': pat, 'is_glob': True})

    if run_lp_qc:
        checks.append({'label': 'Post-LP success', 'pattern': LP_QC_SUCCESS_STRING, 'is_glob': False})
        for cam in cameras:
            for pat in LP_QC_TRANSFER_PATTERNS[cam]:
                checks.append({'label': f'LP-QC transfer [{cam}]: {pat}', 'pattern': pat, 'is_glob': True})
        checks.append({'label': f'LP-QC licks transfer: {LP_QC_LICKS_PATTERN}',
                       'pattern': LP_QC_LICKS_PATTERN, 'is_glob': True})

    return checks


def fetch_job_logs(eid: str) -> str | None:
    """Fetch and return full log string for a job by EID name, or None on failure."""
    try:
        job = Job(eid)
        return job.logs or ''
    except Exception as e:
        print(f'    ERROR fetching logs for job {eid}: {e}')
        return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if not args.lp and not args.lp_qc:
        print('WARNING: neither --lp nor --lp_qc specified; nothing to check.')
        return

    # load EIDs
    print(f'Loading EIDs from {args.eid_file}')
    all_eids = load_eids(args.eid_file)
    indices = parse_list_arg(args.list)
    eids = [all_eids[idx] for idx in indices]

    print(f'Sessions to check : {len(eids)}  (indices {args.list} → {indices})')
    print(f'  --lp            : {args.lp}')
    print(f'  --lp_qc         : {args.lp_qc}')
    print(f'  --cameras       : {args.cameras}')
    print()

    # build check list once (shared for all sessions)
    checks = build_checks(args.lp, args.lp_qc, args.cameras)

    # ── per-session results ──
    results = {}  # eid → {'missing': [...], 'log_error': bool}

    for i, eid in enumerate(eids):
        print(f'{"=" * 70}')
        print(f'[{i+1}/{len(eids)}] {eid}')
        print(f'{"=" * 70}')

        log_content = fetch_job_logs(eid)
        if log_content is None:
            results[eid] = {'missing': ['<could not fetch logs>'], 'log_error': True}
            print()
            continue

        missing = []
        for check in checks:
            pattern = check['pattern']
            is_glob = check['is_glob']
            found = pattern_found_in_log(pattern, log_content) if is_glob else (pattern in log_content)
            status = '✓' if found else '✗'
            print(f'  {status}  {check["label"]}')
            if not found:
                missing.append(check['label'])

        results[eid] = {'missing': missing, 'log_error': False}

        if not missing:
            print('\n  ✅  All checks passed!')
        else:
            print(f'\n  ⚠️  {len(missing)} check(s) MISSING (see ✗ above)')
        print()

    # ── summary ──
    print('=' * 70)
    print('SUMMARY')
    print('=' * 70)

    sessions_ok = [eid for eid, r in results.items() if not r['missing']]
    sessions_fail = [eid for eid, r in results.items() if r['missing']]

    print(f'Total sessions   : {len(eids)}')
    print(f'All checks passed: {len(sessions_ok)}')
    print(f'Missing ≥1 check : {len(sessions_fail)}')

    if sessions_fail:
        print('\nSessions with missing checks:')
        for eid in sessions_fail:
            missing = results[eid]['missing']
            print(f'\n  {eid}  ({len(missing)} missing)')
            for item in missing:
                print(f'    ✗ {item}')

    print()


if __name__ == '__main__':
    main()
