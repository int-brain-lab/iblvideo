# Batch processing on Lightning AI

These scripts run the Lightning Pose (LP) and Post-LP pipeline steps on many IBL sessions from a
[Lightning AI](https://lightning.ai) Studio.

- `process_single_session.py` — runs LP and/or Post-LP for one session (one EID).
- `run_task.py` — batches many sessions by calling `process_single_session.py` for each EID, either
  sequentially on the current Studio or in parallel as separate Lightning Jobs.
- `parse_job_logs.py` — checks the logs of jobs launched with `run_task.py --run_parallel` for the
  expected per-camera success/transfer strings, so you don't have to open every job's log by hand.
- `eids_example.txt` — example EID list format (one session UUID per line).

## 1. Create a Lightning Studio

1. Sign in at [lightning.ai](https://lightning.ai) and pick the teamspace you want to run under.
2. Create a new Studio (`+ New Studio`) and give it a descriptive name, e.g. `iblvideo-inference`.
3. Start the Studio and open a terminal in it. A GPU machine (e.g. T4) is required to run LP.

## 2. Install the environment

Follow the main [README](../../README.md#installation) (the "Standalone / development" section),
with two caveats specific to running on a Lightning Studio:

- **Skip creating a conda environment.** A Studio is effectively one big conda environment already
  (`cloudspace`), and `process_single_session.py` points Lightning Pose at it directly via
  `task.lpenv = Path('/home/zeus/miniconda3/envs/cloudspace/bin/activate')` (see
  `process_lightning_pose`). Just run the `pip install -e ".[pose]"` step directly in the Studio's
  default environment.
- **Don't use the `openalyx` one-time setup snippet.** These scripts register processed datasets
  back to Alyx, which the public `openalyx` mirror doesn't allow. Instead run `ONE.setup()` pointed
  at `alyx.internationalbrainlab.org` with your own Alyx username and password:

```bash
python -c "from one.api import ONE; ONE.setup(base_url='https://alyx.internationalbrainlab.org')"
```

`lightning_sdk` (used by `run_task.py --run_parallel` and `parse_job_logs.py`) is preinstalled on
Lightning Studios. If it's missing for some reason: `pip install lightning-sdk`.

## 3. Prepare your EID list

Create a text file with one experiment ID (EID) per line, e.g. `eids.txt` (see
`eids_example.txt` for the format). `run_task.py` and `parse_job_logs.py` select sessions from this
file by 0-based index via `--list`, either as a comma-separated list (`0,2,5,8`) or an inclusive
range (`0-14`).

## 4. Launch jobs

### Single session

```bash
python process_single_session.py <eid> --run_lp --run_post_lp --cameras left right body
```

Add `--dry` to just verify ONE connectivity without running anything.

### Batch, sequential (one Studio, one session after another)

```bash
python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --cameras left right body
```

### Batch, parallel (one Lightning Job per session, running concurrently)

```bash
python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --run_parallel --cameras left right body
```

Each session is submitted as its own Lightning Job (`Machine.T4_SMALL` by default). `run_task.py`
sleeps 10 minutes between job submissions to avoid an AWS managed-prefix-list rate limit, so
submitting a large batch takes a while up front even though the sessions themselves run
concurrently once submitted.

## 5. Check job status and logs

- **Lightning UI**: open the teamspace's Jobs tab to see each job's status (`Pending`, `Running`,
  `Completed`, `Failed`) and its raw log, named by EID.
- **`parse_job_logs.py`**: pulls each job's log via `lightning_sdk` and checks, per camera, for the
  LP/Post-LP success message and the expected dataset transfer lines, giving a quick pass/fail
  summary without opening every job in the UI:

```bash
python parse_job_logs.py --eid_file eids.txt --list 0-14 --lp --lp_qc --cameras left right body
```

Use `--lp` to check Lightning Pose output, `--lp_qc` to check Post-LP/QC output, or both.
