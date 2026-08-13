# Batch processing on Lightning AI

These scripts run the Lightning Pose (LP), Post-LP, Lightning Action (LA), and Post-LA pipeline
steps on many IBL sessions from a [Lightning AI](https://lightning.ai) Studio.

- `process_single_session.py` — runs any combination of LP, Post-LP, LA, and Post-LA for one
  session (one EID). LA and Post-LA only support the `left` and `right` cameras (no `body`).
- `run_task.py` — batches many sessions by calling `process_single_session.py` for each EID, either
  sequentially on the current Studio or in parallel as separate Lightning Jobs.
- `parse_job_logs.py` — checks the logs of jobs launched with `run_task.py --run_parallel` for the
  expected per-camera success/transfer strings, so you don't have to open every job's log by hand.
  Currently only checks LP/Post-LP strings; it does not yet support LA/Post-LA logs.
- `eids_example.txt` — example EID list format (one session UUID per line).

## 1. Create a Lightning Studio

1. Sign in at [lightning.ai](https://lightning.ai) and pick the teamspace you want to run under.
2. Create a new Studio (`+ New Studio`) and give it a descriptive name, e.g. `iblvideo-inference`.
3. Start the Studio and open a terminal in it. A GPU machine (e.g. T4) is required to run LP/LA.
4. Switch the Studio's Python version to **3.10.20** before installing anything, in the Studio's
   environment/interpreter settings. This is a temporary pin: newer Python versions (e.g. 3.12) can
   hang for a very long time on `pip install -e ".[pose]"` while `nvidia-dali-cuda110` (a
   Lightning Pose dependency) silently fetches its prebuilt binary in the background with no
   progress indicator at the `Preparing metadata (pyproject.toml)` step. 3.10.20 is verified to work.

## 2. Install the environment

Follow the main [README](../../README.md#installation) (the "Standalone / development" section),
with two caveats specific to running on a Lightning Studio:

- **Skip creating a conda environment.** A Studio is effectively one big conda environment already
  (`cloudspace`), and `process_single_session.py` points Lightning Pose/Action at it directly via
  `task.lpenv`/`task.laenv = Path('/home/zeus/miniconda3/envs/cloudspace/bin/activate')` (see
  `process_lightning_pose`/`process_lightning_action`). 
  Just run the `pip install -e ".[pose,action]"` step directly in the Studio's default environment.
- **Don't use the main README's public-database one-time setup snippet on a Studio.** Both
  `process_single_session.py` and the test suite construct their `ONE` client with no `base_url`
  (a bare `ONE()`), so whichever database you set up as your default is what both of them use.
  `process_single_session.py` registers processed datasets back to Alyx, which the public
  `openalyx` mirror doesn't allow, so set up your own credentials for the **production** Alyx
  instead, before running anything (including the tests):

```bash
python -c "from one.api import ONE; ONE.setup(base_url='https://alyx.internationalbrainlab.org')"
```

Enter your credentials as prompted.
With this as your only configured database, `pytest tests/ -m "not dlc"` will also fetch its
sample data through your production credentials rather than the public mirror.

`lightning_sdk` (used by `run_task.py --run_parallel` and `parse_job_logs.py`) is not installed by
default:

```bash
pip install lightning-sdk
```

`iblscripts` is also required (used by the LP, Post-LP, LA, and Post-LA tasks) and isn't part of
the main README's install flow:

```bash
mkdir -p /teamspace/studios/this_studio/Documents/PYTHON
cd /teamspace/studios/this_studio/Documents/PYTHON
git clone https://github.com/int-brain-lab/iblscripts.git
```

## 3. Prepare your EID list

Create a text file with one experiment ID (EID) per line, e.g. `eids.txt` (see
`eids_example.txt` for the format). `run_task.py` and `parse_job_logs.py` select sessions from this
file by 0-based index via `--list`, either as a comma-separated list (`0,2,5,8`) or an inclusive
range (`0-14`).

## 4. Launch jobs

**Run these from inside `scripts/lightning/` (or anywhere other than your Studio's home
directory), not from `~`.** Your Studio's home directory is also where the `iblvideo` repo is
cloned, i.e. `~/iblvideo`. If your shell's current directory is `~` when you run `python`, Python
finds that `iblvideo` folder while scanning the current directory for imports and mistakes it for
an (empty, broken) namespace package — shadowing the real, properly-installed `iblvideo` package
and causing `AttributeError: module 'iblvideo' has no attribute '__version__'` deep inside the LP
task. `run_task.py` already works around this for you by `cd`-ing into `scripts/lightning/` before
launching each session, but a `process_single_session.py` invoked directly from `~` will still hit
it.

### Single session

```bash
python process_single_session.py <eid> --run_lp --run_post_lp --cameras left right body
python process_single_session.py <eid> --run_la --run_post_la --cameras left right
```

Add `--dry` to just verify ONE connectivity without running anything.

### Batch, sequential (one Studio, one session after another)

```bash
python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --cameras left right body
python run_task.py --eid_file eids.txt --list 0-14 --run_la --run_post_la --cameras left right
```

### Batch, parallel (one Lightning Job per session, running concurrently)

```bash
python run_task.py --eid_file eids.txt --list 0-14 --run_lp --run_post_lp --run_parallel --cameras left right body
python run_task.py --eid_file eids.txt --list 0-14 --run_la --run_post_la --run_parallel --cameras left right
```

Each session is submitted as its own Lightning Job (`Machine.T4_SMALL` by default). `run_task.py`
sleeps 2 minutes between job submissions to avoid an AWS managed-prefix-list rate limit, so
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
