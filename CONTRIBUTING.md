# Contributing

We welcome community contributions to the iblvideo repo!
If you have found a bug or would like to request a minor change, please
[open an issue](https://github.com/int-brain-lab/iblvideo/issues).

In order to contribute code to the repo, please follow the steps below.

### Set up a development installation

In order to make changes to iblvideo, you will need to
[fork](https://guides.github.com/activities/forking/#fork) the
[repo](https://github.com/int-brain-lab/iblvideo).

If you are not familiar with `git`, check out
[this guide](https://guides.github.com/introduction/git-handbook/#basic-git).

Clone your fork and install in editable mode with development dependencies:

```bash
git clone https://github.com/<your-username>/iblvideo.git
cd iblvideo
pip install -e ".[dev]"
```

If your work also requires pose estimation or action segmentation dependencies, add the
relevant extras:

```bash
pip install -e ".[dev,pose]"   # for pose estimation
pip install -e ".[dev,action]" # for action segmentation
pip install -e ".[dev,all]"    # for everything
```

### Create a pull request

After making changes in your fork,
[open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
from your fork. Please read through the rest of this document before submitting.

#### Linting

We use [ruff](https://docs.astral.sh/ruff/) for import sorting and linting.
Run both commands from the repo root before submitting a pull request:

```bash
ruff check --fix iblvideo tests
```

To check without modifying files:

```bash
ruff check iblvideo tests
```

If you set up the pre-commit hook (see below), ruff runs automatically on every commit.

#### Pre-commit hook

Install the pre-commit hook so ruff runs automatically before each commit:

```bash
pre-commit install
```

To run it manually against all files:

```bash
pre-commit run --all-files
```

#### Testing

No single environment can run all tests. The LP and LA tests require their respective extras
(`pose` or `action`), and DLC tests must be run in a separate DLC environment — see
[README_DLC.md](README_DLC.md) for setup instructions.

Run the LP/LA test suite from the repo root:

```bash
pytest
```

To run a specific subset of tests:

```bash
pytest tests/test_pose_lp.py
```

Note that some tests require access to a GPU and/or network access to download model
checkpoints. These may be skipped in environments without those resources.

### Releasing a new version

We use semantic versioning, with a prefix: `iblvideo_MAJOR.MINOR.PATCH`. If you update the version, see below for what to adapt.

#### Any version update

Update the version in
```
iblvideo/iblvideo/__init__.py
```
Afterwards, tag the new version on Github.


#### Network model versioning
As of `iblvideo v3.0.0` we are no longer linking the versioning of the networks models with the code version. 
When the models are updated the test data should also be updated.
Then upload both to the private and public S3 bucket in resources/lightning_pose with filename `networks_vX.Y.zip` and `lp_test_data_vX.Y.zip` respectively. 
Always keep the old models for reproducibility.
Then update the default LP version number to the new version `vX.Y` in `iblvideo.__init__.py` (parameter `__lp_version__`).

You should always also bump the code version in `iblvideo/__init__.py` when you update the models. 
This way, the code version that is stored in the alyx task can always be linked to a specific model version.
