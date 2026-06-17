"""Migrate lightning_pose networks from v2.1 to v2.2.

iblvideo is being updated to use lightning-pose 2.2.0 (previously 1.6.1). The 2.2.0 release
includes significant backend refactoring of the model architecture (see lightning-pose PR #279),
which reorganised CRNN and upsampling modules under a unified `head` object. This changes the
state dict key names but not the underlying weights, so the trained models remain valid and only
the keys need updating.

For each network directory this script:
  1. Copies all non-checkpoint files to the mirrored v2.2 location.
  2. Creates a top-level config.yaml from .hydra/config.yaml when none exists at the top level.
  3. Updates config.yaml to use data.keypoint_names (removing data.keypoints if present).
     When keypoint_names is absent, falls back to the column headers in predictions_pixel_error.csv.
  4. Migrates .ckpt state dict keys:
       crnn.X                 -> head.head_mf.X
       upsampling_layers_sf.X -> head.head_sf.upsampling_layers.X
     Drops duplicate keys (upsampling_layers_rnn.*, upsampling_layers.*, unnormalized_weights),
     which held the same tensors as the renamed keys in the old architecture.
"""

import shutil
from pathlib import Path

import torch
import yaml


SRC = Path(
    '/home/mattw/Downloads/ONE/openalyx.internationalbrainlab.org'
    '/resources/lightning_pose/networks_v2.1'
)
DST = Path(
    '/home/mattw/Downloads/ONE/openalyx.internationalbrainlab.org'
    '/resources/lightning_pose/networks_v2.2'
)

KEYS_TO_DROP = {'unnormalized_weights'}


# ---------------------------------------------------------------------------
# checkpoint migration
# ---------------------------------------------------------------------------

def _migrate_state_dict(state_dict: dict) -> dict:
    """Apply v2.1 -> v2.2 key renames to a state dict.

    MHCRNN models (HeatmapTrackerMHCRNN) stored upsampling weights under both
    upsampling_layers_sf.* and a flat upsampling_layers.* duplicate. The flat copy is dropped
    and upsampling_layers_sf.* is renamed to head.head_sf.upsampling_layers.*.

    Plain HeatmapTracker models (e.g. roi_detect) only ever had the flat upsampling_layers.*
    with no _sf counterpart. For those the flat keys are renamed to head.upsampling_layers.*.

    Args:
        state_dict: original state dict loaded from a v2.1 checkpoint.

    Returns:
        Updated state dict with renamed keys and duplicates removed.
    """
    has_upsampling_sf = any(k.startswith('upsampling_layers_sf.') for k in state_dict)

    new_sd = {}
    for key, tensor in state_dict.items():
        if key in KEYS_TO_DROP:
            continue
        elif key.startswith('upsampling_layers_rnn.'):
            # duplicate of crnn.layers.*; covered by the crnn -> head.head_mf rename
            continue
        elif key.startswith('upsampling_layers.') and has_upsampling_sf:
            # flat duplicate of upsampling_layers_sf.* in MHCRNN models; drop it
            continue
        elif key.startswith('upsampling_layers.') and not has_upsampling_sf:
            # plain HeatmapTracker: the flat keys are the real upsampling weights
            new_key = 'head.' + key
        elif key.startswith('crnn.'):
            new_key = 'head.head_mf.' + key[len('crnn.'):]
        elif key.startswith('upsampling_layers_sf.'):
            new_key = 'head.head_sf.upsampling_layers.' + key[len('upsampling_layers_sf.'):]
        else:
            new_key = key
        new_sd[new_key] = tensor
    return new_sd


def migrate_checkpoint(src_path: Path, dst_path: Path) -> None:
    """Load, migrate, and save a single checkpoint file.

    Args:
        src_path: path to the source v2.1 checkpoint.
        dst_path: path to write the migrated v2.2 checkpoint.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(src_path, weights_only=False, map_location='cpu')
    old_n = len(ckpt['state_dict'])
    ckpt['state_dict'] = _migrate_state_dict(ckpt['state_dict'])
    new_n = len(ckpt['state_dict'])
    torch.save(ckpt, dst_path)
    print(f'    checkpoint: {old_n} keys -> {new_n} keys')


# ---------------------------------------------------------------------------
# config migration
# ---------------------------------------------------------------------------

def _keypoints_from_pixel_error_csv(network_dir: Path) -> list[str] | None:
    """Extract keypoint names from the header row of predictions_pixel_error.csv.

    The CSV has columns like: <filename>, kp1, kp2, ..., set.
    We drop the first (filename) and last (set) columns.

    Args:
        network_dir: top-level directory of the network in v2.1.

    Returns:
        List of keypoint name strings, or None if the file is absent.
    """
    csv_path = network_dir / 'predictions_pixel_error.csv'
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        header = f.readline().strip().split(',')
    # drop leading empty/filename column and trailing 'set' column
    return [col for col in header[1:] if col != 'set']


def migrate_config(src_config: Path, dst_config: Path, network_dir: Path) -> None:
    """Load a config, update keypoint field names, and write to dst.

    Renames data.keypoints -> data.keypoint_names (deleting keypoints).
    Falls back to predictions_pixel_error.csv if neither field is present.

    Args:
        src_config: source config.yaml path.
        dst_config: destination config.yaml path.
        network_dir: top-level v2.1 network directory (for CSV fallback).
    """
    dst_config.parent.mkdir(parents=True, exist_ok=True)
    with src_config.open() as f:
        cfg = yaml.safe_load(f)

    data = cfg.get('data', {})
    has_keypoint_names = 'keypoint_names' in data
    has_keypoints = 'keypoints' in data

    if has_keypoint_names and has_keypoints:
        # both present: drop the old field
        del data['keypoints']
        print(f'    config: removed data.keypoints (data.keypoint_names already present)')
    elif has_keypoints and not has_keypoint_names:
        # only old field: rename it
        data['keypoint_names'] = data.pop('keypoints')
        print(f'    config: renamed data.keypoints -> data.keypoint_names')
    elif not has_keypoint_names:
        # neither: fall back to CSV
        names = _keypoints_from_pixel_error_csv(network_dir)
        if names:
            data['keypoint_names'] = names
            print(f'    config: added data.keypoint_names from predictions_pixel_error.csv: {names}')
        else:
            print(f'    config: WARNING - could not determine keypoint names for {network_dir.name}')
    else:
        print(f'    config: data.keypoint_names already correct, no changes needed')

    cfg['data'] = data
    with dst_config.open('w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# per-network orchestration
# ---------------------------------------------------------------------------

def migrate_network(network_dir: Path) -> None:
    """Migrate a single network directory from v2.1 to v2.2.

    Args:
        network_dir: top-level directory of one network within v2.1.
    """
    rel = network_dir.relative_to(SRC)
    dst_network_dir = DST / rel
    print(f'\n{rel}')

    # --- copy all non-checkpoint files ---
    for src_file in network_dir.rglob('*'):
        if src_file.is_dir() or src_file.suffix == '.ckpt':
            continue
        dst_file = DST / src_file.relative_to(SRC)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)

    # --- migrate checkpoints ---
    for src_ckpt in sorted(network_dir.rglob('*.ckpt')):
        dst_ckpt = DST / src_ckpt.relative_to(SRC)
        print(f'  {src_ckpt.name}')
        migrate_checkpoint(src_ckpt, dst_ckpt)

    # --- ensure top-level config.yaml exists ---
    top_config = dst_network_dir / 'config.yaml'
    if not top_config.exists():
        hydra_config = dst_network_dir / '.hydra' / 'config.yaml'
        if hydra_config.exists():
            shutil.copy2(hydra_config, top_config)
            print(f'  config.yaml: created from .hydra/config.yaml')
        else:
            print(f'  config.yaml: WARNING - no source config found')

    # --- update keypoint field names in top-level config ---
    if top_config.exists():
        migrate_config(top_config, top_config, network_dir)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """Migrate all networks from networks_v2.1 to networks_v2.2."""
    network_dirs = sorted(d for d in SRC.iterdir() if d.is_dir())
    print(f'migrating {len(network_dirs)} networks from {SRC.name} -> {DST.name}')
    for network_dir in network_dirs:
        migrate_network(network_dir)
    print('\ndone.')


if __name__ == '__main__':
    main()
