"""Bridge module for integrating wmtask with JacobianODE's data pipeline.

Provides functions that load a trained WM task model, generate hidden state
trajectories, window them to a specific task epoch, and return data in the
format expected by JacobianODE.
"""

import os

import numpy as np
import torch

from .loading import load_wmtask_model
from .data_generation import generate_wmtask_data, generate_model_trajectories
from .dynamics import WMTaskEq


def _hiddens_cache_path(params, project, name, model_to_load, dataloader_to_use):
    """Build the on-disk cache path for raw (pre-window) hiddens.

    Cache is colocated with the model checkpoint so it travels with the model.
    Key includes everything that affects `get_hiddens` output for a given
    checkpoint: the dataloader split, the epoch, and the params that drive
    dataset construction (num_trials, random_state).
    """
    cache_dir = os.path.join(params['save_dir'], project, name, '_jacobianode_cache')
    fname = (
        f"hiddens__{dataloader_to_use}"
        f"__epoch{model_to_load}"
        f"__trials{params['num_trials']}"
        f"__seed{params['random_state']}.pt"
    )
    return os.path.join(cache_dir, fname)


def _window_hiddens(hiddens, params, traj_window='delay2'):
    """Window hidden state trajectories to a specific task epoch.

    Args:
        hiddens: Tensor of shape (n_trials, n_timepoints, hidden_dim).
        params: OmegaConf config with task timing parameters.
        traj_window: Which epoch to extract. 'delay2' selects from the start
            of delay2 through the end of response. 'full' returns everything.

    Returns:
        Windowed hidden states as a numpy array.
    """
    dt = params['dt']
    if traj_window == 'delay2':
        start_ind = int((params['fixation_time'] + params['stimuli_time']
                        + params['delay1_time'] + params['cue_time']) / dt)
        end_ind = int((params['fixation_time'] + params['stimuli_time']
                      + params['delay1_time'] + params['cue_time']
                      + params['delay2_time'] + params['response_time']) / dt) - 1
    elif traj_window == 'full':
        start_ind = 0
        end_ind = hiddens.shape[-2]
    else:
        raise ValueError(f"Unknown traj_window: {traj_window}. Use 'delay2' or 'full'.")
    return hiddens[..., start_ind:end_ind, :]


def make_wmtask_trajectories(
    project,
    name,
    model_to_load='final',
    dataloader_to_use='all',
    traj_window='delay2',
    save_dir=None,
    verbose=False,
    device=None,
    use_cache=True,
):
    """Load a trained WM task model and generate windowed trajectories.

    This is the main entry point for obtaining wmtask data in a format
    compatible with JacobianODE's training pipeline.

    Raw (pre-window) hiddens are cached on disk next to the model checkpoint
    so repeated calls skip the RNN rollout entirely. The cache is keyed on
    `(dataloader_to_use, model_to_load, num_trials, random_state)`; changing
    `traj_window` does not invalidate it.

    Args:
        project: W&B project name.
        name: W&B run name.
        model_to_load: Which checkpoint to load ('final', 'init', or epoch int).
        dataloader_to_use: Which data split to use ('all', 'train', 'val', 'test').
        traj_window: Task epoch to window to ('delay2' or 'full').
        save_dir: Directory containing model checkpoints. If None, resolved
            via WMTASK_MODELS_DIR env var or legacy path detection.
        verbose: Whether to show progress bars.
        device: Torch device to run the RNN rollout on. If None, auto-selects
            'cuda' when available and falls back to 'cpu'.
        use_cache: If True (default), read/write the on-disk hiddens cache.

    Returns:
        Tuple of (eq, sol, dt) where:
            - eq: WMTaskEq dynamics object
            - sol: dict with 'values' key, shape (n_trials, n_timepoints, hidden_dim)
            - dt: timestep
    """
    model, params = load_wmtask_model(project, name, model_to_load=model_to_load, save_dir=save_dir)

    # Resolve the concrete epoch for cache keying ('final' -> max_epochs - 1;
    # load_wmtask_model handles this internally but doesn't return the resolved
    # value, so recompute it here).
    resolved_epoch = (params['max_epochs'] - 1) if model_to_load == 'final' else model_to_load

    cache_path = _hiddens_cache_path(params, project, name, resolved_epoch, dataloader_to_use)
    hiddens = None
    if use_cache and os.path.isfile(cache_path):
        if verbose:
            print(f"Loading cached wmtask hiddens from {cache_path}")
        hiddens = torch.load(cache_path, map_location='cpu', weights_only=True)

    if hiddens is None:
        all_dl, train_dl, val_dl, test_dl = generate_wmtask_data(params)
        dataloaders = {
            'all': all_dl,
            'train': train_dl,
            'val': val_dl,
            'test': test_dl,
        }
        dl = dataloaders[dataloader_to_use]

        hiddens = generate_model_trajectories(model, dl, params, verbose=verbose, device=device)

        if use_cache:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            # Atomic write: save to a temp path then rename, so a killed job
            # can't leave a truncated cache behind.
            tmp_path = cache_path + '.tmp'
            torch.save(hiddens, tmp_path)
            os.replace(tmp_path, cache_path)
            if verbose:
                print(f"Cached wmtask hiddens to {cache_path}")

    windowed = _window_hiddens(hiddens, params, traj_window=traj_window)

    # Convert to numpy array with shape (n_trials, n_timepoints, hidden_dim)
    values = windowed.numpy() if hasattr(windowed, 'numpy') else np.asarray(windowed)

    eq = WMTaskEq(model, params)
    sol = {"values": values}
    dt = float(params['dt'])

    return eq, sol, dt


def load_wmtask_for_jacobianode(
    project,
    name,
    model_to_load='final',
    dataloader_to_use='all',
    traj_window='delay2',
    save_dir=None,
    verbose=False,
    device=None,
    use_cache=True,
):
    """Convenience wrapper returning (eq, sol, dt) for JacobianODE's dataset_loader pattern.

    This function is designed to be called via Hydra's ``instantiate()`` from a
    JacobianODE data config file. It returns the WMTaskEq dynamics object along
    with data, enabling analytical Jacobian (eq.rhs, eq.jac) analyses in
    JacobianODE.

    Args:
        Same as ``make_wmtask_trajectories``.

    Returns:
        Tuple of (eq, sol, dt) where eq is WMTaskEq, sol is a dict with 'values' key.
    """
    eq, sol, dt = make_wmtask_trajectories(
        project=project,
        name=name,
        model_to_load=model_to_load,
        dataloader_to_use=dataloader_to_use,
        traj_window=traj_window,
        save_dir=save_dir,
        verbose=verbose,
        device=device,
        use_cache=use_cache,
    )
    return eq, sol, dt
