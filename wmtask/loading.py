import os

import torch
import wandb
from omegaconf import OmegaConf

from .model import BiologicalRNN


def _unwrap_wandb_config(obj):
    """Recursively unwrap W&B API config dicts like {"value": x} -> x.

    Implemented via an inner recursive function to avoid name resolution issues.
    """
    def _fn(o):
        if isinstance(o, dict):
            if 'value' in o and all(k in {'value', 'desc', 'type'} for k in o.keys()):
                return _fn(o['value'])
            return {k: _fn(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(_fn(v) for v in o)
        return o
    return _fn(obj)


def _resolve_save_dir(save_dir=None):
    """Resolve the directory where WMTask model checkpoints are stored.

    Priority:
    1. Explicit ``save_dir`` argument
    2. ``WMTASK_MODELS_DIR`` environment variable
    3. Legacy path detection (backward compatibility)
    """
    if save_dir is not None:
        return save_dir
    if "WMTASK_MODELS_DIR" in os.environ:
        return os.environ["WMTASK_MODELS_DIR"]
    # Legacy path detection for backward compatibility
    for path, dir_val in [
        ('/Users/adameisen', "/Users/adameisen/Documents/MIT/Data/WMTaskModels"),
        ('/scratch2', "/scratch2/weka/millerlab/eisenaj/ControlJacobians/WMTaskModels"),
        ('/orcd/data/ekmiller/001/eisenaj/', "/orcd/data/ekmiller/001/eisenaj/ControlJacobians/WMTaskModels"),
        ('/home/millerlab-gpu', "/home/millerlab-gpu/data/ControlJacobians/WMTaskModels"),
    ]:
        if os.path.exists(path):
            return dir_val
    raise ValueError(
        "No save directory for WMTaskModels found. "
        "Set the WMTASK_MODELS_DIR environment variable or pass save_dir explicitly."
    )


def load_wmtask_model(project, name, model_to_load='final', save_dir=None):
    """Load a trained WMTask BiologicalRNN model from W&B.

    Args:
        project: W&B project name.
        name: W&B run name.
        model_to_load: Which checkpoint to load. 'final' for the last epoch,
            'init' for the initial (untrained) model, or an int for a specific epoch.
        save_dir: Directory containing model checkpoints. If None, resolved
            via WMTASK_MODELS_DIR env var or legacy path detection.

    Returns:
        Tuple of (model, params) where model is a BiologicalRNN and params
        is an OmegaConf config dict.
    """
    api = wandb.Api()
    runs = api.runs(project)
    run = [run for run in runs if run.name == name][0]

    # Get run config, handling W&B's public API format which may wrap values.
    params_dict = run.config
    if isinstance(params_dict, str):
        try:
            import json
            params_dict = json.loads(params_dict)
        except Exception:
            pass
    params_dict = _unwrap_wandb_config(params_dict)
    params = OmegaConf.create(params_dict)

    params['save_dir'] = _resolve_save_dir(save_dir)

    model_load_dir = os.path.join(params['save_dir'], project, name)

    if 'enforce_fixation' not in params:
        params['enforce_fixation'] = False

    if model_to_load == 'final':
        model_to_load = params['max_epochs'] - 1
    elif model_to_load == 'init':
        pass
    elif isinstance(model_to_load, int):
        model_to_load = model_to_load
    else:
        raise ValueError(f"model_to_load must be 'final', 'init', or an integer, got {model_to_load}")

    # LOAD MODEL
    if model_to_load == 'init':
        torch.manual_seed(params['random_state'])
        if 'init_mode' in params.keys():
            init_mode = params['init_mode']
        else:
            init_mode = 'learned'
        model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'], enforce_fixation=params['enforce_fixation'], init_mode=init_mode)
    else:
        filename = f"model-epoch={model_to_load}.ckpt"
        if torch.cuda.is_available():
            state_dict = torch.load(os.path.join(model_load_dir, filename), weights_only=False)['state_dict']
        else:
            state_dict = torch.load(os.path.join(model_load_dir, filename), weights_only=False, map_location='cpu')['state_dict']

        state_dict = {k.split('.')[1]: v for k, v in state_dict.items()}
        if 'init_mode' in params.keys():
            init_mode = params['init_mode']
        else:
            init_mode = 'learned'

        if 'enforce_fixation' in params.keys():
            model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'], enforce_fixation=params['enforce_fixation'], init_mode=init_mode)
        else:
            model = BiologicalRNN(params['input_dim'], params['hidden_dim'], output_dim=params['num_stimuli'], dt=params['dt'], tau=params['tau'], init_mode=init_mode)
        model.load_state_dict(state_dict)
        print(f"loaded wmtask RNN model checkpoint {model_to_load}")
    return model, params
