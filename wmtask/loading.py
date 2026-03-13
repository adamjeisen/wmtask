import os
import re

import torch
from omegaconf import OmegaConf

from .model import BiologicalRNN


# Default task parameters — matches conf/wmtask_params/default.yaml
_DEFAULT_PARAMS = {
    "fixation_time": 0.5,
    "stimuli_time": 0.5,
    "delay1_time": 0.75,
    "cue_time": 0.1,
    "delay2_time": 0.75,
    "response_time": 0.25,
    "num_stimuli": 4,
    "num_trials": 4096,
    "enforce_fixation": False,
    "train_percent": 0.8,
    "batch_size": 32,
    "learning_rate": 5e-4,
    "max_epochs": 42,
    "N1": 64,
    "N2": 64,
    "tau": 0.05,
    "dt": 0.02,
    "eig_lower_bound": 0.1,
    "random_state": 42,
    "init_mode": "random",
}


def _parse_name_params(name):
    """Parse key=value pairs from a run name like 'BiologicalRNN__k1_v1__k2_v2'.

    Returns a dict of parsed values with automatic type coercion.
    """
    parsed = {}
    # Split on __ and look for key_value patterns
    parts = name.split("__")
    for part in parts:
        # Match patterns like "N1_64", "dt_0.02", "enforce_fixation_False"
        match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)_([^_].*)$', part)
        if match:
            key, val_str = match.group(1), match.group(2)
            # Type coercion
            if val_str.lower() in ('true', 'false'):
                parsed[key] = val_str.lower() == 'true'
            else:
                try:
                    parsed[key] = int(val_str)
                except ValueError:
                    try:
                        parsed[key] = float(val_str)
                    except ValueError:
                        parsed[key] = val_str
    return parsed


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
    """Load a trained WMTask BiologicalRNN model from local checkpoints.

    Params are constructed from defaults merged with values parsed from the
    project and run name strings, so no W&B query is needed.

    Args:
        project: Project directory name (e.g. 'WMSelectionTask__cue_time_0.1__...').
        name: Run name (e.g. 'BiologicalRNN__N1_64__N2_64__tau_0.05__...').
        model_to_load: Which checkpoint to load. 'final' for the last epoch,
            'init' for the initial (untrained) model, or an int for a specific epoch.
        save_dir: Directory containing model checkpoints. If None, resolved
            via WMTASK_MODELS_DIR env var or legacy path detection.

    Returns:
        Tuple of (model, params) where model is a BiologicalRNN and params
        is an OmegaConf config dict.
    """
    # Build params from defaults + values parsed from project/run names
    params_dict = dict(_DEFAULT_PARAMS)
    params_dict.update(_parse_name_params(project))
    params_dict.update(_parse_name_params(name))

    # Derive computed fields
    if params_dict.get('N2') is None:
        params_dict['N2'] = params_dict['N1']
    params_dict['hidden_dim'] = params_dict['N1'] + params_dict['N2']
    params_dict['input_dim'] = 2 * params_dict['num_stimuli'] + 2  # stimuli + context

    params = OmegaConf.create(params_dict)
    params['save_dir'] = _resolve_save_dir(save_dir)

    model_load_dir = os.path.join(params['save_dir'], project, name)

    if not os.path.isdir(model_load_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_load_dir}\n"
            f"Check that project='{project}' and name='{name}' match a directory under {params['save_dir']}"
        )

    if model_to_load == 'final':
        model_to_load = params['max_epochs'] - 1
    elif model_to_load == 'init':
        pass
    elif isinstance(model_to_load, int):
        pass
    else:
        raise ValueError(f"model_to_load must be 'final', 'init', or an integer, got {model_to_load}")

    init_mode = params.get('init_mode', 'learned')

    if model_to_load == 'init':
        torch.manual_seed(params['random_state'])
        model = BiologicalRNN(
            params['input_dim'], params['hidden_dim'],
            output_dim=params['num_stimuli'],
            dt=params['dt'], tau=params['tau'],
            enforce_fixation=params['enforce_fixation'],
            init_mode=init_mode,
        )
    else:
        filename = f"model-epoch={model_to_load}.ckpt"
        filepath = os.path.join(model_load_dir, filename)
        map_location = None if torch.cuda.is_available() else 'cpu'
        state_dict = torch.load(filepath, weights_only=False, map_location=map_location)['state_dict']
        state_dict = {k.split('.', 1)[1]: v for k, v in state_dict.items()}

        model = BiologicalRNN(
            params['input_dim'], params['hidden_dim'],
            output_dim=params['num_stimuli'],
            dt=params['dt'], tau=params['tau'],
            enforce_fixation=params['enforce_fixation'],
            init_mode=init_mode,
        )
        model.load_state_dict(state_dict)
        print(f"loaded wmtask RNN model checkpoint {model_to_load}")

    return model, params
