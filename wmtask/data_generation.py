import numpy as np
import torch
import torch.nn as nn

from .dataset import WMSelectionDataset
from .analysis import get_hiddens


def generate_wmtask_data(params):
    """Generate working memory task data and return dataloaders.

    Args:
        params: OmegaConf config with task parameters (num_stimuli, num_trials,
            dt, timing params, etc.)

    Returns:
        Tuple of (all_dataloader, train_dataloader, val_dataloader, test_dataloader).
    """
    np.random.seed(params.random_state)
    torch.manual_seed(params.random_state)
    color_stimuli = nn.functional.one_hot(torch.arange(params.num_stimuli), params.num_stimuli).type(torch.FloatTensor)

    color_nums = torch.arange(4)
    color1_index = torch.randint(low=0, high=params.num_stimuli, size=(params.num_trials,))
    color1_input = color_stimuli[color1_index]
    color2_index = torch.tensor([torch.cat((color_nums[:c_ind], color_nums[c_ind + 1:]))[torch.randint(low=0, high=3, size=(1,))][0] for c_ind in color1_index])
    color2_input = color_stimuli[color2_index]

    context_input = nn.functional.one_hot(torch.randint(low=0, high=2, size=(params.num_trials,)), 2)
    color_labels = torch.cat((color1_index.unsqueeze(-1), color2_index.unsqueeze(-1)), axis=1)[context_input.type(torch.BoolTensor)]

    stacked_inputs = torch.cat((color1_input, color2_input, context_input), axis=1)

    train_inds = np.sort(np.random.choice(np.arange(params.num_trials), size=(int(params.train_percent*params.num_trials)), replace=False))
    val_inds = np.array([i for i in np.arange(params.num_trials) if i not in train_inds])
    test_inds = np.sort(np.random.choice(val_inds, size=(150,)))
    if 'enforce_fixation' in params.keys():
        all_dataset = WMSelectionDataset(stacked_inputs, color_labels, params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
        test_dataset = WMSelectionDataset(stacked_inputs[test_inds], color_labels[test_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time, params.response_time, params.enforce_fixation)
    else:
        all_dataset = WMSelectionDataset(stacked_inputs, color_labels, params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)
        test_dataset = WMSelectionDataset(stacked_inputs[test_inds], color_labels[test_inds], params.dt, params.input_dim, params.fixation_time, params.stimuli_time, params.delay1_time, params.cue_time, params.delay2_time)

    num_workers = 1
    all_dataloader = torch.utils.data.DataLoader(all_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)

    return all_dataloader, train_dataloader, val_dataloader, test_dataloader


def generate_model_trajectories(model, dataloader, params, verbose=False):
    """Extract hidden state trajectories from a model given a dataloader.

    Args:
        model: A BiologicalRNN model.
        dataloader: A DataLoader with WMSelectionDataset.
        params: Config dict with at least 'N1' and 'N2' keys.
        verbose: Whether to show progress bar.

    Returns:
        Hidden state tensor of shape (n_trials, n_timepoints, hidden_dim).
    """
    hiddens = get_hiddens(model, dataloader, verbose=verbose)
    return hiddens
