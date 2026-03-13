import numpy as np
import torch
from torch import nn


class WMSelectionDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, labels, dt, input_dim, fixation_time, stimuli_time, delay1_time, cue_time, delay2_time, response_time, enforce_fixation):
        self.inputs = inputs
        self.labels = labels

        self.input_dim = input_dim

        self.fixation_time = fixation_time
        self.stimuli_time = stimuli_time
        self.delay1_time = delay1_time
        self.cue_time = cue_time
        self.delay2_time = delay2_time
        response_time = response_time if response_time is not None else dt
        self.response_time = response_time
        self.n_response_t = int(np.round(response_time/dt))
        self.total_t = int(np.round((fixation_time + stimuli_time + delay1_time + cue_time + delay2_time + response_time)/dt))
        self.stim_start_t = int(np.round((self.fixation_time/dt)))
        self.stim_end_t = int(np.round((self.fixation_time + self.stimuli_time)/dt))
        self.cue_start_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time)/dt))
        self.cue_end_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time)/dt))
        self.response_start_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time + self.delay2_time)/dt))
        self.response_end_t = int(np.round((self.fixation_time + self.stimuli_time + self.delay1_time + self.cue_time + self.delay2_time + self.response_time)/dt))

        self.enforce_fixation = enforce_fixation

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        stacked_input = self.inputs[idx]

        input_sample = torch.zeros(self.total_t, len(stacked_input) + (1 if self.enforce_fixation else 0)).type(stacked_input.dtype).to(stacked_input.device)
        input_sample[self.stim_start_t:self.stim_end_t, :self.input_dim*2] = stacked_input[:self.input_dim*2] # stimulus inputs
        input_sample[self.cue_start_t:self.cue_end_t, self.input_dim*2:self.input_dim*2 + 2] = stacked_input[self.input_dim*2:] # cue input
        if self.enforce_fixation:
            input_sample[:self.response_start_t, -1] = 1

        label_sample = self.labels[idx].repeat(self.n_response_t)
        return input_sample, label_sample
