import hydra
import lightning as L
import logging
import numpy as np
import os
import torch
from torch import nn
from omegaconf import OmegaConf
import wandb

from .model import BiologicalRNN, LitBiologicalRNN
from .dataset import WMSelectionDataset

log = logging.getLogger('WMTask Logger')


@hydra.main(config_path="conf", config_name="config.yaml", version_base='1.3')
def run_wmtask(cfg):
    log.info(f"dt = {cfg.wmtask_params.dt}")
    # ----------------
    # Update config
    # ----------------
    cfg.wmtask_params.input_dim = cfg.wmtask_params.num_stimuli
    cfg.wmtask_params.N2 = cfg.wmtask_params.N1
    cfg.wmtask_params.hidden_dim = cfg.wmtask_params.N1 + cfg.wmtask_params.N2

    project = "__".join(["WMSelectionTask"] + [f"{k}_{v}" for k, v, in cfg.wmtask_params.items() if k in ['cue_time', 'response_time', 'enforce_fixation']])

    name_keys = ['N1', 'N2', 'tau', 'dt', 'eig_lower_bound', 'learning_rate', 'max_epochs', 'cue_time', 'init_mode']
    name = "BiologicalRNN__" + "__".join([f"{k}_{v}" for k, v, in cfg.wmtask_params.items() if k in name_keys])

    model_save_dir = os.path.join(cfg.wmtask_params.save_dir, project, name)
    lit_dir = os.path.join(cfg.wmtask_params.save_dir, project, 'lightning')

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(lit_dir, exist_ok=True)

    log.info("Checking for preexisting runs...")
    api = wandb.Api()
    runs = api.runs(project)
    try:
        found_run = False
        for run in runs:
            if run.name == name:
                found_run = True
                break
        if found_run:
            print(f"Found run {name}, skipping...")
            log.info(f"Found run {name}, skipping...")
            return
    except ValueError:
        log.info(f"Project {project} does not exist!")

    # ----------------
    # Data generation
    # ----------------
    log.info("Generating data...")
    np.random.seed(cfg.wmtask_params.random_state)
    color_stimuli = nn.functional.one_hot(torch.arange(cfg.wmtask_params.num_stimuli), cfg.wmtask_params.num_stimuli).type(torch.FloatTensor)

    color_nums = torch.arange(4)
    color1_index = torch.randint(low=0, high=cfg.wmtask_params.num_stimuli, size=(cfg.wmtask_params.num_trials,))
    color1_input = color_stimuli[color1_index]
    color2_index = torch.tensor([torch.cat((color_nums[:c_ind], color_nums[c_ind + 1:]))[torch.randint(low=0, high=3, size=(1,))][0] for c_ind in color1_index])
    color2_input = color_stimuli[color2_index]

    context_input = nn.functional.one_hot(torch.randint(low=0, high=2, size=(cfg.wmtask_params.num_trials,)), 2)
    color_labels = torch.cat((color1_index.unsqueeze(-1), color2_index.unsqueeze(-1)), axis=1)[context_input.type(torch.BoolTensor)]

    stacked_inputs = torch.cat((color1_input, color2_input, context_input), axis=1)

    train_inds = np.sort(np.random.choice(np.arange(cfg.wmtask_params.num_trials), size=(int(cfg.wmtask_params.train_percent*cfg.wmtask_params.num_trials)), replace=False))
    val_inds = np.array([i for i in np.arange(cfg.wmtask_params.num_trials) if i not in train_inds])
    train_dataset = WMSelectionDataset(stacked_inputs[train_inds], color_labels[train_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time, cfg.wmtask_params.response_time, cfg.wmtask_params.enforce_fixation)
    val_dataset = WMSelectionDataset(stacked_inputs[val_inds], color_labels[val_inds], cfg.wmtask_params.dt, cfg.wmtask_params.input_dim, cfg.wmtask_params.fixation_time, cfg.wmtask_params.stimuli_time, cfg.wmtask_params.delay1_time, cfg.wmtask_params.cue_time, cfg.wmtask_params.delay2_time, cfg.wmtask_params.response_time, cfg.wmtask_params.enforce_fixation)

    num_workers = 1
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.wmtask_params.batch_size, shuffle=True, num_workers=num_workers, persistent_workers=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.wmtask_params.batch_size, shuffle=False, num_workers=num_workers, persistent_workers=False)

    # ----------------
    # Lightning model
    # ----------------
    torch.manual_seed(cfg.wmtask_params.random_state)
    model = BiologicalRNN(cfg.wmtask_params.input_dim, cfg.wmtask_params.hidden_dim, output_dim=cfg.wmtask_params.num_stimuli, dt=cfg.wmtask_params.dt, tau=cfg.wmtask_params.tau, eig_lower_bound=cfg.wmtask_params.eig_lower_bound, enforce_fixation=cfg.wmtask_params.enforce_fixation, init_mode=cfg.wmtask_params.init_mode)
    lit_model = LitBiologicalRNN(model, learning_rate=cfg.wmtask_params.learning_rate, enforce_fixation=cfg.wmtask_params.enforce_fixation)
    logger = L.pytorch.loggers.WandbLogger(save_dir=lit_dir, log_model=True, name=name, project=project)
    logger.experiment.config.update(OmegaConf.to_container(cfg.wmtask_params))

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=None,
        dirpath=model_save_dir,
        filename='model-{epoch}',
        save_top_k = -1,
        every_n_epochs = 1,
    )

    torch.autograd.set_detect_anomaly(True)
    trainer = L.Trainer(logger=logger, max_epochs=cfg.wmtask_params.max_epochs, callbacks=[checkpoint_callback])
    trainer.fit(model=lit_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    wandb.finish()


if __name__ == "__main__":
    run_wmtask()
