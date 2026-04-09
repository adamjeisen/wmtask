import torch
from torch import nn
from tqdm.auto import tqdm


def get_hiddens(model, dataloader_to_use, device=None, verbose=False):
    """Run `model` over `dataloader_to_use` and return hidden states.

    Args:
        model: A BiologicalRNN (or compatible) model.
        dataloader_to_use: DataLoader yielding (input_seq, labels) batches.
        device: Torch device to run on. If None, auto-selects 'cuda' when
            available and falls back to 'cpu'. The returned tensor is always
            on CPU.
        verbose: Show a tqdm progress bar over the dataloader.

    Returns:
        Hidden state tensor of shape (n_trials, n_timepoints, hidden_dim) on CPU.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    hiddens_all = torch.zeros(dataloader_to_use.dataset.labels.shape[0], dataloader_to_use.dataset.total_t, model.hidden_dim)
    model = model.to(device)
    with torch.no_grad():
        batch_loc = 0
        for input_seq, labels in tqdm(dataloader_to_use, disable=not verbose):
            input_seq = input_seq.to(device)
            hiddens = torch.zeros(input_seq.shape[0], input_seq.shape[1], model.hidden_dim, device=device)
            hidden = None
            for i in range(input_seq.shape[1]):
                if i == 0:
                    out, hidden = model(input_seq[:, [i]])
                else:
                    out, hidden = model(input_seq[:, [i]], hidden)
                hiddens[:, i] = hidden
            hiddens_all[batch_loc:batch_loc + input_seq.shape[0]] = hiddens.cpu()

            batch_loc += input_seq.shape[0]

    return hiddens_all


def ELU_deriv(h):
    deriv = torch.zeros(h.shape).type(h.dtype).to(h.device)
    deriv[h > 0] = 1
    deriv[h <= 0] = torch.exp(h[h <= 0])
    return deriv


def compute_model_jacs(model, h, dt, tau, discrete=False):
    if discrete:
        Js = torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + (dt/tau)*(-torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + model.W_hh.detach().type(h.dtype).to(h.device) @ torch.diag_embed(ELU_deriv(h).type(h.dtype).to(h.device)))
    else:
        Js = (1/tau)*(-torch.eye(h.shape[-1]).unsqueeze(0).type(h.dtype).to(h.device) + model.W_hh.detach().type(h.dtype).to(h.device) @ torch.diag_embed(ELU_deriv(h).type(h.dtype).to(h.device)))
    return Js


def compute_model_rhs(model, h, dt, tau):
    return (1/tau)*(-h + (model.W_hh.detach().type(h.dtype).to(h.device) @ nn.ELU()(h.unsqueeze(-1))).squeeze(-1).type(h.dtype).to(h.device))
