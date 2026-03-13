import torch
from torch import nn
from tqdm.auto import tqdm


def get_hiddens(model, dataloader_to_use, device='cpu', verbose=False):
    hiddens_all = torch.zeros(dataloader_to_use.dataset.labels.shape[0], dataloader_to_use.dataset.total_t, model.hidden_dim)
    model = model.to(device)
    with torch.no_grad():
        batch_loc = 0
        for input_seq, labels in tqdm(dataloader_to_use, disable=not verbose):
            hiddens = torch.zeros(input_seq.shape[0], input_seq.shape[1], model.hidden_dim)
            for i in range(input_seq.shape[1]):
                if i == 0:
                    out, hidden = model(input_seq[:, [i]].to(device))
                else:
                    out, hidden = model(input_seq[:, [i]].to(device), hidden)
                hiddens[:, i] = hidden.cpu()
            hiddens_all[batch_loc:batch_loc + input_seq.shape[0]] = hiddens

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
