import lightning as L
import numpy as np
import torch
from torch import nn
from omegaconf import OmegaConf


def make_matrix(d, eig_lower_bound=1e-1):
    eigvals_real = - torch.rand(int(d/2))*(1e-1)
    freqs = torch.rand(int(d/2))*2*np.pi
    eigvals_diag = torch.zeros(d, d)
    for i in range(0, d, 2):
        eigvals_diag[i, i] = eigvals_real[int(i/2)]
        eigvals_diag[i + 1, i + 1] = eigvals_real[int(i/2)]
        eigvals_diag[i, i+1] = -freqs[int(i/2)]
        eigvals_diag[i+1, i] = freqs[int(i/2)]
    eigvecs = torch.linalg.qr(torch.randn(d, d))[0]
    M = torch.matrix_exp(eigvecs @ eigvals_diag @ torch.linalg.pinv(eigvecs))

    return M


class BiologicalRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dt, tau, bias=True, eig_lower_bound=1e-1, enforce_fixation=False, init_mode='learned'):
        super(BiologicalRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        output_dim = output_dim + (2 if enforce_fixation else 0)
        self.output_dim = output_dim
        self.dt = dt
        self.tau = tau
        self.eig_lower_bound = eig_lower_bound
        self.activation = nn.ELU()

        W_hi = torch.randn(hidden_dim, input_dim + input_dim + 2 + (1 if enforce_fixation else 0))
        W_hi /= torch.linalg.norm(W_hi, ord=2, dim=(-2, -1))
        self.W_hi = nn.Parameter(W_hi)

        M = torch.zeros(hidden_dim, hidden_dim)
        M[:int(hidden_dim/2), :int(hidden_dim/2)] = make_matrix(int(hidden_dim/2), eig_lower_bound=eig_lower_bound)
        M[int(hidden_dim/2):, int(hidden_dim/2):] = make_matrix(int(hidden_dim/2), eig_lower_bound=eig_lower_bound)

        W_inter1 = torch.randn(int(hidden_dim/2), int(hidden_dim/2))
        W_inter1 /= torch.linalg.norm(W_inter1, ord=2, dim=(-2, -1))
        W_inter1 *= 0.05
        M[:int(hidden_dim/2), int(hidden_dim/2):] = W_inter1
        W_inter2 = torch.randn(int(hidden_dim/2), int(hidden_dim/2))
        W_inter2 /= torch.linalg.norm(W_inter2, ord=2, dim=(-2, -1))
        W_inter2 *= 0.05
        M[int(hidden_dim/2):, :int(hidden_dim/2)] = W_inter2

        self.W_hh = nn.Parameter(M)

        self.b = nn.Parameter(torch.zeros(hidden_dim))

        self.input_mask = torch.zeros(self.W_hi.shape)
        self.input_mask[:int(hidden_dim/2)] = 1

        W_oh = torch.randn(output_dim, hidden_dim)
        W_oh /= torch.linalg.norm(W_oh, ord=2, dim=(-2, -1))
        self.W_oh = nn.Parameter(W_oh)

        self.output_mask = torch.zeros(self.W_oh.shape)
        self.output_mask[:, int(hidden_dim/2):] = 1

        self.init_mode = init_mode
        if init_mode == 'learned':
            self.hidden_init = nn.Parameter(torch.randn(self.hidden_dim))

    def forward(self, x, hidden=None):
        """
        Given an input sequence x from time 0:t
        """
        h0 = hidden
        if h0 is None:
            if len(x.shape) == 3:
                if self.init_mode == 'learned':
                    h0 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
                    h0[:] = self.hidden_init
                elif self.init_mode == 'random':
                    h0 = torch.randn(x.size(0), self.hidden_dim).to(x.device)
            else:
                if self.init_mode == 'learned':
                    h0 = torch.zeros(self.hidden_dim).to(x.device)
                    h0 = self.hidden_init
                elif self.init_mode == 'random':
                    h0 = torch.randn(self.hidden_dim).to(x.device)

        h = h0

        squeeze = 0
        if len(x.shape) == 2:
            squeeze = 1
            x = x.unsqueeze(0)
        elif len(x.shape) == 1:
            squeeze = 2
            x = x.unsqueeze(0).unsqueeze(0)

        self.input_mask = self.input_mask.type(x.dtype).to(x.device)
        self.output_mask = self.output_mask.type(x.dtype).to(x.device)

        outs = []
        for i in range(x.size(1)):
            h = h + (self.dt/self.tau)*(-h + ((self.W_hi * self.input_mask) @ x[:, i].unsqueeze(-1)).squeeze(-1) + (self.W_hh @ self.activation(h).unsqueeze(-1)).squeeze(-1) + self.b)
            outs.append(((self.W_oh * self.output_mask) @ h.unsqueeze(-1)).squeeze(-1))
        outs = torch.stack(outs, dim=1)
        if squeeze == 1:
            outs = outs.squeeze(0)
        elif squeeze == 2:
            outs = outs.squeeze(0).squeeze(0)

        return outs, h


class LitBiologicalRNN(L.LightningModule):
    def __init__(self, model, save_dir=None, learning_rate=1e-4, enforce_fixation=False):
        super().__init__()
        self.model = model
        self.save_dir = save_dir

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        self.enforce_fixation = enforce_fixation

    def model_step(self, batch, batch_idx, dataloader_idx=0, all_metrics=False, generate=None):

        input_seq, labels = batch
        out, hidden = self.model(input_seq)
        loss = self.criterion(out[:, -labels.shape[-1]:, :self.model.input_dim].transpose(-2, -1), labels)
        if self.enforce_fixation:
            fix_loss = self.criterion(out[:, :, self.model.input_dim:].transpose(-2, -1), input_seq[:, :, -1].type(torch.LongTensor).to(out.device))
            loss += fix_loss
            fix_accuracy = torch.sum(nn.Softmax(dim=-1)(out[:, :, self.model.input_dim:]).argmax(dim=-1) == input_seq[:, :, -1])/(input_seq.shape[0]*input_seq.shape[1])
        accuracy = torch.sum(nn.Softmax(dim=-1)(out[:, -labels.shape[-1]:, :self.model.input_dim]).argmax(dim=-1) == labels)/(labels.shape[0]*labels.shape[1])

        if self.enforce_fixation:
            return {'accuracy': accuracy, 'loss': loss, 'fix_accuracy': fix_accuracy}
        else:
            return {'accuracy': accuracy, 'loss': loss}

    def training_step(self, batch, batch_idx):
        ret = self.model_step(batch, batch_idx)

        self.log("train_loss", ret['loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_accuracy", ret['accuracy'], on_step=True, on_epoch=True, sync_dist=True)

        if self.enforce_fixation:
            self.log("train_fix_accuracy", ret['fix_accuracy'], on_step=True, on_epoch=True, sync_dist=True)

        return ret['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ret = self.model_step(batch, batch_idx)

        self.log("val_loss", ret['loss'], sync_dist=True)
        self.log("val_accuracy", ret['accuracy'], sync_dist=True)

        if self.enforce_fixation:
            self.log("val_fix_accuracy", ret['fix_accuracy'], sync_dist=True)

        return ret['loss']

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss = self.model_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
