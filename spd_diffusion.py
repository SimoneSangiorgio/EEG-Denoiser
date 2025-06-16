import torch
from torch import optim
import logging
import warnings
import numpy as np
import random
from pyriemann.datasets import sample_gaussian_spd

from torch.utils.data import DataLoader

from signal_functions import *
from artefacts import blink_signal, eye_movements_signal
from spd_functions import *
from plotter import *
from main import *
from spd_unet import SPD_UNET
from dataloader import SPDataset

warnings.filterwarnings("ignore")

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# X: SPD matrix
# t: time steps
# T: tot time steps

artefacts_signal = blink_signal
spd_artefacts = signal_spd(artefacts_signal)

def forward_process(X_0, t, T, noise):

    beta = torch.linspace(1e-4, 0.08, T)
    alpha = torch.sqrt(1 - beta**2)

    alpha_bar = torch.cumprod(alpha, dim=0)
    beta_bar = torch.sqrt(1-alpha_bar**2)

    alpha_bar_t = alpha_bar[t].unsqueeze(1).unsqueeze(2)
    beta_bar_t = beta_bar[t].unsqueeze(1).unsqueeze(2) 

    if noise == "gaussian":
        mean = np.eye(X_0.shape[1])
        epsilon = torch.tensor(sample_gaussian_spd(X_0.shape[0], mean, 1, n_jobs=6)) # noise

    elif noise == "real":
        epsilon = spd_artefacts.double()

    X_t = spd_plus(spd_mul(X_0, alpha_bar_t), spd_mul(epsilon, beta_bar_t)) # noisy matrix = spd matrix + noise

    return X_t


def snr_loss(epsilon_true, epsilon_pred, eps=1e-8):
    signal_power = torch.sum(epsilon_true ** 2, dim=[1,2]) 
    noise_power = torch.sum((epsilon_true - epsilon_pred) ** 2, dim=[1,2]) + eps

    snr = 10 * torch.log10(signal_power / noise_power)
    return -snr.mean() 


def loss(model, X_0, T):
    mean = np.eye(X_0.shape[1]) 
    epsilon = torch.tensor(sample_gaussian_spd(X_0.shape[0], mean, 1, n_jobs=6))
    t = torch.randint(1, T, (X_0.shape[0],))
    X_t = forward_process(X_0, t, T, "gaussian")
    epsilon_theta = model(X_t, t) # predicted noise
    loss = spd_dis(epsilon, epsilon_theta).mean() #+ snr_loss(epsilon, epsilon_theta)
    return loss


def train(dataset_files, batch_size, time_size, lr, epochs, T):

    edf_file = dataset_files[0]
    spd_matrix = edf_spd(edf_file, seconds, useless_channels)

    spd_dataset = SPDataset(dataset_files, seconds, useless_channels)
    dataloader = DataLoader(spd_dataset, batch_size, shuffle=True, num_workers=4)

    model = SPD_UNET(spd_matrix.shape[1], time_size)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    loss_history = []
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, X_0 in enumerate(dataloader):
            optimizer.zero_grad()
            loss_val = loss(model, X_0, T)
            loss_val.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss_val.item()
            loss_history.append(loss_val.item())

        avg_loss = running_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch} - Average Loss: {avg_loss:.6f} - LR: {current_lr:.6f}")

        # ReduceLROnPlateau step
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), models_folder / "best_model.pth")
            print(f"Saved new best model at epoch {epoch} with loss {best_loss:.6f}")

        # Save model checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(), models_folder / f"model_epoch_{epoch}.pth")

    plot_loss(loss_history)
    return model


if __name__ == '__main__':
   train(dataset_files, batch_size, time_size, lr, epochs, T)


