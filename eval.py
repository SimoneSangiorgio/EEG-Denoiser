import torch
import random
import logging
import warnings
import numpy as np
from torch.utils.data import DataLoader
from pyriemann.datasets import sample_gaussian_spd

# Local imports (assuming these files exist and are correct)
from signal_functions import edf_spd
from spd_functions import spd_plus, spd_mul, spd_minus, spd_dis, spd_snr
from plotter import plot_three_spd_matrices
from spd_unet import SPD_UNET
from dataloader import SPDataset
from main import dataset_files, model_path, seconds, useless_channels, batch_size, time_size, T

# --- Configuration & Setup ---
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# --- Function Definitions ---

def forward_process(X_0, t, T, alpha_bar, beta_bar):
    """Adds noise to the SPD matrix."""
    alpha_bar_t = alpha_bar[t].unsqueeze(1).unsqueeze(2)
    beta_bar_t = beta_bar[t].unsqueeze(1).unsqueeze(2)

    mean = np.eye(X_0.shape[1])
    # Use n_jobs=1 if you continue to see issues, but the main guard should fix it.
    epsilon = torch.tensor(sample_gaussian_spd(X_0.shape[0], mean, 1, n_jobs=6))

    X_t = spd_plus(spd_mul(X_0, alpha_bar_t), spd_mul(epsilon, beta_bar_t))

    return X_t, epsilon


def eval_loss(model, dataloader, T, alpha_bar, beta_bar):
    """Calculates the average evaluation loss over a dataset."""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for X_0 in dataloader:
            batch_size_current = X_0.shape[0]
            t = torch.randint(1, T, (batch_size_current,))

            X_t, epsilon_true = forward_process(X_0, t, T, alpha_bar, beta_bar)
            epsilon_pred = model(X_t.double(), t)
            loss_val = spd_dis(epsilon_true, epsilon_pred).mean().item()

            total_loss += loss_val
            count += 1
            if count % 10 == 0:
                print(f"  Processed {count} batches...")

    avg_loss = total_loss / count if count > 0 else 0
    print(f"Final Average Evaluation Loss: {avg_loss:.6f}")
    return avg_loss


def main():
    """Main execution function to run the evaluation."""
    # --- Single Sample Denoising Demo ---
    print("--- 1. Running Single Sample Denoising Demo ---")
    
    # Diffusion schedule parameters
    beta = torch.linspace(1e-4, 0.08, T)
    alpha = torch.sqrt(1 - beta**2)
    alpha_bar = torch.cumprod(alpha, dim=0)
    beta_bar = torch.sqrt(1 - alpha_bar**2)
    
    # Load a random file for the demo
    edf_file = random.choice(dataset_files)
    print(f"Loading data from: {edf_file.name}")
    X_0 = edf_spd(edf_file, seconds, useless_channels)

    # Load pre-trained model
    print(f"Loading model from: {model_path}")
    model = SPD_UNET(X_0.shape[1], time_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Create a highly noisy sample
    noise_quantity = T - 1  # Use a high noise level for a clear demo
    t_test = torch.tensor([noise_quantity])
    X_noisy, epsilon_true = forward_process(X_0.double(), t_test, T, alpha_bar, beta_bar)

    # Denoise the sample
    with torch.no_grad():
        epsilon_pred = model(X_noisy, t_test)
    
    alpha_bar_t = alpha_bar[t_test].unsqueeze(1).unsqueeze(2).double()
    beta_bar_t = beta_bar[t_test].unsqueeze(1).unsqueeze(2).double()

    # Reconstruct the original matrix from the prediction
    scaled_epsilon_pred = spd_mul(epsilon_pred, beta_bar_t)
    intermediate_matrix = spd_minus(X_noisy, scaled_epsilon_pred)
    X_denoised = spd_mul(intermediate_matrix, 1 / alpha_bar_t)

    # Print metrics
    dist_noisy = spd_dis(X_0.double(), X_noisy.double()).item()
    dist_denoised = spd_dis(X_0.double(), X_denoised.double()).item()
    snr_noisy = spd_snr(X_0.double(), X_noisy.double())
    snr_denoised = spd_snr(X_0.double(), X_denoised.double())

    print("\n--- Metrics for Single Sample ---")
    print(f"Geodesic Distance (Original vs Noisy):   {dist_noisy:.4f}")
    print(f"Geodesic Distance (Original vs Denoised): {dist_denoised:.4f}")
    print(f"\nSNR (Original vs Noisy):                  {snr_noisy:.4f}")
    print(f"SNR (Original vs Denoised):               {snr_denoised:.4f}")
    print(f"\nGeodesic Distance Reduction:              {dist_noisy - dist_denoised:.4f}")
    print(f"SNR Increase:                             {snr_denoised - snr_noisy:.4f}")

    # Display the matrices
    plot_three_spd_matrices(
        X_0[0],
        X_noisy[0].detach().numpy(),
        X_denoised[0].detach().numpy(),
        titles=["Original SPD", "Noisy SPD", "Denoised SPD"]
    )

    # --- Full Dataset Evaluation ---
    print("\n--- 2. Running Full Dataset Evaluation ---")
    spd_dataset = SPDataset(dataset_files, seconds, useless_channels)
    dataloader = DataLoader(spd_dataset, batch_size, shuffle=False, num_workers=4)
    
    #eval_loss(model, dataloader, T, alpha_bar, beta_bar)


# This is the crucial part!
# It ensures the 'main' function only runs when the script is executed directly.
if __name__ == '__main__':
    main()