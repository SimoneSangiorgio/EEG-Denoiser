import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_spd_matrix(spd_matrix):
    plt.figure(figsize=(10, 8))
    plt.title("SPD Matrix Heatmap")
    sns.heatmap(spd_matrix.squeeze(0), fmt=".2f", cmap="coolwarm", cbar=True, square=True)     ###  
    plt.show()

def plot_three_spd_matrices(spd_matrix1, spd_matrix2, spd_matrix3, titles=None):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    if titles is None:
        titles = ["Matrix 1", "Matrix 2", "Matrix 3"]

    sns.heatmap(spd_matrix1, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=axes[0])
    axes[0].set_title(titles[0])

    sns.heatmap(spd_matrix2, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=axes[1])
    axes[1].set_title(titles[1])

    sns.heatmap(spd_matrix3, fmt=".2f", cmap="coolwarm", cbar=True, square=True, ax=axes[2])
    axes[2].set_title(titles[2])

    plt.tight_layout()
    plt.show()

def plot_loss(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_signal(signal, signal_info, n_channels=None):
    seconds = signal_info[0]
    sample_rate = signal_info[2]
    channels = signal_info[4]
    num_samples = int(sample_rate * int(seconds))
    time = np.linspace(0, int(seconds), num_samples, endpoint=False)

    signal_std = np.std(signal, axis=1, keepdims=True)
    signal_std[signal_std == 0] = 1
    
    signal_normalized = (signal - np.mean(signal, axis=1, keepdims=True)) / signal_std
    #signal_normalized = signal ###

    plt.figure(figsize=(16, 9))

    num_total_channels_in_data = signal_normalized.shape[0]


    if n_channels is None:
        num_channels_to_display = num_total_channels_in_data
    else:
        num_channels_to_display = min(n_channels, num_total_channels_in_data)


    for i in range(num_channels_to_display):

        channel_data = signal_normalized[i]
        channel_label = channels[i]
        
        stacking_factor = 5  
        vertical_offset = (num_channels_to_display - 1 - i) * stacking_factor
        
        plt.plot(time[:len(channel_data)], channel_data + vertical_offset, label=f'Channel {channel_label}')

    plt.title("Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (V)")
    
    #plt.yticks([]) 
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0)) 
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    plt.show()