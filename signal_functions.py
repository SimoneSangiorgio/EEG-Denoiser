import torch.nn as nn
import numpy as np
import logging
import torch
import mne

logging.getLogger('mne').setLevel(logging.ERROR)

def edf_reader(edf_file):
    raw_signal = mne.io.read_raw_edf(edf_file, preload=True)
    print(str(raw_signal).split(' ')[2].split('.')[0])
    print(str(raw_signal).split(',')[-3].split(' ')[1]+" channels")
    print(raw_signal.info['ch_names'])
    print(str(raw_signal).split(',')[-3].split(' ')[3]+" samples")
    print(str(raw_signal).split(',')[-3].split(' ')[4].split('(')[1]+" seconds")
    sample_rate = float(str(raw_signal).split(',')[-3].split(' ')[3])/float(str(raw_signal).split(',')[-3].split(' ')[4].split('(')[1])
    print(str(sample_rate)+" Hz")

def edf_signal(edf_file, seconds = None, useless_channels = None):

    raw_signal = mne.io.read_raw_edf(edf_file, preload=True)
    signal_matrix = raw_signal.get_data()
    info = raw_signal.info
    sample_rate = int(info['sfreq']) 

    if seconds is None:
        signal_matrix = signal_matrix    
    else:
        signal_matrix = signal_matrix[:, :sample_rate*seconds]

    if useless_channels is None:
        signal_matrix = signal_matrix
    else:
        signal_matrix = np.delete(signal_matrix, [x - 1 for x in useless_channels], axis=0)    
    
    return signal_matrix

def edf_info(edf_file, seconds = None, useless_channels = None):

    raw_signal = mne.io.read_raw_edf(edf_file, preload=True)
    info = raw_signal.info
    sample_rate = int(info['sfreq']) 

    if seconds is None:
        seconds = float(str(raw_signal).split(',')[-3].split(' ')[4].split('(')[1])
    else:
        seconds = seconds

    if useless_channels is None:
        channels = info['ch_names']

    else:
        channels = [ch for i, ch in enumerate(info['ch_names']) if i not in [x - 1 for x in useless_channels]]

    samples = seconds*sample_rate
    n_channels = len(channels)
    
    return seconds, int(samples), sample_rate, n_channels, channels

class signal2spd(nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device('cpu')
    def forward(self, x):
        
        x = x.squeeze(0)
        mean = x.mean(axis=-1).unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x - mean
        cov = x@x.permute(0, 2, 1)
        cov = cov.to(self.dev)
        cov = cov/(x.shape[-1]-1)
        tra = cov.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        tra = tra.view(-1, 1, 1)
        cov /= tra
        identity = torch.eye(cov.shape[-1], cov.shape[-1], device=self.dev).to(self.dev).repeat(x.shape[0], 1, 1)
        cov = cov+(1e-5*identity)

        #cov = cov.squeeze(0) ###

        return cov 
    
def edf_spd(edf_file, seconds = None, useless_channels = None):
    
    signal = edf_signal(edf_file, seconds, useless_channels)
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    spd_converter = signal2spd()
    spd_matrix = spd_converter(signal_tensor)

    return spd_matrix

def signal_spd(signal):
    
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    spd_converter = signal2spd()
    spd_matrix = spd_converter(signal_tensor)

    return spd_matrix