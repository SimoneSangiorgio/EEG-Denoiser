import torch
import math
import random

from plotter import *
from main import *

def eog(type, reference_signal, seconds):

    signal_abs_values = np.abs(reference_signal)
    mean_amplitude = np.mean(signal_abs_values)
    sampling_rate = reference_signal.shape[1] / seconds
    t = torch.linspace(0, seconds, steps = int(seconds * sampling_rate))
    #print(mean_amplitude)

    if type == 'blink':

        frequency_range = (2, 4)
        amplitude_scale = (10, 20)
        amplitude_range = tuple(mean_amplitude * factor for factor in amplitude_scale)

        amplitude = torch.empty(1).uniform_(amplitude_range[0], amplitude_range[1]).item()
        #print(amplitude)
        frequency = torch.empty(1).uniform_(frequency_range[0], frequency_range[1]).item()
        eog_signal = amplitude * torch.sin(2 * math.pi * frequency * t)
        eog_signal = torch.minimum(eog_signal, torch.zeros_like(eog_signal)) #
    
        cycles = int(frequency*seconds)
        #print(cycles)
        #windows = random.sample(range(1, cycles), random.randint(1, round(cycles/8)))
        windows = random.sample(range(1, cycles), random.randint(round(seconds/6), round(seconds/3)))
        #print(windows)

    if type == 'lateral eye movements':

        frequency_range = (0.35, 1)
        amplitude_scale = (5, 10)
        amplitude_range = tuple(mean_amplitude * factor for factor in amplitude_scale)

        amplitude = torch.empty(1).uniform_(amplitude_range[0], amplitude_range[1]).item()
        frequency = torch.empty(1).uniform_(frequency_range[0], frequency_range[1]).item()
        eog_signal = amplitude * torch.sin(2 * math.pi * frequency * t)
        eog_signal = torch.maximum(eog_signal, eog_signal/2)

        cycles = int(frequency*seconds)
        #print(cycles)

        windows = random.sample(range(1, cycles), random.randint(round(seconds/6), round(seconds/3)))
        #print(windows)

    window_total = np.zeros_like(t)

    for window in windows:
        window_signal = np.where((t >= (window/frequency)) & (t <= (window/frequency) + (1/frequency)), 1, 0)
        window_total += window_signal

    eog_signal *= window_total
    
    eog_signal = np.expand_dims(eog_signal.numpy(), axis=0)

    return eog_signal

blink_monosignal = eog('blink', reference_signal, seconds)
factors_blink = [1.0, 1.0, 0.6, 0.6, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

eye_movements_monosignal = eog('lateral eye movements', reference_signal, seconds)
factors_eye_movement = [-1.0, 1.0, 0.0, 0.0, -0.8, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def artefact_signal(artefacts, factors):
    multichannel_artefacts = np.vstack([artefacts * f for f in factors])
    return multichannel_artefacts

blink_signal = artefact_signal(blink_monosignal, factors_blink)
eye_movements_signal = artefact_signal(eye_movements_monosignal, factors_eye_movement)

artefacts_signal = blink_signal
spd_artefacts = signal_spd(artefacts_signal)


#plot_signal(blink_signal, reference_signal_info)
#plot_signal(blink_signal + reference_signal, reference_signal_info)
#plot_spd_matrix(spd_artefacts)
#plot_spd_matrix(spd_artefacts + reference_spd)















