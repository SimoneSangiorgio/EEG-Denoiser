from pathinator import *
from signal_functions import *

#MULTI-CHANNEL SIGNAL PARAMETERS

seconds = 60
useless_channels = [21, 20]

#REFERENCES

#edf_reader(edf_file)

reference_edf_file = dataset_files[29]
reference_signal = edf_signal(reference_edf_file, seconds, useless_channels)
reference_signal_info = edf_info(reference_edf_file, seconds, useless_channels)
reference_spd = edf_spd(reference_edf_file, seconds, useless_channels)

#TRAIN PARAMETERS

batch_size = 9
time_size = 128  
lr = 0.0025
epochs = 80
T = 250  #tot time steps

#EVALUATION PARAMETERES

model_path = models_folder / "best_model_1.pth"






