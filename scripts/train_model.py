# Imports
import os
import shutil
import numpy as np
import torch
import sys
from pathlib import Path
import yaml
import datasets
import scipy.io.wavfile
from tqdm import tqdm

target_phrase = "queen of lights" # This will generate the "queen of lights" custom wake word
number_of_examples = 1000 
number_of_training_steps = 10000  
false_activation_penalty = 1500  

# Load default YAML config file for training
config = yaml.load(open("openwakeword/examples/custom_model.yml", 'r').read(), yaml.Loader)

config["target_phrase"] = [ target_phrase ]
config["model_name"] = target_phrase.replace(" ", "_")
config["n_samples"] = number_of_examples
config["n_samples_val"] = max(500, number_of_examples//10)
config["steps"] = number_of_training_steps
config["target_accuracy"] = 0.6
config["target_recall"] = 0.25

config["background_paths"] = ['./audioset_16k', './fma']  # multiple background datasets are supported
config["false_positive_validation_data_path"] = "validation_set_features.npy"
config["feature_data_files"] = {"ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"}

with open('my_model.yaml', 'w') as file:
    documents = yaml.dump(config, file)


# delete "model_name" folder if exists
model_dir = config[ "output_dir" ]
if os.path.exists(model_dir) and os.path.isdir(model_dir):
    shutil.rmtree(model_dir)
    print(f"The directory '{model_dir}' was successfully deleted.")
else:
    print(f"No directory named '{model_dir}' exists, so nothing was deleted.")

os.system("python3 ./openwakeword/openwakeword/train.py --training_config my_model.yaml --generate_clips")
os.system("python3 ./openwakeword/openwakeword/train.py --training_config my_model.yaml --augment_clips")
os.system("python3 ./openwakeword/openwakeword/train.py --training_config my_model.yaml --train_model")






