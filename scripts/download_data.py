# Imports
import os
import numpy as np
import torch
import sys
from pathlib import Path
import yaml
import datasets
import scipy.io.wavfile
from tqdm import tqdm

# Function to ensure a directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

# Download room impulse responses collected by MIT only if the directory does not exist
# https://mcdermottlab.mit.edu/Reverb/IR_Survey.html
output_dir = "./mit_rirs"
if not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0:
    ensure_dir(output_dir)
    try:
        rir_dataset = datasets.load_dataset(
            "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
        )
        for row in tqdm(rir_dataset, desc="Downloading MIT RIRs"):
            name = row["audio"]["path"].split("/")[-1]
            scipy.io.wavfile.write(
                os.path.join(output_dir, name),
                16000,
                (row["audio"]["array"] * 32767).astype(np.int16),
            )
    except Exception as e:
        print(f"Error downloading or processing MIT RIRs: {e}")
else:
    print("MIT RIRs directory already exists, skipping download.")

# Download noise and background audio
# Audioset Dataset
ensure_dir("audioset")


fname = "bal_train09.tar"
out_dir = f"audioset/{fname}"
link = f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{fname}"

# Only download if the file doesn't already exist
if not os.path.exists(out_dir):
    try:
        os.system(f"wget -O {out_dir} {link}")
        os.system(f"tar -xvf {out_dir} -C audioset")
    except Exception as e:
        print(f"Error downloading or extracting AudioSet data: {e}")
else:
    print(f"{fname} already exists, skipping download and extraction.")

output_dir = "./audioset_16k"
ensure_dir(output_dir)

try:
    audioset_files = [str(i) for i in Path("audioset/audio").glob("**/*.flac")]
    audioset_dataset = datasets.Dataset.from_dict({"audio": audioset_files})
    audioset_dataset = audioset_dataset.cast_column(
        "audio", datasets.Audio(sampling_rate=16000)
    )

    for row in tqdm(audioset_dataset, desc="Processing AudioSet files"):
        name = row["audio"]["path"].split("/")[-1].replace(".flac", ".wav")
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
except Exception as e:
    print(f"Error processing AudioSet files: {e}")

# Free Music Archive dataset
output_dir = "./fma"
ensure_dir(output_dir)

try:
    fma_dataset = datasets.load_dataset(
        "rudraml/fma", name="small", split="train", streaming=True
    )
    fma_dataset = iter(
        fma_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    )

    n_hours = 1  # Use only 1 hour of clips for this example
    clips_to_process = n_hours * 3600 // 30  # 30-second clips
    for _ in tqdm(range(clips_to_process), desc="Processing FMA dataset"):
        row = next(fma_dataset)
        name = row["audio"]["path"].split("/")[-1].replace(".mp3", ".wav")
        scipy.io.wavfile.write(
            os.path.join(output_dir, name),
            16000,
            (row["audio"]["array"] * 32767).astype(np.int16),
        )
except StopIteration:
    print("End of FMA dataset reached.")
except Exception as e:
    print(f"Error processing FMA dataset: {e}")

# Download pre-computed openWakeWord features for training and validation
openwakeword_features_file = "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
validation_features_file = "validation_set_features.npy"

# Only download if they do not already exist
if not os.path.exists(openwakeword_features_file):
    try:
        os.system(
            f"wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/{openwakeword_features_file}"
        )
    except Exception as e:
        print(f"Error downloading {openwakeword_features_file}: {e}")
else:
    print(f"{openwakeword_features_file} already exists, skipping download.")

if not os.path.exists(validation_features_file):
    try:
        os.system(
            f"wget https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/{validation_features_file}"
        )
    except Exception as e:
        print(f"Error downloading {validation_features_file}: {e}")
else:
    print(f"{validation_features_file} already exists, skipping download.")
