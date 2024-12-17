import os
import subprocess

def run_command(cmd):
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result

# Check and install piper-sample-generator
if not os.path.isdir("piper-sample-generator"):
    run_command(["git", "clone", "https://github.com/rhasspy/piper-sample-generator"])

models_dir = "piper-sample-generator/models"
os.makedirs(models_dir, exist_ok=True)

pt_file = os.path.join(models_dir, "en_US-libritts_r-medium.pt")
if not os.path.isfile(pt_file):
    run_command(["wget", "-O", pt_file, 
                 "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"])

# Install required Python packages
run_command(["pip", "install", "piper-phonemize"])
run_command(["pip", "install", "webrtcvad"])

# Check and install openwakeword
if not os.path.isdir("openwakeword"):
    run_command(["git", "clone", "https://github.com/dscripka/openwakeword"])

# Install openwakeword in editable mode
run_command(["pip", "install", "-e", "./openwakeword"])

# Change directory to openwakeword
os.chdir("openwakeword")

# Install other dependencies
run_command(["pip", "install", "mutagen==1.47.0"])
run_command(["pip", "install", "torchinfo==1.8.0"])
run_command(["pip", "install", "torchmetrics==1.2.0"])
run_command(["pip", "install", "speechbrain==0.5.14"])
run_command(["pip", "install", "audiomentations==0.33.0"])
run_command(["pip", "install", "torch-audiomentations==0.11.0"])
run_command(["pip", "install", "acoustics==0.2.6"])
run_command(["pip", "uninstall", "tensorflow", "-y"])
run_command(["pip", "install", "tensorflow-cpu==2.8.1"])
run_command(["pip", "install", "protobuf==3.20.3"])
run_command(["pip", "install", "tensorflow_probability==0.16.0"])
run_command(["pip", "install", "onnx_tf==1.10.0"])
run_command(["pip", "install", "pronouncing==0.2.0"])
run_command(["pip", "install", "datasets==2.14.6"])
run_command(["pip", "install", "deep-phonemizer==0.0.19"])

# Create directories and download required models
models_path = "./openwakeword/openwakeword/resources/models"
os.makedirs(models_path, exist_ok=True)

embedding_model_onnx = os.path.join(models_path, "embedding_model.onnx")
if not os.path.isfile(embedding_model_onnx):
    run_command([
        "wget",
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
        "-O", embedding_model_onnx
    ])

embedding_model_tflite = os.path.join(models_path, "embedding_model.tflite")
if not os.path.isfile(embedding_model_tflite):
    run_command([
        "wget",
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
        "-O", embedding_model_tflite
    ])

melspectrogram_onnx = os.path.join(models_path, "melspectrogram.onnx")
if not os.path.isfile(melspectrogram_onnx):
    run_command([
        "wget",
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
        "-O", melspectrogram_onnx
    ])

melspectrogram_tflite = os.path.join(models_path, "melspectrogram.tflite")
if not os.path.isfile(melspectrogram_tflite):
    run_command([
        "wget",
        "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
        "-O", melspectrogram_tflite
    ])

print("Environment setup complete.")
