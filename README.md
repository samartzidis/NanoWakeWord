# NanoWakeWord

NanoWakeWord is a minimal C# port of the Python [openWakeWord](https://github.com/dscripka/openWakeWord) wake-word detection engine.

It runs efficiently on any platform supporting .NET Standard 2.0, including the RaspberryPi Zero 2/2W (linux-arm64).


## Dependencies

It has only one external library dependency: the Microsoft.ML.OnnxRuntime.


## Why

I was specifically looking for a simple wake-word library for C# that could run on desktop Windows/Linux systems as well as on Raspberry Pi devices.

Among the options I evaluated, PocketSphinx did not perform adequately, SnowBoy was end-of-life (EOL), lacked source code, and had no runtime support for Windows. Picovoice/Porcupine performed well but came with a frustrating licensing and registration model. It requires constant internet connectivity to regularly validate the license, tracks usage, and may permanently ban users for any form of misuse—whether intentional or accidental. This approach felt more like spyware than a legitimate tool.

Of all the options investigated, openWakeWord turned out to be the best choice. It demonstrated surprisingly good performance and was completely free. Its Python implementation was straightforward and clean, making it easy to port to C#.


## Usage

Build NanoWakeWord and reference the resulting nuget package.
Alternatively, reference the resulting dll library.


### Sample Code

```csharp
var runtime = new WakeWordRuntime(new WakeWordRuntimeConfig { 
    Debug = false, WakeWords = [ new WakeWordConfig { Model = "alexa_v0.1" } ] 
});

using var recorder = PvRecorder.Create(frameLength: 512);
recorder.Start();

Console.WriteLine($"Using recording device: {recorder.SelectedDevice}");

Console.WriteLine("Listening for wake word.");
while (recorder.IsRecording)
{
    var frame = recorder.Read();

    var result = runtime.Process(frame);
    if (result >= 0)
    {
        Console.WriteLine($"Detected wake word at index: #{result}.");
    }
}  
```

## Training Custom Wake-Word Models

NanoWakeWord comes with embedded wake-word models as part of the openWakeWord port: alexa, hey_jarvis, hey_marvin, hey_mycroft.

By following the openWakeWord project [instructions](https://github.com/dscripka/openWakeWord#training-new-models), you can train custom models and use them in NanoWakeWord
as you would normally do in openWakeWord.

### Training Models Locally Using Podman and Python scripts

To facilitate the training process, the *scripts* folder contains Python scripts for automating model training using a Podman container.

Kick off the Podman Linux container (note - you will need to enable Cuda GPU support in Podman):

```
podman run --gpus=all --shm-size=50G -p 127.0.0.1:9000:8080 us-docker.pkg.dev/colab-images/public/runtime
```

Copy the scripts to the /content directory and run the Python scripts in this order:

```
python setup_environment.py
python download_data.py
```
Edit `train_model.py` as needed, then run: 
```
python train_model.py
```


