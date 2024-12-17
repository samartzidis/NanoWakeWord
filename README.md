# NanoWakeWord

NanoWakeWord is a minimal C# port of the Python [openWakeWord](https://github.com/dscripka/openWakeWord) wake-word detection engine.

It runs efficiently on any platform supporting .NET Standard 2.0, including the RaspberryPi Zero 2/2W (linux-arm64).


## Dependencies

It has only one external library dependency: the Microsoft.ML.OnnxRuntime.


## Why

I was specifically looking for a simple wake-word library for C# that would run on desktop Windows/Linux and also on Raspberry Pi devices.

Amongst the investigated options, *PocketSphinx* was not performing adequately enough, *SnowBoy* was EOL, 
without source code and no Windows runtime support. *Picovoice/Porcupine* was performing well but had an annoying licensing and 
registration model. It tracks down library usage (over the internet) and potentially bans you for life if their platform detects any deliberate 
or accidental misuse. I was also unsure about how their licensing engine would behave if there was no internet connectivity for quite some time.

From the above investigated options, *openWakeWord* was the best choice as it demonstrated surprisingly good performance and it was free. 
Since its Python implementation was clear and simple enough - it was ported over to C#.


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

NanoWakeWord comes with 5 embedded wake-word models as part of the openWakeWord port: alexa, hey_anna, hey_jarvis, hey_marvin, hey_mycroft.

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


