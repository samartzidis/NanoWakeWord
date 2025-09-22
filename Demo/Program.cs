using NanoWakeWord;
using Pv;

namespace Demo;

internal class Program
{
    static void Main(string[] args)
    {
        var runtime = new WakeWordRuntime(new WakeWordRuntimeConfig { 
            Debug = true, 
            WakeWords = [ new WakeWordConfig { Model = "hey_marvin_v0.1", Threshold = 0.9f } ] 
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
    }
}