using System.Security.Cryptography;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text;

namespace NanoWakeWord;

public class WakeWordConfig
{
    /// <summary>
    /// The name of the model file, e.g. "alexa_v0.1".
    /// </summary>
    public string Model { get; set; }

    /// <summary>
    /// Defines the minimum probability level from the model at which a wake word candidate is considered a potential detection.
    /// </summary>
    public float Threshold { get; set; } = 0.5f;

    /// <summary>
    /// Determines how many consecutive threshold crossings (or activations) are needed before the wake word is officially considered "detected."
    /// </summary>
    public int TriggerLevel { get; set; } = 4;

    /// <summary>
    /// Specifies a cooldown period, in frames, during which the model ignores further detections after it has just triggered a wake word event.
    /// </summary>
    public int Refractory { get; set; } = 20;

    public override bool Equals(object obj)
    {
        if (obj is not WakeWordConfig other)
            return false;

        return Model == other.Model &&
               Threshold == other.Threshold &&
               TriggerLevel == other.TriggerLevel &&
               Refractory == other.Refractory;
    }

    // HashCode.Combine unavailable on netstandard2.0
    //public override int GetHashCode()
    //{
    //    return HashCode.Combine(Model, Threshold, TriggerLevel, Refractory);
    //}

    public override int GetHashCode()
    {
        return GetHashCode($"{Model ?? string.Empty}|{Threshold}|{TriggerLevel}|{Refractory}");
    }

    private int GetHashCode(string value)
    {
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(value));
        var hash = 0;
        for (var i = 0; i < Math.Min(4, hashBytes.Length); i++)
            hash = (hash * 31) + hashBytes[i];

        return hash;
    }
}

public class WakeWordRuntimeConfig
{    
    public WakeWordConfig[] WakeWords { get; set; }

    /// <summary>
    /// By adjusting StepFrames, you can control how frequently the pipeline runs the mel spectrogram inference. 
    /// A higher StepFrames means processing a larger audio segment at once, which could reduce how often you run inference but increase latency. 
    /// A lower StepFrames means you'll produce mel features more often, potentially improving responsiveness at the cost of increased computational overhead.
    /// Since the pipeline goes from audio samples ? mel spectrogram ? embeddings ? wake word detection, the size of each processing step sets the overall timing. 
    /// The StepFrames value is a key part of this timing: it controls how quickly new mel frames become available to downstream models (the embedding and wake word models).
    /// </summary>
    public int StepFrames { get; set; } = 4;    

    /// <summary>
    /// Optionally receive frame debug data.
    /// </summary>
    public Action<string, float, bool> DebugAction { get; set; }
}

public class WakeWordRuntime : IDisposable
{
    public const string MelModelPath = "models/melspectrogram.onnx";
    public const string EmbModelPath = "models/embedding_model.onnx";

    private readonly WakeWordRuntimeConfig _settings;
    private readonly InferenceSession _melSession;
    private readonly InferenceSession _embSession;
    private readonly List<InferenceSession> _wwSessions;

    private readonly List<float> _samples = new();
    private readonly List<float> _mels = new();
    private readonly List<List<float>> _features; // Embedding feature buffers for each model

    // Per-model state tracking
    private readonly int[] _activations;
    private readonly int[] _refractoryCounts;

    private readonly int _frameSize;

    // Constants (specific to the ONNX models used in the wake word detection pipeline.)
    private const int ChunkSamples = 1280; // 80 ms at 16kHz
    private const int NumMels = 32; // Vertical resolution of the mel spectrogram.
    private const int EmbWindowSize = 76; // ~775 ms
    private const int EmbStepSize = 8;    // ~80 ms
    private const int EmbFeatures = 96; // Embedding vector size.
    private const int WWFeatures = 16; //  Consecutive embedding vectors count form the input window for the wake word detection model.   

    private int _wakeWordDetectedIndex = -1;

    public WakeWordRuntime(WakeWordRuntimeConfig settings)
    {
        _settings = settings;
        _frameSize = _settings.StepFrames * ChunkSamples;

        // One-off initialization
        var initialized = IsInitialized.Value;

        // Load ONNX models
        _melSession = new InferenceSession(MelModelPath);
        _embSession = new InferenceSession(EmbModelPath);

        _wwSessions = new List<InferenceSession>();
        foreach (var ww in _settings.WakeWords)
        {
            var modelPath = Path.Combine("models", $"{ww.Model}.onnx");
            if (!File.Exists(modelPath))
                throw new InvalidOperationException($"Model file {modelPath} does not exist.");

            _wwSessions.Add(new InferenceSession(modelPath));
        }

        _features = new List<List<float>>(Enumerable.Range(0, _wwSessions.Count).Select(_ => new List<float>()));

        // Initialize per-model activation and refractory counters
        _activations = new int[_wwSessions.Count];
        _refractoryCounts = new int[_wwSessions.Count];
    }

    /// <summary>
    /// One-off initialization.
    /// </summary>
    private static readonly Lazy<bool> IsInitialized = new(() =>
    {
        // Extract models
        typeof(WakeWordRuntime).Assembly.ExtractModels();

        return true;
    });

    /// <summary>
    /// Add a batch of 16-bit PCM audio samples for processing.
    /// Returns the index of the WakeWordModelPaths array that triggered the wake word, or -1 if no detection occurred.
    /// </summary>
    /// <param name="samples">Array of 16-bit PCM audio samples.</param>
    /// <returns>Index of the detected wake word model, or -1 if none detected.</returns>
    public int Process(short[] samples)
    {
        // Convert int16 to float and add to sample buffer
        foreach (var s in samples)
        {
            _samples.Add(s);
        }

        // Run pipeline steps
        AudioToMels();
        MelsToFeatures();
        FeaturesToOutput(out var detectedIndex);

        return detectedIndex;
    }

    private void AudioToMels()
    {
        while (_samples.Count >= _frameSize)
        {
            var frameData = _samples.Take(_frameSize).ToArray();
            _samples.RemoveRange(0, _frameSize);

            var melInput = new DenseTensor<float>(new[] { 1, _frameSize });
            for (int i = 0; i < _frameSize; i++)
                melInput[0, i] = frameData[i];

            var melInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor("input", melInput)
            };

            using var melResults = _melSession.Run(melInputs);
            var melOutput = melResults.First().AsEnumerable<float>().ToArray();

            // Scale mels: (melData[i]/10.0f) + 2.0f
            for (int i = 0; i < melOutput.Length; i++)
            {
                _mels.Add((melOutput[i] / 10.0f) + 2.0f);
            }
        }
    }

    private void MelsToFeatures()
    {
        int melFrames = _mels.Count / NumMels;

        while (melFrames >= EmbWindowSize)
        {
            var windowData = _mels.Take(EmbWindowSize * NumMels).ToArray();

            var embInput = new DenseTensor<float>(new[] { 1, EmbWindowSize, NumMels, 1 });
            for (int f = 0; f < EmbWindowSize; f++)
            {
                for (int m = 0; m < NumMels; m++)
                {
                    embInput[0, f, m, 0] = windowData[f * NumMels + m];
                }
            }

            var inputName = _embSession.InputMetadata.First().Key;
            var embInputs = new List<NamedOnnxValue> {
                NamedOnnxValue.CreateFromTensor(inputName, embInput)
            };

            using var embResults = _embSession.Run(embInputs);
            var embOutput = embResults.First().AsEnumerable<float>().ToArray();

            // Add embeddings to all model feature buffers
            foreach (var featBuf in _features)
            {
                featBuf.AddRange(embOutput);
            }

            // Advance by EmbStepSize frames
            _mels.RemoveRange(0, EmbStepSize * NumMels);
            melFrames = _mels.Count / NumMels;
        }
    }

    private void FeaturesToOutput(out int detectedIndex)
    {
        detectedIndex = -1;

        for (int i = 0; i < _wwSessions.Count; i++)
        {
            var featBuf = _features[i];
            int numBufferedFeatures = featBuf.Count / EmbFeatures;

            //A model enters its refractory period immediately after it successfully triggers a wake word detection.
            // If the model is in refractory, count down.
            if (_refractoryCounts[i] > 0)
            {
                _refractoryCounts[i]--;
            }

            while (numBufferedFeatures >= WWFeatures)
            {
                // Take WWFeatures embeddings
                var wwData = featBuf.Take(WWFeatures * EmbFeatures).ToArray();
                // Remove one embedding step after inference to create a sliding window
                featBuf.RemoveRange(0, EmbFeatures);

                var wwInput = new DenseTensor<float>(new[] { 1, WWFeatures, EmbFeatures });
                for (int f = 0; f < WWFeatures; f++)
                {
                    for (int e = 0; e < EmbFeatures; e++)
                    {
                        wwInput[0, f, e] = wwData[f * EmbFeatures + e];
                    }
                }

                var inputName = _wwSessions[i].InputMetadata.First().Key;
                var wwInputs = new List<NamedOnnxValue> {
                    NamedOnnxValue.CreateFromTensor(inputName, wwInput)
                };

                using var wwResults = _wwSessions[i].Run(wwInputs);
                var wwOutput = wwResults.First().AsEnumerable<float>().ToArray();

                float threshold = _settings.WakeWords[i].Threshold;

                foreach (var probability in wwOutput)
                {
                    var model = _settings.WakeWords[i].Model;

                    _settings.DebugAction?.Invoke(model, probability, false);

                    // If we are in refractory period, ignore detection attempts
                    if (_refractoryCounts[i] > 0)
                    {
                        // We still remove embeddings and continue, but don't trigger.
                        continue;
                    }

                    if (probability > threshold)
                    {
                        _activations[i]++;
                        if (_activations[i] >= _settings.WakeWords[i].TriggerLevel)
                        {
                            // Trigger detected
                            _settings.DebugAction?.Invoke(model, probability, true);

                            detectedIndex = i;

                            _activations[i] = 0;
                            _refractoryCounts[i] = _settings.WakeWords[i].Refractory;
                        }
                    }
                    else
                    {
                        // If below threshold, decay activation towards zero
                        if (_activations[i] > 0)
                            _activations[i] = Math.Max(0, _activations[i] - 1);
                        else
                            _activations[i] = Math.Min(0, _activations[i] + 1);
                    }
                }

                numBufferedFeatures = featBuf.Count / EmbFeatures;
            }
        }
    }

    /// <summary>
    /// Dispose of the InferenceSession instances.
    /// </summary>
    public void Dispose()
    {
        _melSession?.Dispose();
        _embSession?.Dispose();
        foreach (var session in _wwSessions)
            session.Dispose();
    }
}