using System.Reflection;

namespace NanoWakeWord;

public static class WakeWordUtil
{
    public static void ExtractModels(this Assembly asm, bool overwrite = false)
    {
        var baseDir = AppContext.BaseDirectory;
        var modelsDir = Path.Combine(baseDir, "models");
        if (!Directory.Exists(modelsDir))
            Directory.CreateDirectory(modelsDir);
        var names = asm.GetResourceNames("Resources.models");
        foreach (var name in names)
        {
            var outputPath = Path.Combine(modelsDir, name);
            if (File.Exists(outputPath) && !overwrite)
                continue;

            var bytes = asm.GetBinaryResource(name, "Resources.models");
            File.WriteAllBytes(outputPath, bytes);
        }
    }

    public static IEnumerable<string> GetModels()
    {
        var baseDir = AppContext.BaseDirectory;
        var modelsDir = Path.Combine(baseDir, "models");

        if (!Directory.Exists(modelsDir))
            return Enumerable.Empty<string>();

        return Directory.EnumerateFiles(modelsDir)
            .Select(Path.GetFileNameWithoutExtension)
            .Where(name => !string.Equals(name, Path.GetFileNameWithoutExtension(WakeWordRuntime.MelModelPath), StringComparison.OrdinalIgnoreCase) &&
                           !string.Equals(name, Path.GetFileNameWithoutExtension(WakeWordRuntime.EmbModelPath), StringComparison.OrdinalIgnoreCase));
    }

    private static byte[] GetBinaryResource(this Assembly asm, string resourceName, string containerName = "Resources")
    {
        containerName = containerName.Trim('.');

        //var asm = typeof(ResourceUtil).Assembly;
        var fullResourceName = asm.GetName().Name + $".{containerName}." + resourceName;
        using var stream = asm.GetManifestResourceStream(fullResourceName);
        if (stream == null)
            throw new FileNotFoundException("Resource not found: " + fullResourceName);

        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);

        return memoryStream.ToArray();
    }

    private static IEnumerable<string> GetResourceNames(this Assembly asm, string containerName = "Resources")
    {
        containerName = containerName.Trim('.');

        //var asm = typeof(ResourceUtil).Assembly;
        var resRoot = asm.GetName().Name + $".{containerName}.";
        return asm.GetManifestResourceNames()
            .Where(name => name.StartsWith(resRoot, StringComparison.Ordinal))
            .Select(name => name.Substring(resRoot.Length)); // Trim the prefix
    }
}