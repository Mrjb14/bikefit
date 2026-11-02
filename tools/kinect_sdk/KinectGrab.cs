// Capture Kinect Xbox 360 (v1) via le Kinect SDK 1.8 -> flux binaire sur stdout.
//
// Consommé par scripts/kinect_capture.py (--driver sdk). Pour chaque frame :
//   "KFRM" (4 octets) | timestamp_ms int32 | profondeur 640*480 uint16 (mm, alignée sur
//   la caméra couleur, 0 = invalide) | couleur 640*480*3 octets BGR
// Tout est little-endian. Les messages (état, erreurs) vont sur stderr.
// Build : tools/kinect_sdk/build.ps1
using System;
using System.IO;
using System.Threading;
using Microsoft.Kinect;

static class KinectGrab
{
    const int W = 640, H = 480;
    const ColorImageFormat ColorFmt = ColorImageFormat.RgbResolution640x480Fps30;
    const DepthImageFormat DepthFmt = DepthImageFormat.Resolution640x480Fps30;

    static Stream stdout;
    static KinectSensor sensor;
    static DepthImagePixel[] depthPixels;
    static DepthImagePoint[] mapped;
    static byte[] colorBgra, packet;
    static readonly object gate = new object();
    static long t0 = -1;

    static int Main()
    {
        foreach (var s in KinectSensor.KinectSensors)
            if (s.Status == KinectStatus.Connected) { sensor = s; break; }
        if (sensor == null)
        {
            Console.Error.WriteLine("Kinect introuvable (ou non alimentée : adaptateur secteur ?).");
            return 2;
        }
        stdout = Console.OpenStandardOutput();
        sensor.ColorStream.Enable(ColorFmt);
        sensor.DepthStream.Enable(DepthFmt);
        depthPixels = new DepthImagePixel[sensor.DepthStream.FramePixelDataLength];
        mapped = new DepthImagePoint[W * H];
        colorBgra = new byte[sensor.ColorStream.FramePixelDataLength];
        packet = new byte[4 + 4 + W * H * 2 + W * H * 3];
        packet[0] = (byte)'K'; packet[1] = (byte)'F'; packet[2] = (byte)'R'; packet[3] = (byte)'M';
        sensor.AllFramesReady += OnFrames;
        sensor.Start();
        Console.Error.WriteLine("Kinect démarrée (" + sensor.UniqueKinectId + ").");
        // Tourne jusqu'à ce que Python ferme stdin ou tue le processus.
        try { while (Console.In.Read() >= 0) { } } catch { }
        lock (gate) { sensor.AllFramesReady -= OnFrames; }
        sensor.Stop();
        return 0;
    }

    static void OnFrames(object s, AllFramesReadyEventArgs e)
    {
        lock (gate)
        {
            using (var cf = e.OpenColorImageFrame())
            using (var df = e.OpenDepthImageFrame())
            {
                if (cf == null || df == null) return;
                cf.CopyPixelDataTo(colorBgra);
                df.CopyDepthImagePixelDataTo(depthPixels);
                // Pour chaque pixel couleur : la profondeur (mm) au point correspondant.
                sensor.CoordinateMapper.MapColorFrameToDepthFrame(ColorFmt, DepthFmt, depthPixels, mapped);
                if (t0 < 0) t0 = cf.Timestamp;
                int ts = (int)(cf.Timestamp - t0);
                Buffer.BlockCopy(BitConverter.GetBytes(ts), 0, packet, 4, 4);
                int o = 8;
                for (int i = 0; i < W * H; i++)
                {
                    var p = mapped[i];
                    int d = (p.X >= 0 && p.X < W && p.Y >= 0 && p.Y < H) ? p.Depth : 0;
                    if (d < 0 || d > 65535) d = 0;
                    packet[o++] = (byte)(d & 0xFF);
                    packet[o++] = (byte)(d >> 8);
                }
                for (int i = 0; i < W * H; i++)
                {
                    packet[o++] = colorBgra[i * 4];
                    packet[o++] = colorBgra[i * 4 + 1];
                    packet[o++] = colorBgra[i * 4 + 2];
                }
            }
            try { stdout.Write(packet, 0, packet.Length); stdout.Flush(); }
            catch (IOException) { Environment.Exit(0); }
        }
    }
}
