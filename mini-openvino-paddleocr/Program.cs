using OpenCvSharp;
using Sdcb.OpenVINO.PaddleOCR.Models.Online;
using Sdcb.OpenVINO.PaddleOCR.Models;
using Sdcb.OpenVINO.PaddleOCR;
using System.Diagnostics;
using System;

FullOcrModel model = await OnlineFullModels.ChineseV4.DownloadAsync();

using Mat src = Cv2.ImDecode(await new HttpClient().GetByteArrayAsync("https://io.starworks.cc:88/paddlesharp/ocr/samples/xdr5450.webp"), ImreadModes.Color);

using (PaddleOcrAll all = new(model)
{
    AllowRotateDetection = true,
    Enable180Classification = true,
})
{
    // Load local file by following code:
    // using (Mat src2 = Cv2.ImRead(@"C:\test.jpg"))
    Stopwatch sw = Stopwatch.StartNew();
    PaddleOcrResult result = all.Run(src);
    Console.WriteLine($"elapsed={sw.ElapsedMilliseconds} ms");
    Console.WriteLine("Detected all texts: \n" + result.Text);
    foreach (PaddleOcrResultRegion region in result.Regions)
    {
        Console.WriteLine($"Text: {region.Text}, Score: {region.Score}, RectCenter: {region.Rect.Center}, RectSize:    {region.Rect.Size}, Angle: {region.Rect.Angle}");
    }
}
