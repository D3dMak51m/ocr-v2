using IronOcr;
using Microsoft.AspNetCore.Http.Features;

var builder = WebApplication.CreateBuilder(args);

// Optional: set your IronOCR license via env var (trial works without this)
var licenseKey = Environment.GetEnvironmentVariable("IRON_OCR_LICENSE_KEY");
if (!string.IsNullOrWhiteSpace(licenseKey))
{
    IronOcr.License.LicenseKey = licenseKey;
}

// Allow large uploads (adjust as needed)
builder.Services.Configure<FormOptions>(o =>
{
    o.MultipartBodyLengthLimit = 1024L * 1024L * 200L; // 200 MB
});

var app = builder.Build();

app.MapGet("/", () => Results.Ok(new
{
    name = "ironocr-service",
    version = "1.0",
    endpoints = new[] { "POST /ocr" }
}));

// POST /ocr?lang=uzbek or /ocr?lang=uzbek,english
app.MapPost("/ocr", async (HttpRequest request) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest("Send multipart/form-data with a 'file' field.");

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("file");
    if (file is null || file.Length == 0)
        return Results.BadRequest("Missing 'file'.");

    // Language selection: defaults to Uzbek
    var langQuery = request.Query["lang"].ToString();
    var tokens = string.IsNullOrWhiteSpace(langQuery)
        ? new[] { "uzbek" }
        : langQuery.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    var ocr = new IronTesseract();

    // Map tokens to IronOCR languages: first is primary, rest are secondary
    bool primarySet = false;
    foreach (var token in tokens)
    {
    OcrLanguage mapped;
    switch (token.ToLowerInvariant())
    {
        case "uzbek":
        case "uzb":
            mapped = OcrLanguage.Uzbek;            // Latin-focused
            break;
        case "uzbek-cyrillic":
        case "uzbek_cyrillic":
        case "uzb-cyr":
        case "uzb_cyr":
        case "uz":
        case "uz-cyrl":
            mapped = OcrLanguage.UzbekCyrillic;    // <- Cyrillic
            break;
        case "uzbek-cyrillic-best":
            mapped = OcrLanguage.UzbekCyrillicBest;
            break;
        case "uzbek-cyrillic-fast":
            mapped = OcrLanguage.UzbekCyrillicFast;
            break;
        case "russian":
        case "rus":
        case "ru":
            mapped = OcrLanguage.Russian;
            break;
        case "english":
        case "eng":
            mapped = OcrLanguage.English;
            break;
        default:
            return Results.BadRequest($"Unknown language token: '{token}'.");
    }
        if (!primarySet)
        {
            ocr.Language = mapped;               // primary
            primarySet = true;
        }
        else
        {
            ocr.AddSecondaryLanguage(mapped);    // additional languages
        }
    }

    using var ms = new MemoryStream();
    await file.CopyToAsync(ms);
    ms.Position = 0;

    // Use the new OcrInput loaders (image vs pdf)
    using var input = new OcrInput();
    var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
    var contentType = file.ContentType?.ToLowerInvariant();

    if (ext == ".pdf" || contentType == "application/pdf")
        input.LoadPdf(ms);      // Load PDF from stream
    else
        input.LoadImage(ms);    // Load image from stream

    var result = ocr.Read(input);

    return Results.Ok(new
    {
        language = string.Join(",", tokens),
        confidence = result.Confidence,
        text = result.Text
    });
});

app.Run();
