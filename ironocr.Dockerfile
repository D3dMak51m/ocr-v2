FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY ./src/*.csproj ./src/
RUN dotnet restore ./src/IronOcrService.csproj
COPY ./src ./src
RUN dotnet publish ./src/IronOcrService.csproj -c Release -o /app/publish

# Runtime stage
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app

# Native deps often needed by OCR/image codecs + GDI+ for font rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdiplus \
    tesseract-ocr \
    libjpeg62-turbo \
    libtiff6 \
    libwebp7 \
    libpng16-16 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["dotnet", "IronOcrService.dll"]
