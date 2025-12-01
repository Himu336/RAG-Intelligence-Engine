# How to Install FFmpeg on Windows

FFmpeg is required for audio processing (WebM to WAV conversion and Whisper transcription).

## Quick Installation Methods

### Method 1: Windows Package Manager (Recommended - Easiest)
```bash
winget install ffmpeg
```
Then restart your terminal/server.

### Method 2: Chocolatey
```bash
choco install ffmpeg
```
Then restart your terminal/server.

### Method 3: Manual Installation

1. **Download FFmpeg:**
   - Go to: https://www.gyan.dev/ffmpeg/builds/
   - Click "Download Build" (choose the latest version)
   - Or direct link: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip

2. **Extract the ZIP file:**
   - Extract to a location like `C:\ffmpeg`

3. **Add to PATH:**
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin` (or wherever you extracted ffmpeg)
   - Click "OK" on all dialogs

4. **Verify Installation:**
   - Open a NEW terminal (important - restart terminal)
   - Run: `ffmpeg -version`
   - You should see version information

5. **Restart Your Server:**
   - Close your uvicorn server
   - Restart it in the new terminal

## Verify Installation

After installation, verify it works:
```bash
ffmpeg -version
```

You should see output like:
```
ffmpeg version 6.x.x ...
```

## Troubleshooting

**Problem**: "ffmpeg is not recognized"
- Solution: Make sure you added ffmpeg to PATH and restarted your terminal

**Problem**: "Still getting errors after installation"
- Solution: Restart your terminal/server completely

**Problem**: "Can't find ffmpeg in PATH"
- Solution: Check that the `bin` folder (containing ffmpeg.exe) is in your PATH

## Alternative: Use Pre-built Binary

You can also download a pre-built binary from:
- https://github.com/BtbN/FFmpeg-Builds/releases
- Extract and add `bin` folder to PATH

## After Installation

Once installed, restart your FastAPI server:
```bash
uvicorn app.main:app
```

The WebM to WAV conversion should now work!

