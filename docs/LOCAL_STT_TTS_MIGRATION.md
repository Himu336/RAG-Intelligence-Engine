# Migration Guide: API to Local Open-Source STT/TTS

## Overview

This migration replaces Eleven Labs API with fully free and open-source solutions:
- **STT**: OpenAI Whisper (fully open-source, free)
- **TTS**: edge-tts (Microsoft Edge TTS - simple, free, no compilation needed)

## Benefits

✅ **Fully Free** - No API costs  
✅ **Open Source** - Complete control over the code  
✅ **Privacy** - All processing happens locally  
✅ **No Rate Limits** - Process as much as you want  
✅ **Same Interface** - No changes needed in your code

## Installation

### 1. Install Dependencies

```bash
pip install openai-whisper edge-tts
```

**That's it!** No C++ build tools or ffmpeg needed. edge-tts is pure Python and works out of the box.

### 2. Download Models (Automatic)

The models will be downloaded automatically on first use:
- **Whisper**: Downloads the model size you specify (default: "base")
- **edge-tts**: No model download needed - uses Microsoft Edge TTS API (free, no API key required)

### 3. Update Code

The code has already been updated! Just change the import in `router.py`:

**Before:**
```python
from app.voice_interview.speech_to_text import SpeechToTextService, TextToSpeechService
```

**After:**
```python
from app.voice_interview.speech_to_text_local import SpeechToTextService, TextToSpeechService
```

## Configuration

### Whisper Model Sizes

You can choose different Whisper model sizes in `speech_to_text_local.py`:

```python
# In SpeechToTextService.__init__()
model_size: str = "base"  # Options: "tiny", "base", "small", "medium", "large"
```

**Model Comparison:**
- **tiny**: Fastest, lowest accuracy (~39M parameters)
- **base**: Good balance (default, ~74M parameters)
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy (~1550M parameters)

### edge-tts Voices

edge-tts automatically selects voices based on gender and language. You can customize this in `speech_to_text_local.py`:

```python
# In TextToSpeechService._get_voice_by_gender()
# Default voices are automatically selected based on:
# - Language code (e.g., "en-US")
# - Gender preference ("MALE", "FEMALE", "NEUTRAL")
```

Available voices include:
- English: `en-US-AriaNeural` (female), `en-US-GuyNeural` (male)
- Many other languages supported

## Performance Considerations

### Whisper (STT)
- **First load**: Downloads model (~150MB for "base")
- **Processing time**: ~1-2 seconds per minute of audio (CPU)
- **GPU**: Much faster if available (set `device="cuda"` in whisper.load_model())

### edge-tts
- **First load**: No model download needed
- **Processing time**: ~0.5-1 second per sentence (depends on internet speed)
- **Internet**: Requires internet connection (uses Microsoft Edge TTS service, but free)

## Removing API Dependencies

### 1. Make ELEVEN_LABS_API_KEY Optional

Update `app/config.py`:

```python
# --- Eleven Labs API (No longer needed) ---
ELEVEN_LABS_API_KEY: str | None = Field(
    default=None,
    description="Eleven Labs API Key (optional - not needed with local services)"
)
```

### 2. Remove from Environment Variables

You can remove `ELEVEN_LABS_API_KEY` from your `.env` file.

## Testing

Test the new implementation:

```python
from app.voice_interview.speech_to_text_local import SpeechToTextService, TextToSpeechService

# Test STT
stt = SpeechToTextService()
# ... test with audio data ...

# Test TTS
tts = TextToSpeechService()
audio_base64 = tts.synthesize_to_base64("Hello, this is a test.")
```

## Troubleshooting

### Whisper Issues

**Problem**: Model download fails
**Solution**: Check internet connection, models download on first use

**Problem**: Slow transcription
**Solution**: 
- Use smaller model ("tiny" or "base")
- Use GPU if available
- Process audio in chunks

### edge-tts Issues

**Problem**: No audio generated
**Solution**: Check internet connection (edge-tts requires internet)

**Problem**: Voice not found
**Solution**: The code automatically falls back to default voices. Check language code format (e.g., "en-US")

**Problem**: Slow generation
**Solution**: This is normal - edge-tts depends on internet speed. Consider caching frequently used audio.

## Cost Savings

**Before (Eleven Labs API)**:
- STT: ~$0.006 per minute
- TTS: ~$0.015 per 1,000 characters
- **15-minute interview**: ~$0.09 (STT) + ~$0.03 (TTS) = **$0.12**

**After (Local Open-Source)**:
- STT: **$0.00** (free)
- TTS: **$0.00** (free)
- **15-minute interview**: **$0.00**

**Savings**: 100% cost reduction! 🎉

## Next Steps

1. ✅ Install dependencies: `pip install openai-whisper edge-tts`
2. ✅ Update import in `router.py` (already done)
3. ✅ Test the implementation
4. ✅ Remove `ELEVEN_LABS_API_KEY` from environment variables (optional)

## Notes

- **Whisper**: Models are downloaded once and cached locally, all processing happens on your server
- **edge-tts**: Requires internet connection (uses Microsoft Edge TTS service, but completely free)
- First Whisper request may be slower (model loading)
- GPU acceleration available for Whisper (set `device="cuda"` in whisper.load_model())
- edge-tts is pure Python - no compilation needed, works on Windows/Mac/Linux

