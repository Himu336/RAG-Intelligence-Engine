# app/voice_interview/speech_to_text_local.py

"""
Fully free and open-source Speech-to-Text and Text-to-Speech services.

STT: OpenAI Whisper (fully open-source, free)
TTS: Coqui TTS (fully open-source, free, high quality)
"""

import base64
import io
import tempfile
import os
from typing import Optional, Tuple
import numpy as np


class SpeechToTextService:
    """
    Service for converting audio to text using OpenAI Whisper (fully open-source).
    Supports batch transcription.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initialize the Speech-to-Text service with Whisper.
        
        Args:
            model_size: Whisper model size - "tiny", "base", "small", "medium", "large"
                       "base" is a good balance of speed and accuracy
        """
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
        
        self.model_size = model_size
        self.model = None
        print(f"✅ Speech-to-Text service initialized (Whisper - {model_size} model)")
        print("💡 Loading Whisper model (first time may take a moment)...")
        
        # Lazy load model on first use
        self._load_model()

    def _load_model(self):
        """Lazy load the Whisper model."""
        if self.model is None:
            try:
                self.model = self.whisper.load_model(self.model_size)
                print(f"✅ Whisper model '{self.model_size}' loaded successfully")
            except Exception as e:
                print(f"❌ Error loading Whisper model: {e}")
                raise

    def transcribe_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        audio_format: str = "webm"
    ) -> Tuple[str, float]:
        """
        Transcribe audio data to text using Whisper.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Audio sample rate (Hz) - Whisper handles this automatically
            language_code: Language code (e.g., "en-US") - Whisper auto-detects if None
            audio_format: Audio format (webm, wav, flac, mp3, etc.)
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            if not audio_data or len(audio_data) == 0:
                print("❌ Audio data is empty - cannot transcribe")
                return ("", 0.0)

            # Ensure model is loaded
            self._load_model()

            # Convert language code to Whisper format (e.g., "en-US" -> "en")
            language = None
            if language_code:
                language = language_code.split("-")[0].lower() if "-" in language_code else language_code.lower()
                # Whisper uses ISO 639-1 codes
                if language not in ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]:
                    language = None  # Let Whisper auto-detect

            # Whisper needs ffmpeg to load audio files
            # Convert WebM to WAV using pydub (which uses ffmpeg internally)
            tmp_file_path = None
            try:
                # Check if ffmpeg is available
                try:
                    from pydub.utils import which
                    ffmpeg_path = which("ffmpeg")
                    if not ffmpeg_path:
                        print("❌ ffmpeg not found in PATH")
                        print("📋 INSTALLATION INSTRUCTIONS:")
                        print("   1. Download ffmpeg from: https://ffmpeg.org/download.html")
                        print("   2. Or use Windows Package Manager: winget install ffmpeg")
                        print("   3. Or use Chocolatey: choco install ffmpeg")
                        print("   4. Add ffmpeg to your system PATH")
                        print("   5. Restart your terminal/server after installation")
                        return ("", 0.0)
                except ImportError:
                    pass  # pydub.utils might not be available, continue anyway
                
                # Always convert to WAV first for better compatibility
                if audio_format.lower() == "webm":
                    try:
                        from pydub import AudioSegment
                        # Load WebM and convert to WAV in memory
                        print("🔄 Converting WebM to WAV...")
                        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data), format="webm")
                        wav_buffer = io.BytesIO()
                        audio_segment.export(wav_buffer, format="wav")
                        wav_data = wav_buffer.getvalue()
                        
                        # Save WAV to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                            tmp_file.write(wav_data)
                            tmp_file_path = tmp_file.name
                        print("✅ WebM converted to WAV")
                    except ImportError:
                        print("❌ pydub not installed - required for WebM conversion")
                        print("💡 Install with: pip install pydub")
                        return ("", 0.0)
                    except FileNotFoundError as e:
                        print("❌ ffmpeg not found - required for audio conversion")
                        print("📋 INSTALLATION INSTRUCTIONS:")
                        print("   Windows:")
                        print("   1. Download from: https://www.gyan.dev/ffmpeg/builds/")
                        print("   2. Extract and add 'bin' folder to PATH")
                        print("   3. Or use: winget install ffmpeg")
                        print("   4. Restart terminal/server")
                        return ("", 0.0)
                    except Exception as e:
                        print(f"❌ Error converting WebM to WAV: {e}")
                        print("💡 Make sure ffmpeg is installed and in PATH")
                        print("💡 Download: https://ffmpeg.org/download.html")
                        return ("", 0.0)
                else:
                    # For other formats, save directly
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp_file:
                        tmp_file.write(audio_data)
                        tmp_file_path = tmp_file.name

                # Transcribe using Whisper
                print(f"🎤 Transcribing audio file: {tmp_file_path}")
                result = self.model.transcribe(
                    tmp_file_path,
                    language=language,  # None = auto-detect
                    task="transcribe"
                )
            
                transcript = result.get("text", "").strip()
                
                # Whisper doesn't provide confidence scores directly
                # We can use the average log probability as a proxy
                segments = result.get("segments", [])
                if segments:
                    avg_logprob = sum(seg.get("avg_logprob", -1.0) for seg in segments) / len(segments)
                    # Convert log probability to confidence (rough estimate)
                    # Log probs are typically between -1 and 0, we normalize to 0-1
                    confidence = max(0.0, min(1.0, (avg_logprob + 1.0)))
                else:
                    confidence = 0.9 if transcript else 0.0
                
                print(f"✅ Transcription successful: {len(transcript)} characters")
                return (transcript, confidence)
                
            finally:
                # Clean up temporary file
                if tmp_file_path and os.path.exists(tmp_file_path):
                    try:
                        os.unlink(tmp_file_path)
                    except Exception:
                        pass

        except Exception as e:
            print(f"❌ Error transcribing audio: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return ("", 0.0)

    def transcribe_base64(
        self,
        base64_audio: str,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        audio_format: str = "webm"
    ) -> Tuple[str, float]:
        """
        Transcribe base64-encoded audio to text.
        
        Args:
            base64_audio: Base64-encoded audio string (may include data URL prefix)
            sample_rate: Audio sample rate (Hz)
            language_code: Language code
            audio_format: Audio format
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Remove data URL prefix if present
            if "," in base64_audio:
                base64_audio = base64_audio.split(",", 1)[1]
            
            if not base64_audio or len(base64_audio.strip()) == 0:
                print("❌ Empty base64 audio string")
                return ("", 0.0)
            
            # Decode base64
            try:
                audio_data = base64.b64decode(base64_audio, validate=True)
            except Exception as decode_error:
                print(f"❌ Invalid base64 encoding: {decode_error}")
                return ("", 0.0)
            
            if not audio_data or len(audio_data) == 0:
                print("❌ Decoded audio data is empty")
                return ("", 0.0)
            
            print(f"📊 Audio data size: {len(audio_data)} bytes, format: {audio_format}")
            
            return self.transcribe_audio(audio_data, sample_rate, language_code, audio_format)
            
        except Exception as e:
            print(f"❌ Error decoding base64 audio: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return ("", 0.0)


class TextToSpeechService:
    """
    Service for converting text to speech using edge-tts (Microsoft Edge TTS).
    Fully free, simple, no compilation required, good quality.
    """

    def __init__(self):
        """Initialize the Text-to-Speech service with edge-tts."""
        try:
            import edge_tts
            self.edge_tts = edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Install with: pip install edge-tts"
            )
        
        self.voice_cache = None
        print("✅ Text-to-Speech service initialized (edge-tts)")
        print("💡 edge-tts is ready to use (no model loading required)")

    async def _get_voice_by_gender_async(self, gender: str, language_code: str = "en-US") -> str:
        """
        Get a voice ID based on gender preference (async version).
        
        Args:
            gender: "MALE", "FEMALE", or "NEUTRAL"
            language_code: Language code (e.g., "en-US")
            
        Returns:
            Voice ID string
        """
        try:
            # Cache voices list (only fetch once)
            if self.voice_cache is None:
                voices = await self.edge_tts.list_voices()
                self.voice_cache = voices
            
            # Extract language from language_code (e.g., "en-US" -> "en")
            lang = language_code.split("-")[0].lower() if "-" in language_code else language_code.lower()
            
            # Filter voices by language and gender
            gender_lower = gender.upper()
            matching_voices = []
            
            for voice in self.voice_cache:
                voice_lang = voice.get("Locale", "").split("-")[0].lower()
                voice_gender = voice.get("Gender", "").upper()
                
                if voice_lang == lang:
                    if gender_lower == "NEUTRAL" or voice_gender == gender_lower:
                        matching_voices.append(voice)
            
            # Return first matching voice, or default English voice
            if matching_voices:
                return matching_voices[0].get("ShortName", "en-US-AriaNeural")
            
            # Default fallback voices
            defaults = {
                "en": {
                    "MALE": "en-US-GuyNeural",
                    "FEMALE": "en-US-AriaNeural",
                    "NEUTRAL": "en-US-AriaNeural"
                }
            }
            
            lang_defaults = defaults.get(lang, defaults["en"])
            return lang_defaults.get(gender_lower, "en-US-AriaNeural")
            
        except Exception as e:
            print(f"⚠️ Error selecting voice: {e}")
            # Return default voice
            return "en-US-AriaNeural"
    
    def _get_voice_by_gender(self, gender: str, language_code: str = "en-US") -> str:
        """
        Get a voice ID based on gender preference (sync wrapper).
        Uses default voices without async call.
        """
        # Default fallback voices (no need to fetch all voices)
        lang = language_code.split("-")[0].lower() if "-" in language_code else language_code.lower()
        gender_lower = gender.upper()
        
        defaults = {
            "en": {
                "MALE": "en-US-GuyNeural",
                "FEMALE": "en-US-AriaNeural",
                "NEUTRAL": "en-US-AriaNeural"
            },
            "es": {
                "MALE": "es-ES-AlvaroNeural",
                "FEMALE": "es-ES-ElviraNeural",
                "NEUTRAL": "es-ES-ElviraNeural"
            },
            "fr": {
                "MALE": "fr-FR-DeniseNeural",
                "FEMALE": "fr-FR-DeniseNeural",
                "NEUTRAL": "fr-FR-DeniseNeural"
            }
        }
        
        lang_defaults = defaults.get(lang, defaults["en"])
        return lang_defaults.get(gender_lower, "en-US-AriaNeural")

    async def synthesize_speech_async(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3"
    ) -> Optional[bytes]:
        """
        Synthesize speech from text using edge-tts (async version).
        
        Args:
            text: Text to convert to speech
            language_code: Language code (e.g., "en-US")
            voice_name: Voice name (optional, uses default based on gender if not provided)
            ssml_gender: Gender ("NEUTRAL", "MALE", "FEMALE") - used to select voice
            audio_format: Output format ("mp3" is default for edge-tts)
            
        Returns:
            Audio bytes or None if error
        """
        try:
            if not text or len(text.strip()) == 0:
                print("❌ Empty text - cannot synthesize speech")
                return None

            # Select voice
            if voice_name is None:
                voice_name = await self._get_voice_by_gender_async(ssml_gender, language_code)
            
            # edge-tts outputs MP3 by default, which is perfect
            print(f"🎤 Synthesizing speech with voice: {voice_name}")
            
            # Generate speech
            communicate = self.edge_tts.Communicate(text, voice_name)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            if audio_data:
                print(f"✅ Speech synthesis successful: {len(audio_data)} bytes")
                return audio_data
            else:
                print("❌ No audio data generated")
                return None

        except Exception as e:
            print(f"❌ Error synthesizing speech: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return None

    def synthesize_speech(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3"
    ) -> Optional[bytes]:
        """
        Synthesize speech from text using edge-tts (sync wrapper - for compatibility).
        Note: This creates a new event loop. Use synthesize_speech_async() in async contexts.
        """
        try:
            import asyncio
            # Try to get existing event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use a different approach
                    # Create a task in the existing loop
                    import nest_asyncio
                    nest_asyncio.apply()
                    return loop.run_until_complete(
                        self.synthesize_speech_async(text, language_code, voice_name, ssml_gender, audio_format)
                    )
                else:
                    return loop.run_until_complete(
                        self.synthesize_speech_async(text, language_code, voice_name, ssml_gender, audio_format)
                    )
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(
                    self.synthesize_speech_async(text, language_code, voice_name, ssml_gender, audio_format)
                )
        except Exception as e:
            print(f"❌ Error in sync wrapper: {e}")
            return None

    async def synthesize_to_base64_async(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3"
    ) -> Optional[str]:
        """
        Synthesize speech and return as base64 string (async version).
        
        Args:
            text: Text to convert to speech
            language_code: Language code (kept for compatibility)
            voice_name: Voice name (optional)
            ssml_gender: Gender preference (kept for compatibility)
            audio_format: Output format ("mp3" recommended, falls back to "wav")
            
        Returns:
            Base64-encoded audio string or None if error
        """
        audio_bytes = await self.synthesize_speech_async(
            text, language_code, voice_name, ssml_gender, audio_format
        )
        
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None

    def synthesize_to_base64(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3"
    ) -> Optional[str]:
        """
        Synthesize speech and return as base64 string (sync wrapper).
        """
        audio_bytes = self.synthesize_speech(
            text, language_code, voice_name, ssml_gender, audio_format
        )
        
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None

