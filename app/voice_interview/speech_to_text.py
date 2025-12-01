from typing import Optional, Tuple
import base64
import io
import tempfile
import os

from openai import OpenAI
from groq import Groq
from app.config import settings


class SpeechToTextService:
    """
    Speech-to-text using Groq Whisper via OpenAI-compatible Audio API.

    Replaces previous implementations while keeping the same public methods
    used by the router.
    """

    def __init__(self):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=settings.GROQ_API_KEY,
        )
        print("✅ Speech-to-Text service initialized (Groq Whisper via OpenAI API)")

    def transcribe_audio(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        audio_format: str = "webm",
    ) -> Tuple[str, float]:
        """
        Transcribe raw audio bytes to text using Groq Whisper.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Unused (OpenAI handles this)
            language_code: Language code (hint only)
            audio_format: File extension hint (webm, wav, mp3, etc.)
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        if not audio_data:
                print("❌ Audio data is empty - cannot transcribe")
                return ("", 0.0)
            
        try:
            # Wrap bytes in a file-like object and give it a name with extension
            file_like = io.BytesIO(audio_data)
            # API uses the filename extension to infer type
            file_like.name = f"audio.{audio_format or 'webm'}"

            # Use Groq Whisper Large v3 Turbo for fast, cost-effective STT
            resp = self.client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=file_like,
                # You can also set: response_format="text", language="<code>" etc.
            )

            text = getattr(resp, "text", "") or ""
            confidence = 0.9 if text else 0.0
            print(f"✅ Transcription successful: {len(text)} characters")
            return (text, confidence)
        except Exception as e:
            print(f"❌ Groq STT error: {e}")
            return ("", 0.0)

    def transcribe_base64(
        self,
        base64_audio: str,
        sample_rate: int = 16000,
        language_code: str = "en-US",
        audio_format: str = "webm",
    ) -> Tuple[str, float]:
        """
        Transcribe base64-encoded audio to text.
        
        Args:
            base64_audio: Base64-encoded audio string (may include data URL prefix)
            sample_rate: Audio sample rate (Hz) - unused
            language_code: Language code
            audio_format: Audio format
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        try:
            # Remove data URL prefix if present (e.g., "data:audio/webm;base64,")
            if "," in base64_audio:
                base64_audio = base64_audio.split(",", 1)[1]
            
            if not base64_audio or not base64_audio.strip():
                print("❌ Empty base64 audio string")
                return ("", 0.0)
            
            try:
                audio_data = base64.b64decode(base64_audio, validate=True)
            except Exception as decode_error:
                print(f"❌ Invalid base64 encoding: {decode_error}")
                return ("", 0.0)
            
            if not audio_data:
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
    Text-to-speech using Groq PlayAI TTS via OpenAI-compatible Audio API.

    Produces MP3 audio that is returned as raw bytes or base64, similar to
    the previous implementations.
    """

    def __init__(self):
        # Use Groq's native Python SDK for TTS (PlayAI)
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        # Default Groq TTS model and voice
        self.model = "playai-tts"
        self.voice = "Fritz-PlayAI"
        print("✅ Text-to-Speech service initialized (Groq PlayAI TTS)")

    def synthesize_speech(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3",
    ) -> Optional[bytes]:
        """
        Synthesize speech from text using Groq TTS.
        
        Args:
            text: Text to convert to speech
            language_code: Unused (OpenAI infers from text)
            voice_name: Optional override for voice (kept for compatibility)
            ssml_gender: Unused, kept for compatibility
            audio_format: Output format ("mp3" supported)
        
        Returns:
            Audio bytes or None if error
        """
        if not text or not text.strip():
            print("❌ Empty text - cannot synthesize speech")
            return None

        try:
            voice = voice_name or self.voice

            # Call Groq TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=voice,
                input=text,
                response_format=audio_format,
            )

            # Write to a temporary file and read back into memory
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp_file:
                tmp_path = tmp_file.name

            try:
                response.write_to_file(tmp_path)
                with open(tmp_path, "rb") as f:
                    audio_bytes = f.read()
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if audio_bytes:
                print(f"✅ Speech synthesis successful: {len(audio_bytes)} bytes")
                return audio_bytes

            print("❌ No audio data generated from Groq TTS")
            return None
        except Exception as e:
            print(f"❌ Groq TTS error: {e}")
            return None

    def synthesize_to_base64(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3",
    ) -> Optional[str]:
        """
        Synthesize speech and return as base64 string.
        
        This matches the interface used by the voice interview router.
        """
        audio_bytes = self.synthesize_speech(
            text=text,
            language_code=language_code,
            voice_name=voice_name,
            ssml_gender=ssml_gender,
            audio_format=audio_format,
        )
        
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode("utf-8")

        return None

    async def synthesize_to_base64_async(
        self,
        text: str,
        language_code: str = "en-US",
        voice_name: str = None,
        ssml_gender: str = "NEUTRAL",
        audio_format: str = "mp3",
    ) -> Optional[str]:
        """
        Async wrapper used by the voice interview router.
        Offloads the synchronous TTS call to a thread so it doesn't block the event loop.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.synthesize_to_base64,
            text,
            language_code,
            voice_name,
            ssml_gender,
            audio_format,
        )
