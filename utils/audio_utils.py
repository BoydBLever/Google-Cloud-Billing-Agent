# utils/audio_utils.py
import os
import uuid
import io
from typing import Optional
from gtts import gTTS

from google.cloud import speech_v2 as speech
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

class AudioProcessor:
    """
    Cloud Run-safe audio processor:
    - Expects audio files already converted to 16k mono WAV (via ffmpeg)
    - Uses Google Speech-to-Text v2 (Chirp 3) for transcription
    - Uses gTTS for speech output
    """
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.speech_client = speech.SpeechClient()

        # Cloud Run auto-populates GOOGLE_CLOUD_PROJECT
        self.project_id = (
            os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
            or os.getenv("PROJECT_ID")
        )
        if not self.project_id:
            raise ValueError("Missing GOOGLE_CLOUD_PROJECT for STT v2.")

        self.location = os.getenv("SPEECH_LOCATION", "us")

        # Point the Speech-to-Text v2 client at the correct regional endpoint
        endpoint = f"{self.location}-speech.googleapis.com"

        self.speech_client = speech.SpeechClient(
            client_options=ClientOptions(api_endpoint=endpoint)
        )

        # Default ephemeral recognizer
        self.recognizer = f"projects/{self.project_id}/locations/{self.location}/recognizers/_"

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using Google Cloud Speech-to-Text v2 + Chirp 3.
        """
        try:
            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                model="chirp_3",
                language_codes=["en-US"],
            )

            request = cloud_speech.RecognizeRequest(
                recognizer=self.recognizer,
                config=config,
                content=content,
            )

            response = self.speech_client.recognize(request=request)

            if response.results:
                return response.results[0].alternatives[0].transcript.strip()
            return None

        except Exception as e:
            print(f"CHIRP_ERROR: {e}")
            return f"CHIRP_ERROR: {e}"

    def text_to_speech(self, text: str) -> Optional[str]:
        """
        Convert text → mp3 via gTTS and return the file path.
        """
        try:
            filename = f"tts_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text, lang="en")
            tts.save(filename)
            return filename
        except Exception:
            return None
