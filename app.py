# app.py
import streamlit as st
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import io
import uuid
import subprocess
import traceback
import sys

from utils.audio_utils import AudioProcessor
from utils.llm_utils import LLMProcessor

# Load environment variables (harmless on Cloud Run; useful locally)
load_dotenv()

# Configuration parameters
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
LLM_MODEL = os.getenv("LLM_MODEL")
if not LLM_MODEL:
    raise ValueError("Missing LLM_MODEL in env (e.g., gemini-2.5-flash-lite)")

# Initialize processors
audio_processor = AudioProcessor(sample_rate=SAMPLE_RATE)
llm_processor = LLMProcessor(model_name=LLM_MODEL)

# Initialize Streamlit state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "recording_duration" not in st.session_state:
    st.session_state.recording_duration = 5
if "mode" not in st.session_state:
    st.session_state.mode = "customer_service"


def visualize_audio(audio_path: str):
    """Visualize audio waveform"""
    try:
        import soundfile as sf
        data, _ = sf.read(audio_path)
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(np.arange(len(data)) / SAMPLE_RATE, data)
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Audio Waveform")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        st.image(buf)
    except Exception as e:
        st.warning(f"Unable to visualize audio: {str(e)}")


def process_transcript(transcript: str):
    """Shared logic for taking transcript -> Gemini -> TTS -> UI"""
    st.info(f"Transcription: {transcript}")

    with st.spinner("Generating response..."):
        response = llm_processor.run_agent_step(
            transcript,
            conversation_history=st.session_state.messages,
        )

    st.session_state.messages.append({"role": "user", "content": transcript})
    st.session_state.messages.append({"role": "assistant", "content": response})

    audio_response = audio_processor.text_to_speech(response)
    st.success("AI response:")
    st.write(response)
    if audio_response:
        st.audio(audio_response)


def _convert_to_wav(input_path: str) -> str:
    """Convert arbitrary browser/uploaded audio to 16k mono wav for Whisper."""
    wav_path = f"/tmp/converted_{uuid.uuid4()}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path],
        check=True,
        capture_output=True,
    )
    return wav_path


def handle_audio_input_cloudrun():
    """Old/simple UI mic capture via Streamlit audio_input."""
    audio_file_obj = st.audio_input("Speak to G-bot")
    if audio_file_obj is None:
        return

    audio_bytes = audio_file_obj.getvalue()
    st.write(f"DEBUG: raw MIME type = {audio_file_obj.type}")

    # Check for all-zero data (silence or invalid recording)
    if all(b == 0 for b in audio_bytes[:500]):
        st.warning("DEBUG: audio seems to be all zeros (silent/invalid capture)")

    # Best-guess extension from mime type
    in_ext = "webm"
    try:
        if getattr(audio_file_obj, "type", None):
            in_ext = audio_file_obj.type.split("/")[-1]
    except Exception:
        pass

    in_path = f"/tmp/input_{uuid.uuid4()}.{in_ext}"
    with open(in_path, "wb") as f:
        f.write(audio_bytes)

    try:
        temp_path = _convert_to_wav(in_path)
    except Exception as e:
        tb = traceback.format_exc()
        print("FFMPEG_CONVERSION_ERROR:\n", tb, file=sys.stderr)
        st.error(f"Audio conversion failed: {e}")
        st.text(tb)
        return

    try:
        size = os.path.getsize(temp_path)
        st.write(f"DEBUG: converted WAV size = {size} bytes")

        st.audio(temp_path)

        with open(temp_path, "rb") as f:
            st.download_button(
                "Download converted WAV (debug)",
                f.read(),
                file_name="debug_converted.wav",
                mime="audio/wav"
            )
    except Exception as e:
        st.error(f"DEBUG: Could not inspect WAV: {e}")

    # Basic empty-audio guard
    try:
        if os.path.getsize(temp_path) < 2000:
            st.error("No audio captured (file too small). Try again.")
            return
    except Exception:
        pass

    st.audio(temp_path)
    visualize_audio(temp_path)

    with st.spinner("Transcribing..."):
        try:
            transcript = audio_processor.transcribe_audio(temp_path)
        except Exception as e:
            tb = traceback.format_exc()
            print("TRANSCRIPTION_ERROR:\n", tb, file=sys.stderr)
            st.error(f"Audio transcription failed: {e}")
            st.text(tb)
            transcript = None

    if transcript:
        process_transcript(transcript)
    else:
        st.error("Audio transcription failed")


def handle_text_input():
    user_input = st.text_input("Enter your question:")
    if not user_input:
        return

    with st.spinner("Generating response..."):
        response = llm_processor.run_agent_step(
            user_input,
            conversation_history=st.session_state.messages,
        )

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})

    audio_response = audio_processor.text_to_speech(response)
    st.success("AI response:")
    st.write(response)
    if audio_response:
        st.audio(audio_response)
    st.rerun()


def display_conversation_history():
    st.subheader("Conversation History")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


def settings_section():
    st.sidebar.header("Settings")

    # Kept for parity with old UI; not used by st.audio_input yet.
    st.session_state.recording_duration = st.sidebar.slider(
        "Recording Duration (seconds)",
        min_value=3,
        max_value=15,
        value=st.session_state.recording_duration,
    )

    mode = st.sidebar.radio("Select mode", ["Customer Service", "Lead Generation"])

    if mode == "Customer Service" and st.session_state.mode != "customer_service":
        st.session_state.mode = "customer_service"
        llm_processor.customize_for_call_center()
        st.sidebar.success("Switched to Customer Service mode")
    elif mode == "Lead Generation" and st.session_state.mode != "lead_generation":
        st.session_state.mode = "lead_generation"
        llm_processor.customize_for_lead_generation()
        st.sidebar.success("Switched to Lead Generation mode")

    if st.sidebar.button("Clear Conversation History"):
        st.session_state.messages = []
        st.sidebar.success("Conversation History Cleared")
        st.rerun()


def main():
    st.title("Google Cloud Billing Voice Bot (Geebot)")
    st.write(
        "An omnilingual AI voice assistant for Google Cloud Billing support. "
        "Speak naturally or type your question."
    )

    settings_section()

    tab1, tab2, tab3 = st.tabs(
        ["Voice Interaction", "Text Interaction", "Conversation Analysis"]
    )

    with tab1:
        st.subheader("Voice Input (Cloud Run Safe)")
        handle_audio_input_cloudrun()

        uploaded_file = st.file_uploader("Or Upload Audio File", type=["wav", "mp3", "m4a", "webm", "ogg"])
        if uploaded_file is not None:
            with st.spinner("Processing uploaded audio..."):
                up_bytes = uploaded_file.getvalue()
                up_ext = "wav"
                try:
                    if getattr(uploaded_file, "type", None):
                        up_ext = uploaded_file.type.split("/")[-1]
                except Exception:
                    pass

                up_in_path = f"/tmp/upload_{uuid.uuid4()}.{up_ext}"
                with open(up_in_path, "wb") as f:
                    f.write(up_bytes)

                try:
                    temp_path = _convert_to_wav(up_in_path)
                except Exception as e:
                    tb = traceback.format_exc()
                    print("FFMPEG_UPLOAD_CONVERSION_ERROR:\n", tb, file=sys.stderr)
                    st.error(f"Audio conversion failed: {e}")
                    st.text(tb)
                    temp_path = None

                if temp_path:
                    st.audio(temp_path)
                    visualize_audio(temp_path)

                    try:
                        transcript = audio_processor.transcribe_audio(temp_path)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print("TRANSCRIPTION_ERROR:\n", tb, file=sys.stderr)
                        st.error(f"Audio transcription failed: {e}")
                        st.text(tb)
                        transcript = None

                    if transcript:
                        process_transcript(transcript)
                    else:
                        st.error("Audio transcription failed")

    with tab2:
        st.subheader("Text Interaction")
        handle_text_input()

    with tab3:
        st.subheader("Conversation Analysis")
        if st.session_state.messages:
            if st.button("Analyze Conversation"):
                with st.spinner("Analyzing conversation..."):
                    analysis = llm_processor.analyze_conversation(st.session_state.messages)
                    st.write(analysis.get("analysis", ""))
        else:
            st.info("Conversation history is empty, cannot perform analysis")

    display_conversation_history()


if __name__ == "__main__":
    main()
