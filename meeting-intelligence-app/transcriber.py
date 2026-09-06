import tempfile
import os
from faster_whisper import WhisperModel

# Use base model for balanced speed and cloud CPU memory constraints
model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio_file(uploaded_file) -> str:
    """Saves uploaded audio temporarily and transcribes with timestamps."""
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_path = temp_audio.name

    try:
        segments, info = model.transcribe(temp_path, beam_size=5)
        transcript_lines = []
        for segment in segments:
            start_min = int(segment.start // 60)
            start_sec = int(segment.start % 60)
            transcript_lines.append(f"[{start_min:02d}:{start_sec:02d}] {segment.text.strip()}")
        
        return "\n".join(transcript_lines)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)