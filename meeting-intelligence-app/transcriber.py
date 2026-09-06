import tempfile
from faster_whisper import WhisperModel

# On macOS, CPU execution with int8 quantization is fast and stable
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio_file(uploaded_file) -> str:
    """Takes a Streamlit uploaded file and transcribes with timestamps."""
    suffix = f".{uploaded_file.name.split('.')[-1]}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
    
    transcript_lines = []
    for seg in segments:
        transcript_lines.append(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()}")
        
    return "\n".join(transcript_lines)