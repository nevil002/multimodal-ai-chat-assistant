import re, tempfile
from config import groq

def groq_tts(text):
    response=groq.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="troy",
        input=text,
        response_format="wav"
    )
    output_file="response.wav"
    response.write_to_file(output_file)
    return output_file

def to_speech_text(text):
    text = re.sub(r"【[^】]*】|\[[^\]]*\]", "", text)   # citation brackets
    text = re.sub(r"[*_`#>|]", "", text)               # markdown symbols
    return re.sub(r"\s+", " ", text).strip()