import whisper
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

model = whisper.load_model("tiny")

app = FastAPI()


def transcribe(audio_file_path: str, model):
    # Load audio and run inference
    result = model.transcribe(audio_file_path)
    return result["text"]

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):

    # SAVE THE UPLOAD FILE TEMPORARILY
    with open(file.filename, "wb") as buffer:

        buffer.write(await file.read())

    # TRANSCRIBE THE AUDIO
    transcription = transcribe(file.filename, model)

    return { "transcription" : transcription }
