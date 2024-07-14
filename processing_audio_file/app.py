from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import librosa
import numpy as np
import io

app = FastAPI()

@app.get("/")
async def welcome():
    return "Process Audio"


@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()
    
    # Use BytesIO to create a file-like object from the contents
    audio_file = io.BytesIO(contents)
    
    # Load the audio file using librosa and resample to 16000 Hz
    audio_array, sample_rate = librosa.load(audio_file, sr=16000)
    
    # Convert the numpy array to a list for JSON serialization
    audio_list = audio_array.tolist()
    
    # Return the audio array and sample rate
    return JSONResponse({
        "audio_array": audio_list,
        "sample_rate": sample_rate
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)