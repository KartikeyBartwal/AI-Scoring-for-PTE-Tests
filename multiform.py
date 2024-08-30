from fastapi import FastAPI, File, UploadFile, Form

app = FastAPI()


@app.post("/login/")
async def login(username: str = Form(), password: str = Form()):
    return {"username": username}

@app.post("/pronunciation_fluency_content_scoring/")
async def speech_scoring(speech_topic: str = Form(), audio_file: UploadFile = File(...)):
    return {"username" : 0}
