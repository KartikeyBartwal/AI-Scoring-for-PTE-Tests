import re
import requests
import pyarrow as pa
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from fastapi import FastAPI, File, UploadFile
import warnings
from starlette.formparsers import MultiPartParser
import io
import random
import tempfile
import os


MultiPartParser.max_file_size = 200 * 1024 * 1024

# Initialize FastAPI app
app = FastAPI()

# Load Wav2Vec2 tokenizer and model
# tokenizer = Wav2Vec2Tokenizer.from_pretrained("./models/tokenizer")
# model = Wav2Vec2ForCTC.from_pretrained("./models/model")



# Function to download English word list
def download_word_list():
    print("Downloading English word list...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    response = requests.get(url)
    words = set(response.text.split())
    print("Word list downloaded.")
    return words

english_words = download_word_list()

# Function to count correctly spelled words in text
def count_spelled_words(text, word_list):
    print("Counting spelled words...")
    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    correct = sum(1 for word in words if word in word_list)
    incorrect = len(words) - correct
    
    print("Spelling check complete.")
    return incorrect, correct

# Function to apply spell check to an item (assuming it's a dictionary)
def apply_spell_check(item, word_list):
    print("Applying spell check...")
    if isinstance(item, dict):
        # This is a single item
        text = item['transcription']
        incorrect, correct = count_spelled_words(text, word_list)
        item['incorrect_words'] = incorrect
        item['correct_words'] = correct
        print("Spell check applied to single item.")
        return item
    else:
        # This is likely a batch
        texts = item['transcription']
        results = [count_spelled_words(text, word_list) for text in texts]
        
        incorrect_counts, correct_counts = zip(*results)
        
        item = item.append_column('incorrect_words', pa.array(incorrect_counts))
        item = item.append_column('correct_words', pa.array(correct_counts))
        
        print("Spell check applied to batch of items.")
        return item

# FastAPI routes
@app.get('/')
async def root():
    return "Welcome to the pronunciation scoring API!"

@app.post('/check_post')
async def rnc(number):
    return {
        "your value:" , number
    }

@app.get('/check_get')
async def get_rnc():
    return random.randint(0 , 10)

@app.post('/pronunciation_scoring')
async def upload_audio(file: UploadFile = File(...)):
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        # Write the uploaded file content to the temporary file
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    print("Uploaded file saved to:", temp_file_path)

    # Load the audio file using librosa with a fixed sample rate of 16000 Hz
    audio, sr = librosa.load(temp_file_path, sr=16000)

        # Process the audio data as needed
        # For this example, we'll just return some basic information
    duration = librosa.get_duration(y=audio, sr=sr)
        
    print("filename:" , file.filename)
    print("duration:" , duration)
    print("sample_rate:" , sr)

    os.unlink(temp_file_path)
    # Tokenize audio
    print("Tokenizing audio...")
    input_values = tokenizer(audio, return_tensors="pt").input_values
    
    # Perform inference
    print("Performing inference with Wav2Vec2 model...")
    logits = model(input_values).logits
    
    # Get predictions
    print("Getting predictions...")
    prediction = torch.argmax(logits, dim=-1)
    
    # Decode predictions
    print("Decoding predictions...")
    transcription = tokenizer.batch_decode(prediction)[0]
    
    # Convert transcription to lowercase
    transcription = transcription.lower()
    
    # Print transcription and word counts
    print("Decoded transcription:", transcription)
    incorrect, correct = count_spelled_words(transcription, english_words)
    print("Spelling check - Incorrect words:", incorrect, ", Correct words:", correct)
    
    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    score = round(fraction * 100, 2)
    print("Pronunciation score for", transcription, ":", score)
    
    print("Pronunciation scoring process complete.")
    
    return {
        "transcription": transcription,
        "pronunciation_score": score
    }

@app.post("/upload_audio/")
async def upload_audio(file: UploadFile = File(...)):
    # Read the uploaded file as bytes
    contents = await file.read()
    
    # Load audio from bytes using librosa
    y, sr = librosa.load(io.BytesIO(contents), sr=16000)
    
    # Now you can process the audio data as needed
    # Example: return the sampling rate and length of audio
    return {"sampling_rate": sr, "audio_length": len(y)}