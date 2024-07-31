import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from transformers import AutoTokenizer, DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import torch
import shutil
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from transformers import AutoTokenizer, DistilBertPreTrainedModel, DistilBertModel, DistilBertTokenizer
import torch.nn as nn
import torch
import streamlit as st
import soundfile as sf
import numpy as np
import warnings 
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import requests 
import re  
import tempfile  
import os 
import pyarrow as pa
import json 
import joblib
import re
import nltk
from nltk.corpus import words
import pickle
import sys 
from sentence_transformers import SentenceTransformer, util


nltk.download('words')

''''''''''''''''''''''''' Skeletal Structure for the Models '''''''''''''''''''''''''''

class DistilBertForRegression(DistilBertPreTrainedModel):

    def __init__(self, config):

        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = nn.Linear(config.hidden_size, 1)

        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()
        

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None):

        distilbert_output = self.distilbert ( 

            input_ids=input_ids,

            attention_mask=attention_mask,

            head_mask=head_mask,

            inputs_embeds=inputs_embeds,

        )

        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)

        pooled_output = hidden_state[:, 0]  # (bs, dim)

        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)

        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)

        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        logits = self.classifier(pooled_output)  # (bs, 1)

        return logits


''''''''''''''' Loading the Pronunciation Model and Tokenizer '''''''''

print("Downloading  pronunciation tokenizer...")
pronunciation_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
print("Downloading  pronunciation model...")
pronunciation_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


''''''''''''''''''' Loading the Fluency Model and Tokenizer '''''''''''''

print("Downloading  fluency tokenizer...")
fluency_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Downloading fluency model...")
fluency_model = DistilBertForRegression.from_pretrained("Kartikeyssj2/Fluency_Scoring_V2")
print("Download completed.")



''''''''''''''''''''' LOADING THE BIASING MODELS '''''''''''''''

def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

linreg_fluency = load_pickle_file("fluency_model_biasing.pkl")
linreg_pronunciation = load_pickle_file("pronunciation_model_biasing.pkl")


'''''''''''''''''''''' Load the Content Relevance and Scoring Model '''''''''''''''

content_relevance_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')

print(linreg_fluency)
print(linreg_pronunciation)
print(content_relevance_model)

# BEDI'S TALK
print(linreg_fluency.predict(np.array([[83.33 , 45.23 , 23.33]])))

print(linreg_pronunciation.predict(np.array([[83.33 , 45.23 , 23.33]])))



''''''''''''''''''' FUNCTIONS FOR PREPROCESSING '''''''''''''''

def count_misspelled_words(text):
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(nltk_data_dir)
    
    english_words = set(words.words())
    words_in_text = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words_in_text)
    misspelled = [word for word in words_in_text if word not in english_words]

    incorrect_count = len(misspelled)

    return f"{(incorrect_count / total_words * 100):.2f}"


def get_fluency_score(transcription):
    tokenized_text = fluency_tokenizer(transcription, return_tensors="pt")
    with torch.no_grad():
        output = fluency_model(**tokenized_text)
    fluency_score = output.item()
    return round(fluency_score, 2)

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


def get_pronunciation_score(transcription):
    
    incorrect, correct = count_spelled_words(transcription, english_words)
    
    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    score = round(fraction * 100, 2)
    
    return {
        "transcription": transcription,
        "pronunciation_score": score
    }

def get_pronunciation_and_fluency_scores(transcription):
    
    incorrect, correct = count_spelled_words(transcription, english_words)
    
    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    pronunciation_score = round(fraction * 100, 2)
    
    # Calculate fluency score
    fluency_score = get_fluency_score(transcription)
    
    return {
        "transcription": transcription,
        "pronunciation_score": pronunciation_score,
        "fluency_score": fluency_score,
        "Content Quality and Relevance Score": 0
    }

def transcribe_audio(audio_path):
    warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")
    warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")
    
    # Load audio file
    audio, sample_rate = sf.read(audio_path)
    
    # Check if the audio is mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Resample if needed (Wav2Vec2 expects 16kHz)
    if sample_rate != 16000:
        # Simple resampling (less accurate but doesn't require librosa)
        audio = np.array(audio[::int(sample_rate/16000)])
    
    input_values = pronunciation_tokenizer(audio, return_tensors = "pt").input_values

    logits = pronunciation_model(input_values).logits
    
    prediction = torch.argmax(logits, dim = -1)
    transcription = pronunciation_tokenizer.batch_decode(prediction)[0]
    
    return transcription.lower()


app = FastAPI()

@app.post("/unscripted_speech_scoring/")
@app.post("/unscripted_speech_scoring/")
async def speech_scoring(speech_topic: str, audio_file: UploadFile = File(...)):
    
    ''''''''''''''''''' Get THE RAW TRANSCRIPTION '''''''''''''''''''
    
    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    # Transcribe the audio file
    transcription = transcribe_audio(temp_file_path)
    
    # Clean up the temporary file
    os.remove(temp_file_path)

    ''''''''''''''''''' GET THE PRONUNCIATION AND FLUENCY SCORING '''''''''''''
    result = get_pronunciation_and_fluency_scores(transcription)

    print("*" * 50)
            
    print("transcription:" , transcription)

    print("*" * 50)

    ''''''''''''''''''' GET THE CONTENT AND RELEVANCE SCORING '''''''''''''
    topic_text_embeddings = content_relevance_model.encode(speech_topic)
    transcription_text_embeddings = content_relevance_model.encode(transcription)
    relevance_scores = float(util.dot_score(topic_text_embeddings, transcription_text_embeddings).cpu()[0][0]) * 100

    print("relevance scores:" , relevance_scores) 

    ''''''''''''''''''' PASS THE RAW OUTPUTS TO THE BIASING MODEL '''''''''''''

    base_pronunciation_score = result["pronunciation_score"]
    base_fluency_score = result["fluency_score"]
    incorrect_words_percentage = count_misspelled_words(transcription)

    base_pronunciation_score = float(base_pronunciation_score)
    base_fluency_score = float(base_fluency_score)
    incorrect_words_percentage = float(incorrect_words_percentage)

    if(relevance_scores >= 40):
        relevance_scores = relevance_scores + 30

    print("Base Pronunciation Score:", base_pronunciation_score)
    print("Base fluency Score:", base_fluency_score)
    print("Incorrect Words Percentage:", incorrect_words_percentage)
            
    final_pronunciation_score = max(0, min(100, linreg_pronunciation.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))
    final_fluency_score = max(0, min(100, linreg_fluency.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))          
   
    result["Content Quality and Relevance Score"] = relevance_scores
    result["pronunciation_score"] = final_pronunciation_score
    result["fluency_score"] = final_fluency_score

    return result