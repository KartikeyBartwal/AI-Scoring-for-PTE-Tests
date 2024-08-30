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

from textblob import TextBlob
import nltk

nltk.download('punkt_tab')
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = "cpu"

pronunciation_model.to(device)
fluency_model.to(device)

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


import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


''''''''''''''''''''''' IMAGE CAPTIONING MODEL '''''''''''''''''
image_captioning_processor = BlipProcessor.from_pretrained("noamrot/FuseCap")

image_captioning_model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)



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

    input_values = pronunciation_tokenizer(audio, return_tensors="pt").input_values.to(device)


    logits = pronunciation_model(input_values).logits

    prediction = torch.argmax(logits, dim = -1)
    transcription = pronunciation_tokenizer.batch_decode(prediction)[0]

    return transcription.lower()


app = FastAPI()

@app.post("/pronunciation_fluency_content_scoring/")
async def speech_scoring(speech_topic: str = Form(), audio_file: UploadFile = File(...)):

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



def Get_Captions(context: str , image_captioning_model, image_file):

    # Open and convert the image
    raw_image = Image.open(image_file.file).convert('RGB')

    context = "Describe this image, "
    # Prepare the inputs
    inputs = image_captioning_processor(raw_image, context, return_tensors="pt").to(device)

    print("Generating the output ")

    # Generate the caption
    out = image_captioning_model.generate(**inputs, num_beams=5)

    # Decode and return the caption
    caption = image_captioning_processor.decode(out[0], skip_special_tokens=True)


    return caption


# @app.post("/get_image_description/")
# async def image_captioning(context: str, image_file: UploadFile = File(...)):

#     image_caption_text = Get_Captions(context, image_captioning_model, image_file)

#     return {"image_captions" : context + " " + image_caption_text}


@app.post("/image_description_scoring/")
async def speech_scoring( context : str = Form() ,  audio_file: UploadFile = File(...) ,image_file: UploadFile = File(...) ):


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

    ''''''''''''''''''' GET THE CONTENT AND RELEVANCE SCORING WITH RESPECT TO THE IMAGE '''''''''''''


    image_caption = Get_Captions( context , image_captioning_model , image_file)

    image_caption_embedding = content_relevance_model.encode( image_caption )

    transcription_text_embeddings = content_relevance_model.encode( transcription )

    image_description_score = float(util.dot_score(image_caption_embedding, transcription_text_embeddings).cpu()[0][0]) * 100


    ''''''''''''''''''' PASS THE RAW OUTPUTS TO THE BIASING MODEL '''''''''''''

    base_pronunciation_score = result["pronunciation_score"]
    base_fluency_score = result["fluency_score"]
    incorrect_words_percentage = count_misspelled_words(transcription)

    base_pronunciation_score = float(base_pronunciation_score)
    base_fluency_score = float(base_fluency_score)
    incorrect_words_percentage = float(incorrect_words_percentage)


    print("Base Pronunciation Score:", base_pronunciation_score)
    print("Base fluency Score:", base_fluency_score)
    print("Incorrect Words Percentage:", incorrect_words_percentage)

    final_pronunciation_score = max(0, min(100, linreg_pronunciation.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))
    final_fluency_score = max(0, min(100, linreg_fluency.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))

    result["Content Quality and Relevance Score"] = image_description_score
    result["pronunciation_score"] = final_pronunciation_score
    result["fluency_score"] = final_fluency_score

    return result


# @app.post("/transcribe_audio/")
# async def transcribe( audio_file: UploadFile = File(...) ):
#     temp_file_path = "temp_audio_file.wav"
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(audio_file.file, buffer)

#     # Transcribe the audio file
#     transcription = transcribe_audio(temp_file_path)

#     return {"transcription" : transcription}

import string
import asyncio


async def is_valid_summary_format(summary: str) -> bool:
    # CHECK IF THE SUMMARY CONTAINS ONLY BULLET POINTS
    if '-' in summary or '*' in summary:
        return True

    # CHECK IF THE SUMMARY CONSISTS ONLY OF VERY SHORT SENTENCES
    sentences = re.split(r'[.!?]', summary)
    short_sentences = sum(len(sentence.split()) <= 2 for sentence in sentences if sentence.strip())

    # CONSIDER IT A VALID FORMAT IF MORE THAN HALF OF THE SENTENCES ARE SHORT
    return short_sentences >= len(sentences) / 2

async def form_score(summary: str) -> int:
    # CONVERT THE SUMMARY TO UPPERCASE
    summary_upper = summary.upper()

    # REMOVE PUNCTUATION
    summary_clean = re.sub(r'[^\w\s]', '', summary_upper)

    # COUNT THE NUMBER OF WORDS
    word_count = len(summary_clean.split())

    # CHECK IF THE SUMMARY FORMAT IS VALID
    valid_format = is_valid_summary_format(summary)

    # CALCULATE SCORE BASED ON WORD COUNT AND FORMAT
    if 50 <= word_count <= 70 and valid_format:
        return 100  # BEST SCORE
    elif (40 <= word_count <= 49 or 71 <= word_count <= 100) and valid_format:
        return 50   # AVERAGE SCORE
    else:
        return 0    # WORST SCORE




async def grammar_score(text: str) -> int:
    # Create a TextBlob object
    blob = TextBlob(text)

    # Check for grammatical errors
    errors = 0
    for sentence in blob.sentences:
        if sentence.correct() != sentence:
            errors += 1

    # Determine the score based on the number of errors
    if errors == 0:
        return 100  # BEST SCORE
    elif errors <= 2:
        return 50   # AVERAGE SCORE
    else:
        return 0    # WORST SCORE

async def vocabulary_score(text: str) -> float:
    # Create a TextBlob object
    blob = TextBlob(text)

    # Extract words from the text
    words = blob.words

    # Count the total words and correctly spelled words
    total_words = len(words)
    correctly_spelled = sum(1 for word in words if word == TextBlob(word).correct())

    # Calculate the percentage of correctly spelled words
    if total_words == 0:
        return 0.0  # Avoid division by zero if there are no words

    percentage_correct = (correctly_spelled / total_words) * 100

    percentage_correct = min( percentage_correct , 100)
    percentage_correct = max( 0 , percentage_correct )

    return percentage_correct



async def content_score( essay: str, summarization : str ):
    essay_embedding = content_relevance_model.encode( essay )

    summarization_embedding = content_relevance_model.encode( summarization )

    relevance_score = float(util.dot_score( essay_embedding, summarization_embedding).cpu()[0][0]) * 100

    if(relevance_score >= 40):
        relevance_score = relevance_score + 30

    relevance_score = min( relevance_score , 100 )

    relevance_score = max( 0 , relevance_score )

    return relevance_score



@app.post("/summarization_scoring/")
async def summarization_score( essay : str = Form() , summarization : str = Form() ):

    content_score_result, form_score_result, grammar_score_result, vocabulary_score_result = await asyncio.gather(
        content_score(essay, summarization),
        form_score(summarization),
        grammar_score(summarization),
        vocabulary_score(summarization)
    )


    return {

        "Content Score: " : content_score_result,
        "Form Score: " : form_score_result,
        "Grammar Score: " : grammar_score_result,
        "Vocabulary Score: " : vocabulary_score_result,
        "Summarization Score: " : (content_score_result + form_score_result + grammar_score_result + vocabulary_score_result) / 4
    }
