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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print("Downloading  pronunciation tokenizer...")
pronunciation_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
print("Downloading  pronunciation model...")
pronunciation_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

pronunciation_model.to(device)

''''''''''''''''''' Loading the Fluency Model and Tokenizer '''''''''''''

print("Downloading  fluency tokenizer...")
fluency_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Downloading fluency model...")
fluency_model = DistilBertForRegression.from_pretrained("Kartikeyssj2/Fluency_Scoring_V2")
print("Download completed.")

fluency_model.to(device)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move the models to the appropriate device
# pronunciation_model.to(device)
# fluency_model.to(device)


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


import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


''''''''''''''''''''''' IMAGE CAPTIONING MODEL '''''''''''''''''

image_captioning_processor = BlipProcessor.from_pretrained("noamrot/FuseCap")

image_captioning_model = BlipForConditionalGeneration.from_pretrained("noamrot/FuseCap").to(device)



''''''''''''''''''' FUNCTIONS FOR PREPROCESSING '''''''''''''''

async def count_misspelled_words(text):
    nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
    nltk.data.path.append(nltk_data_dir)

    english_words = set(words.words())
    words_in_text = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words_in_text)
    misspelled = [word for word in words_in_text if word not in english_words]

    incorrect_count = len(misspelled)

    return f"{(incorrect_count / total_words * 100):.2f}"


async def get_fluency_score(transcription):
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
async def count_spelled_words(text, word_list):
    print("Counting spelled words...")
    # Split the text into words
    words = re.findall(r'\b\w+\b', text.lower())

    correct = sum(1 for word in words if word in word_list)
    incorrect = len(words) - correct

    print("Spelling check complete.")
    return incorrect, correct

# Function to apply spell check to an item (assuming it's a dictionary)
async def apply_spell_check(item, word_list):
    print("Applying spell check...")
    if isinstance(item, dict):
        # This is a single item
        text = item['transcription']
        incorrect, correct = await count_spelled_words(text, word_list)
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


async def get_pronunciation_and_fluency_scores(transcription):

    count_spelled_words_response, fluency_score = await asyncio.gather(
        count_spelled_words(transcription, english_words),
        get_fluency_score(transcription)
    )

    incorrect = count_spelled_words_response[ 0 ]
    correct = count_spelled_words_response[ 1 ]


    # Calculate pronunciation score
    fraction = correct / (incorrect + correct)
    pronunciation_score = round(fraction * 100, 2)

    # Calculate fluency score

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



async def content_score( text1: str, text2 : str ):
    essay_embedding = content_relevance_model.encode( text1 )

    summarization_embedding = content_relevance_model.encode( text2 )

    relevance_score = float(util.dot_score( essay_embedding, summarization_embedding).cpu()[0][0]) * 100

    if(relevance_score >= 40):
        relevance_score = relevance_score + 30

    relevance_score = min( relevance_score , 100 )

    relevance_score = max( 0 , relevance_score )

    return relevance_score



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

    print("transcription: " , transcription , "\n\n")

    ''''''''''''''''''' GET THE PRONUNCIATION AND FLUENCY SCORING '''''''''''''
    ''''''''''''''''''' GET THE CONTENT AND RELEVANCE SCORING '''''''''''''

    result, relevance_scores, incorrect_words_percentage = await asyncio.gather(

            get_pronunciation_and_fluency_scores(transcription),
            content_score(speech_topic, transcription),
            count_misspelled_words(transcription)

    )


    ''''''''''''''''''' PASS THE RAW OUTPUTS TO THE BIASING MODEL '''''''''''''

    base_pronunciation_score = result["pronunciation_score"]
    base_fluency_score = result["fluency_score"]

    base_pronunciation_score = float(base_pronunciation_score)
    base_fluency_score = float(base_fluency_score)
    incorrect_words_percentage = float(incorrect_words_percentage)

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
    inputs = image_captioning_processor(raw_image, context, return_tensors="pt")

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
async def image_description_scoring( context : str = Form() ,  audio_file: UploadFile = File(...) ,image_file: UploadFile = File(...) ):


    ''''''''''''''''''' Get THE RAW TRANSCRIPTION '''''''''''''''''''

    # Save the uploaded file to a temporary location
    temp_file_path = "temp_audio_file.wav"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Transcribe the audio file
    transcription = transcribe_audio(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)

    image_captions = Get_Captions( context , image_captioning_model , image_file)

    result, relevance_scores, incorrect_words_percentage = await asyncio.gather(

            get_pronunciation_and_fluency_scores(transcription),
            content_score( image_captions , transcription),
            count_misspelled_words(transcription)

    )

    ''''''''''''''''''' PASS THE RAW OUTPUTS TO THE BIASING MODEL '''''''''''''

    base_pronunciation_score = result["pronunciation_score"]
    base_fluency_score = result["fluency_score"]

    base_pronunciation_score = float(base_pronunciation_score)
    base_fluency_score = float(base_fluency_score)
    incorrect_words_percentage = float(incorrect_words_percentage)

    print("Base Pronunciation Score:", base_pronunciation_score)
    print("Base fluency Score:", base_fluency_score)
    print("Incorrect Words Percentage:", incorrect_words_percentage)

    final_pronunciation_score = max(0, min(100, linreg_pronunciation.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))
    final_fluency_score = max(0, min(100, linreg_fluency.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))

    print("Base Pronunciation Score:", base_pronunciation_score)
    print("Base fluency Score:", base_fluency_score)
    print("Incorrect Words Percentage:", incorrect_words_percentage)

    final_pronunciation_score = max(0, min(100, linreg_pronunciation.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))
    final_fluency_score = max(0, min(100, linreg_fluency.predict(np.array([[base_pronunciation_score, base_fluency_score, incorrect_words_percentage]]))[0]))

    result["Content Quality and Relevance Score"] = relevance_scores
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
    short_sentences = sum(len(sentence.split()) <= 70 for sentence in sentences if sentence.strip())

    print(" Short Sentences: " , short_sentences )

    # CONSIDER IT A VALID FORMAT IF MORE THAN HALF OF THE SENTENCES ARE SHORT
    return short_sentences >= len(sentences) / 2

async def form_score_summary(summary: str) -> float:
    # CONVERT THE SUMMARY TO UPPERCASE
    summary_upper = summary.upper()

    # REMOVE PUNCTUATION
    summary_clean = re.sub(r'[^\w\s]', '', summary_upper)

    # COUNT THE NUMBER OF WORDS
    word_count = len(summary_clean.split())

    # CHECK IF THE SUMMARY FORMAT IS VALID
    valid_format = is_valid_summary_format(summary)

    print("\n\n word count: ", word_count, " valid_format: ", valid_format)

    # CALCULATE SCORE BASED ON WORD COUNT AND FORMAT
    if valid_format:
        if 45 <= word_count <= 75:
            if word_count < 50:
                score = 50 + (word_count - 45) * (50 / 5)  # Gradual increase from 50
            elif word_count <= 75:
                score = 100  # Best score range
            else:
                score = 100 - (word_count - 70) * (50 / 5)  # Gradual decrease from 100
        else:
            score = 0  # Worst score if word count is out of acceptable range
    else:
        score = 0  # Worst score if format is invalid

    # CLAMP SCORE BETWEEN 0 AND 100

    score = float( score )

    return max(0.0, min(100.0, score))




async def grammar_score(text: str) -> int:
    # Create a TextBlob object
    blob = TextBlob(text)

    # Check for grammatical errors
    errors = 0
    for sentence in blob.sentences:
        if sentence.correct() != sentence:
            errors += 1

    print(" \n\n Number of grammatical errors: " , errors )

    errors *= 5

    result = 100 - errors

    return max( 0 , result)


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

    percentage_correct = round( percentage_correct , 2 )

    return percentage_correct


@app.post("/summarization_scoring/")
async def summarization_score( essay : str = Form() , summarization : str = Form() ):

    content_score_result, form_score_result, grammar_score_result, vocabulary_score_result = await asyncio.gather(
        content_score(essay, summarization),
        form_score_summary(summarization),
        grammar_score(summarization),
        vocabulary_score(summarization)
    )


    return {

        "Content Score: " : content_score_result,
        "Form Score: " : form_score_result,
        "Grammar Score: " : grammar_score_result,
        "Vocabulary Score: " : vocabulary_score_result,
        "Overall Summarization Score: " : round( (content_score_result + form_score_result + grammar_score_result + vocabulary_score_result) / 4 , 2)
    }



'''
transitional words can significantly contribute to the development, structure, and coherence of a text.

    Development: Transitional words help to show how ideas build upon each other and progress
        throughout the essay. They can introduce new points, provide examples, or signal a shift in focus.

    Structure: Transitional words help to organize the text by indicating relationships between
        ideas. They can show cause and effect, compare and contrast, or signal a sequence of events.

    Coherence: Transitional words help to create a smooth flow between sentences and paragraphs,
        making the text easier to understand and follow. They can clarify connections between
        ideas and prevent the text from feeling disjointed.
'''


addition_transitional_words = [
    "and", "also", "too", "in addition", "furthermore", "moreover", "besides", "likewise",
    "similarly", "equally important", "not to mention", "as well as", "what's more",
    "on top of that", "to boot", "in the same way", "by the same token", "similarly",
    "likewise", "in a similar vein", "correspondingly", "at the same time", "concurrently",
    "simultaneously", "not only... but also", "both... and", "as well", "and then",
    "and so forth", "and so on"
]
contrast_transitional_words = [
    "but", "however", "nevertheless", "nonetheless", "on the other hand", "on the contrary",
    "in contrast", "conversely", "although", "though", "even though", "despite", "in spite of",
    "regardless of", "while", "whereas", "yet", "still", "even so", "even if", "at the same time",
    "by the same token", "equally", "in common", "similarly", "just like", "just as", "as well as",
    "resemble", "equally", "in common", "by the same token"
]
cause_effect_transitional_words = [
    "because", "since", "as", "due to", "owing to", "thanks to", "on account of",
    "as a result", "consequently", "therefore", "hence", "thus", "so", "accordingly",
    "for this reason", "as a consequence", "in consequence", "in that case",
    "that being the case", "for that reason", "as a result of", "because of",
    "on account of", "owing to", "due to", "thanks to"
]
time_transitional_words = [
    "first", "second", "third", "next", "then", "after", "before", "later", "earlier",
    "previously", "subsequently", "following", "meanwhile", "simultaneously",
    "at the same time", "concurrently", "in the meantime", "in the interim", "afterwards",
    "thereafter", "finally", "lastly", "ultimately", "in conclusion", "to conclude",
    "in summary", "to sum up"
]
emphasis_transitional_words = [
    "indeed", "in fact", "certainly", "assuredly", "without a doubt", "undoubtedly",
    "unquestionably", "undeniably", "absolutely", "positively", "emphatically",
    "decisively", "strongly", "forcefully", "with conviction", "with certainty",
    "with assurance", "without hesitation", "without question", "without fail", "without doubt"
]
example_transitional_words = [
    "for example", "for instance", "such as", "like", "as an illustration", "to illustrate",
    "to demonstrate", "to exemplify", "namely", "specifically", "in particular",
    "particularly", "especially"
]
conclusion_transitional_words = [
    "in conclusion", "to conclude", "in summary", "to sum up", "finally", "lastly",
    "ultimately", "therefore", "hence", "thus", "so", "accordingly", "as a result",
    "consequently"
]
transition_between_sections_transitional_words = [
    "in the following section", "moving on to", "now", "let's explore",
    "turning our attention to", "to delve deeper", "we will now examine",
    "next", "at this point", "at this juncture", "furthermore", "moreover",
    "in addition"
]
miscellaneous_transition_words_list = [
    # Clarification
    "in other words", "that is to say", "namely", "to put it another way",
    "in simpler terms", "to clarify", "to explain further", "to elaborate",
    "to be more specific", "to be more exact",

    # Concession
    "admittedly", "granted", "of course", "naturally", "it is true that",
    "it must be admitted that", "it cannot be denied that", "it goes without saying that",

    # Digression
    "by the way", "incidentally", "aside from that", "apart from that",

    # Repetition
    "again", "once again", "still", "further", "furthermore", "moreover", "in addition"
]
contrast_within_sentence_transitional_words = [
    "but", "however", "nevertheless", "nonetheless", "on the other hand",
    "in contrast", "conversely", "although", "though", "even though",
    "despite", "in spite of", "regardless of", "while", "whereas",
    "yet", "still", "even so", "even if"
]
comparison_transitional_words = [
    "similarly", "likewise", "in the same way", "equally", "in common",
    "by the same token", "just like", "just as", "as well as", "resemble"
]
cause_and_effect_within_sentence_transitional_words = [
    "because", "since", "as", "due to", "owing to", "thanks to",
    "on account of", "as a result", "consequently", "therefore",
    "hence", "thus", "so", "accordingly", "for this reason",
    "as a consequence", "in consequence", "in that case",
    "that being the case", "for that reason", "as a result of",
    "because of", "on account of", "owing to", "due to", "thanks to"
]
emphasis_within_sentence_transitional_words = [
    "indeed", "in fact", "certainly", "assuredly", "without a doubt",
    "undoubtedly", "unquestionably", "undeniably", "absolutely",
    "positively", "emphatically", "decisively", "strongly", "forcefully",
    "with conviction", "with certainty", "with assurance",
    "without hesitation", "without question", "without fail", "without doubt"
]
concession_digression_repetition_transitional_words = [
    # Concession
    "admittedly", "granted", "of course", "naturally",
    "it is true that", "it must be admitted that",
    "it cannot be denied that", "it goes without saying that",

    # Digression
    "by the way", "incidentally", "aside from that",
    "apart from that",

    # Repetition
    "again", "once again", "still", "further",
    "furthermore", "moreover", "in addition"
]

async def dsc_score( essay: str ):
    # Normalize the essay
    essay_lower = essay.lower()

    # Helper function to count occurrences of transitional words
    def count_transitional_words(word_list):
        return sum(essay_lower.count(word) for word in word_list)

    # Calculate counts for each type of transitional word list
    addition_count = count_transitional_words(addition_transitional_words)
    contrast_count = count_transitional_words(contrast_transitional_words)
    cause_effect_count = count_transitional_words(cause_effect_transitional_words)
    time_count = count_transitional_words(time_transitional_words)
    emphasis_count = count_transitional_words(emphasis_transitional_words)
    example_count = count_transitional_words(example_transitional_words)
    conclusion_count = count_transitional_words(conclusion_transitional_words)
    transition_between_sections_count = count_transitional_words(transition_between_sections_transitional_words)
    misc_count = count_transitional_words(miscellaneous_transition_words_list)
    contrast_within_sentence_count = count_transitional_words(contrast_within_sentence_transitional_words)
    comparison_count = count_transitional_words(comparison_transitional_words)
    cause_and_effect_within_sentence_count = count_transitional_words(cause_and_effect_within_sentence_transitional_words)
    emphasis_within_sentence_count = count_transitional_words(emphasis_within_sentence_transitional_words)
    concession_digression_repetition_count = count_transitional_words(concession_digression_repetition_transitional_words)

    # Calculate total transitional word count
    total_transitional_count = (
        addition_count + contrast_count + cause_effect_count + time_count +
        emphasis_count + example_count + conclusion_count +
        transition_between_sections_count + misc_count +
        contrast_within_sentence_count + comparison_count +
        cause_and_effect_within_sentence_count + emphasis_within_sentence_count +
        concession_digression_repetition_count
    )

    print("\n\n\n Total Transitional Words Count: " , total_transitional_count )

    words = essay.split()
    word_count = len(words)

    transitional_words_percentage = round( (  total_transitional_count / ( word_count * 1.00)  ) * 100  , 2 )

    print("]n\n\n transitional_words_percentage: " , transitional_words_percentage)

    '''
    Since a transition_words_percentage of 10% is considered as the ideal percentage of transitional words in an essay,
    we are deducting points with respect to how much is it deviating from its ideal percentage value.

    This have proven to be powerful to determine the Development, Structure and Coherence in essays

    '''
    return 100 - abs( transitional_words_percentage - 10 )


def is_capitalized(text: str) -> bool:
    """Check if the entire text is in capital letters."""
    return text.isupper()

def contains_punctuation(text: str) -> bool:
    """Check if the text contains any punctuation."""
    return bool(re.search(r'[.,!?;:]', text))

def is_bullet_points(text: str) -> bool:
    """Check if the text consists only of bullet points or very short sentences."""
    sentences = text.split('\n')
    bullet_points = any(line.strip().startswith('-') for line in sentences)
    short_sentences = sum(len(sentence.split()) <= 2 for sentence in sentences if sentence.strip())
    return bullet_points or short_sentences > len(sentences) / 2


async def form_score_essay(essay: str) -> float:
    # REMOVE PUNCTUATION AND COUNT WORDS
    word_count = len(re.findall(r'\b\w+\b', essay))

    # CHECK ESSAY FORMAT
    is_capital = is_capitalized(essay)
    has_punctuation = contains_punctuation(essay)
    bullet_points_or_short = is_bullet_points(essay)

    # CALCULATE SCORE
    if 200 <= word_count <= 300 and has_punctuation and not is_capital and not bullet_points_or_short:
        score = 100.0  # BEST SCORE
    elif (120 <= word_count <= 199 or 301 <= word_count <= 380) and has_punctuation and not is_capital and not bullet_points_or_short:
        score = 50.0  # AVERAGE SCORE
    else:
        score = 0.0  # WORST SCORE

    return score


@app.post("/essay_scoring/")
async def essay_score( prompt : str = Form() , essay : str = Form() ):
    content_score_result, form_score_result, dsc_score_result, grammar_score_result = await asyncio.gather(
        content_score( prompt , essay ),
        form_score_essay( essay ),
        dsc_score( essay ),
        grammar_score( essay )
    )

    print( essay )

    return {

        "Content Score: " : content_score_result,
        "Form Score: " : form_score_result,
        "DSC Score: " : dsc_score_result,
        "Grammar Score: " : grammar_score_result,
        "Overall Essay Score" : ( content_score_result + form_score_result + dsc_score_result + grammar_score_result) / 4.0
    }
