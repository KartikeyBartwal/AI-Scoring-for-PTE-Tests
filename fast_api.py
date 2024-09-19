import whisper
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import gensim.downloader as api
from gensim.models import KeyedVectors
import torch
import pickle
import numpy as np

word2vec_model = KeyedVectors.load("word2vec-google-news-300.model")

model = whisper.load_model("tiny")

# Load the saved state dictionary
model_state = torch.load("whisper_tiny_model.pt")

# Load the state dictionary into the model
model.load_state_dict(model_state)

def load_model(pickle_file_path: str):
    """Load a model from a pickle file."""
    with open(pickle_file_path, 'rb') as file:
        model = pickle.load(file)
    return model


pronunciation_fluency_model = load_model("pronunciation_fluency_v2.pkl")

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


def Get_P_F_Score( transcription : str ):
    words = transcription.split()

    cumulative_vector_representation = [0] * 300
    for word in words:
        if word in word2vec_model:
            cumulative_vector_representation += word2vec_model[word]

    print( cumulative_vector_representation[ 0 : 5] )

    print( len( cumulative_vector_representation) )

    if np.any(np.isnan(cumulative_vector_representation)):
        print("Input contains NaN values, handle missing values before prediction.")


    print("\n\n")

    output = pronunciation_fluency_model.predict( [ cumulative_vector_representation] )

    print( output )

    return output


def get_average_vector(sentence):
    # TOKENIZE THE SENTENCE INTO WORDS
    words = sentence.lower().split()

    # FILTER OUT WORDS NOT IN THE WORD2VEC VOCABULARY
    valid_words = [word for word in words if word in word2vec_model]

    # RETURN ZERO VECTOR IF NO VALID WORDS FOUND
    if not valid_words:
        return np.zeros(word2vec_model.vector_size)

    # COMPUTE AVERAGE VECTOR FOR VALID WORDS
    return np.mean([word2vec_model[word] for word in valid_words], axis=0)

from sklearn.metrics.pairwise import cosine_similarity

def get_similarity_score(topic, transcription ):
    # GET AVERAGE VECTORS FOR BOTH STRINGS
    topic_vector = get_average_vector(topic)
    transcription_vector = get_average_vector(transcription)

    print("topic vector: " , topic_vector)

    print(" transcription vector: " , transcription_vector )

    # RESHAPE VECTORS FOR COSINE SIMILARITY
    topic_vector = topic_vector.reshape(1, -1)
    transcription_vector = transcription_vector.reshape(1, -1)

    print(" reshaping done ")

    # COMPUTE COSINE SIMILARITY
    similarity = cosine_similarity(topic_vector, transcription_vector)

    print(" Similarity: " , similarity )

    output = similarity[ 0 ][ 0 ]

    output = max( output , 0 )

    output = min( 100 , output )

    # RETURN SIMILARITY SCORE (IT'S A SINGLE VALUE)
    return output



@app.post("/pronunciation_fluency_score")

async def pronunciation_fluency_scoring(
    file: UploadFile = File(...),
    topic: str = File(...)
):
    # SAVE THE UPLOAD FILE TEMPORARILY
    with open(file.filename, "wb") as buffer:

        buffer.write(await file.read())

    # TRANSCRIBE THE AUDIO
    transcription = transcribe(file.filename, model)

    pronunciation_fluency_score = Get_P_F_Score( transcription )

    print( pronunciation_fluency_score)

    print( type( pronunciation_fluency_score ) )

    content_score = get_similarity_score( topic , transcription) * 100




    return {

        "pronunciation score" : pronunciation_fluency_score[ 0 ][ 0 ] * 10 ,
        "fluency score" : pronunciation_fluency_score[ 0 ][ 1 ] * 10 ,
        "content score" : content_score
    }



import string
import asyncio
import re
from textblob import TextBlob
import nltk

def is_valid_summary_format(summary: str) -> bool:
    # CHECK IF THE SUMMARY CONTAINS ONLY BULLET POINTS
    if '-' in summary or '*' in summary:
        return True

    # CHECK IF THE SUMMARY CONSISTS ONLY OF VERY SHORT SENTENCES
    sentences = re.split(r'[.!?]', summary)
    short_sentences = sum(len(sentence.split()) <= 70 for sentence in sentences if sentence.strip())

    print(" Short Sentences: " , short_sentences )

    # CONSIDER IT A VALID FORMAT IF MORE THAN HALF OF THE SENTENCES ARE SHORT
    return short_sentences >= len(sentences) / 2

def form_score_summary(summary: str) -> float:
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




def grammar_score(text: str) -> int:
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


def vocabulary_score(text: str) -> float:

    print(" Performing vocabulary score \n\n")

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


    print(" Percentage Correct: " , percentage_correct )


    return percentage_correct


@app.post("/summarization_scoring/")
def summarization_score( essay : str = Form() , summarization : str = Form() ):

    content_score_result, form_score_result, grammar_score_result, vocabulary_score_result = (
        float( get_similarity_score(essay, summarization) ) * 100,
        float( form_score_summary(summarization) ),
        float( grammar_score(summarization) ),
        float( vocabulary_score(summarization) )
    )

    print(" Completed \n\n\n ")

    response  = {

        "Content Score: " : content_score_result ,
        "Form Score: " : form_score_result  ,
        "Grammar Score: " : grammar_score_result ,
        "Vocabulary Score: " : vocabulary_score_result ,
        "Overall Summarization Score: " : round( (content_score_result + form_score_result + grammar_score_result + vocabulary_score_result) / 4 , 2)
    }

    print( response )

    return response



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

def dsc_score( essay: str ):
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


def form_score_essay(essay: str) -> float:
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
    content_score_result, form_score_result, dsc_score_result, grammar_score_result = (
        float( get_similarity_score( prompt , essay ) ) * 100,
        float( form_score_essay( essay ) ),
        float( dsc_score( essay ) ),
        float( grammar_score( essay ) )
    )

    print( essay )

    return {

        "Content Score: " : content_score_result,
        "Form Score: " : form_score_result,
        "DSC Score: " : dsc_score_result,
        "Grammar Score: " : grammar_score_result,
        "Overall Essay Score" : ( content_score_result + form_score_result + dsc_score_result + grammar_score_result) / 4.0
    }
