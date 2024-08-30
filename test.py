from textblob import TextBlob
import nltk

nltk.download('punkt_tab')

def grammar_score(text: str) -> int:
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


text = input("Enter the text: ")
score = grammar_score(text)
print(f"The grammar score for the text is: {score}")
