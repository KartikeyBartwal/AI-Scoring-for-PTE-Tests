import requests


def download_word_list():
    print("Downloading English word list...")
    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    response = requests.get(url)
    words = set(response.text.split())
    print("Word list downloaded.")
    return words

english_words = download_word_list()