import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from __download__ import get_text_from_folder

def get_text(path) -> str:
    print('Reading text...')

    with open(path, 'r', encoding='utf-8') as f:
        return f.read(10 ** 9).lower()

def clean_text(text: str) -> str:
    print('Cleaning...')

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(' +', ' ', text)

    return text

def preprocess_text(text: str) -> str:
    # text = clean_text(text)
    
    # stop_words = get_text_from_folder('dataset_preparing/input_dataset/stop-words')
    # stop_words = []
    porter_stemmer = PorterStemmer()

    print('Processing...')

    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    # tokens = word_tokenize(text)
    # tokens = [porter_stemmer.stem(token) for token in tokens if token not in stop_words]

    # return ' '.join(tokens)    
    return text

if __name__ == '__main__':

    text: str = get_text('C:/Users/Atynate/Downloads/ru-2.txt')
    result_text: str = clean_text(text)

    print('Saving...')

    with open('C:/Users/Atynate/Downloads/ru-2.txt', 'w', encoding='utf-8') as f:
        f.write(result_text)