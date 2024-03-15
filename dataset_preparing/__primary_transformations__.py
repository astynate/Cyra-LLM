import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from __download__ import get_text_from_folder

def clean_text(text: str) -> str:

    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(' +', ' ', text)

    return text

def preprocess_text(text: str) -> str:

    text = clean_text(text)
    nltk.download('punkt')

    stop_words = get_text_from_folder('dataset_preparing/input_dataset/stop-words')
    porter_stemmer = PorterStemmer()

    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)

    tokens = word_tokenize(text)
    tokens = [porter_stemmer.stem(token) for token in tokens if token not in stop_words]

    return ' '.join(tokens)    

if __name__ == '__main__':

    text: str = get_text_from_folder('dataset_preparing/input_dataset/dataset-002')
    result_text: str = preprocess_text(text)

    print(result_text)