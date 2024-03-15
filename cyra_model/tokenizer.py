import os
import pickle
import tensorflow_datasets as tfds
from keras.preprocessing.sequence import pad_sequences

"""
Initializing the tokenizer. If the file with
the tokenizer exists, it is loaded.
Otherwise, a new tokenizer is created based on
provided data set.
"""
class CyraTokenizer:

    def __init__(self, path: str, sequence_length=50, dataset=None) -> None:

        self.sequence_length = sequence_length

        if os.path.exists(path):

            with open(path, 'rb') as f:

                self.tokenizer = pickle.load(f)
                print(f'Cyra Tokenizer was loaded, tokens: {self.get_dimension()}')

        elif dataset is None:

            raise ValueError("No dataset provided to train a new tokenizer.")
        
        else:

            self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (text for text in dataset.split()), 
                target_vocab_size=2**14
            )

            print(f'Cyra Tokenizer was created, tokens: {self.get_dimension()}')

            with open(path, 'wb') as f:

                print(f'Saving tokenizer...')
                pickle.dump(self.tokenizer, f)

    def get_full_sequence(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def get_sequence(self, text: str) -> list:
        return pad_sequences([self.tokenizer.encode(text)], maxlen=self.sequence_length, padding='post')[0]

    def get_text(self, sequences: list) -> str:
        return self.tokenizer.decode(sequences) 
    
    def get_dimension(self) -> int:
        return self.tokenizer.vocab_size

'''
Loads a dataset from text files 
in the specified directory.
'''
def load_dataset(path: str) -> str:
    combined_text = ""
    
    for filename in os.listdir(path):
        print(f'Loading: {filename}')

        if filename.endswith(".txt"):
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:

                combined_text += file.read() + " "

    return combined_text

'''
Train a tokenizer based on a data 
set and save it to a file.
'''
def train_tokenizer(project_path: str, name: str):

    print(f'Starting training tokenizer...')
    tokenizer_path: str = f'{project_path}/trained-models/{name}.pickle'

    print(f'Loading text data...')
    text_dataset: str = load_dataset(f'{project_path}/dataset-preparing/input_dataset/dataset-001').lower()

    tokenizer = CyraTokenizer(tokenizer_path, dataset=text_dataset)

    test_token = tokenizer.get_sequence('привет как дела')
    original_text = tokenizer.get_text(test_token)

    print(test_token)
    print(original_text)

def print_all_tokens(tokenizer: CyraTokenizer) -> None:

    for token in range(tokenizer.get_dimension()):
        print(tokenizer.get_text([token]))

'''
If this model runs independently, 
we train the tokenizer
'''
if __name__ == '__main__':

    train_tokenizer('D:/Exider Company/Cyra/', 'cyra_tokenizer')

    # tokenizer = CyraTokenizer('D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle')
    # print_all_tokens(tokenizer)