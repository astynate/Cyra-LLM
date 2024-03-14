import os
import pickle
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_dataset(path: str) -> list:
    with open(path, encoding='utf-8') as data:
        dataset = data.read().lower()

    return dataset.split()

class CyraTokenizer:

    def __init__(self, path: str, **kwargs) -> None:

        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.tokenizer = pickle.load(f)
        else:
            
            dataset = kwargs.get('dataset')

            if dataset is not None:
                self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                    (text for text in dataset), target_vocab_size=2**13)
                
                with open(path, 'wb') as f:
                    pickle.dump(self.tokenizer, f)
            else:
                raise ValueError("No existing tokenizer found and no dataset provided to train a new one.")


    def get_full_sequence(self, text: str) -> list:
        return self.tokenizer.encode(text)

    def get_sequences(self, text: str) -> list:
        sequences = self.tokenizer.encode(text)
        return pad_sequences([sequences], maxlen=50, padding='post')[0]

    def get_text(self, sequences: list) -> str:
        return self.tokenizer.decode(sequences) 
    
    def get_dimension(self) -> int:
        return self.tokenizer.vocab_size

if __name__ == '__main__':

    model_path: str = 'D:/Exider Company/Lexa/Lexa AI/Assistant/models/lexa_tokenizer.pickle'
    dataset = load_dataset('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/book_one.txt')

    tokenizer = CyraTokenizer(model_path, dataset=dataset)

    test_token = tokenizer.get_sequences('привет как дела')
    original_text = tokenizer.get_text(test_token)

    for token in range(tokenizer.get_dimension()):
        print(tokenizer.get_text([token]))

    print(len(dataset))

    print(test_token)
    print(original_text)

    print("Количество токенов в токенизаторе: ", tokenizer.tokenizer.vocab_size)