import re, os
import unicodedata
from collections import Counter

def get_text_from_folder(folder_path) -> None:
    combined_text = ""
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            print(f'Reading: {filename}')
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                combined_text += file.read() + " "
    
    return combined_text

def load_dataset(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

class CyraTokenizer:

    """

    Initializing the tokenizer. If the file with
    the tokenizer exists, it is loaded.
    Otherwise, a new tokenizer is created based on
    provided data set.

    """

    def __init__(
            self, 
            path: str = None, 
            text: str = None, 
            count_iterations: int = 10,
            sequence_length=50,
            special_tokens=['<sentence>', '</sentence>', '<textend>', '<padding>', '<unknown>', ' ']
        ):
        
        self.dictionary = set()
        self.sequence_length = sequence_length
        self.special_tokens = special_tokens

        if path is not None and os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.dictionary = f.read().split('\n')
                print(f'Cyra tokenizer was loaded, dimention: {len(self.dictionary)}')
        else:
            self.dictionary = self.create_vocabulary(text, count_iterations)
            self.append_special_tokens(special_tokens=special_tokens)

            self.dictionary = list(set(self.dictionary))
            print(f'Cyra tokenizer was created, dimention: {len(self.dictionary)}')
    
    def cleaning_dataset(self, text: str) -> str:
        print("Cleaning dataset...")

        text = text.replace(u'\xa0', u' ')
        text = text.lower()

        return re.sub(r'[^\w]', ' ', text)

    def append_special_tokens(self, special_tokens: list):

        # Lowercase ASCII

        for i in range(97, 123):
                char = chr(i)
                self.dictionary.append(char)
            
        # Lowercase Russian alphabet
                
        for i in range(1072, 1104):
            char = chr(i)
            self.dictionary.append(char)

        # Lowercase separation sumbols
            
        for i in range(0, 0x10FFFF):
            char = chr(i)
            if unicodedata.category(char)[0] in ('Z', 'P'):
                self.dictionary.append(char)

        self.dictionary += special_tokens
        
    def create_pairs(self, tokens):
        return [tokens[i:i+2] for i in range(len(tokens) - 1) if tokens[i] != ' ' and tokens[i + 1] != ' ']

    def create_vocabulary(self, text: str, count_iterations: int) -> list:
        text = self.cleaning_dataset(text)

        vocabulary = [char for char in text]
        base_vocabulary = list(set([char for char in text]))

        print(f'Starting Byte pair encoding, target_size: {count_iterations}')

        for iteration in range(count_iterations):
            print(f"Iteration: {iteration + 1} / {count_iterations} | {int(100 * (iteration + 1) / count_iterations)}%")
            pairs = self.create_pairs(vocabulary)

            if len(pairs) < 1:
                break

            counter = Counter(map(tuple, pairs))
            pair_to_merge = [counter[tuple(i)] for i in pairs]
            max_element = max(pair_to_merge)

            if max_element == 1:
                break

            pair_to_merge = pair_to_merge.index(max_element)
            new_token = ''.join(pairs[pair_to_merge])
            new_vocabulary = []

            i = 0

            while i < len(vocabulary):
                if (i < len(vocabulary) - 1 and 
                    vocabulary[i] == pairs[pair_to_merge][0] and 
                    vocabulary[i + 1] == pairs[pair_to_merge][1]):
                    
                    new_vocabulary.append(new_token)
                    i += 2

                else:
                    new_vocabulary.append(vocabulary[i])
                    i += 1

            vocabulary = new_vocabulary
            base_vocabulary.append(new_token)

        return base_vocabulary

    def save_dictionary(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for token in self.dictionary:
                f.write(f'{token}\n')

    def get_text(self, sequence: list) -> str:
        return ''.join([self.dictionary[token] for token in sequence])

    def get_full_sequence(self, text: str) -> list:
        sentences = re.split('(?<=[.!?]) +', text)
        sentences = ['<sentence>' + sentence + '</sentence>' for sentence in sentences]
        text = ' '.join(sentences)
        text = text.lower()
        space, unknown_token = self.dictionary.index(' '), self.dictionary.index('<unknown>')
        sequence = []

        for i in range(len(text)):
            start, end = 0, len(text[i]) + 1
            separater = 0

            while separater < len(text[i]) and end >= 0:
                if text[i][start:end] in self.dictionary:
                    sequence.append(self.dictionary.index(text[i][start:end]))
                    start, separater = end, end
                    end = len(text[i]) + 1
                
                elif end - start == 1:
                    sequence.append(unknown_token)
                    start, separater = end, end
                    end = len(text[i]) + 1

                end -= 1

            if len(text) >= 1 and i != len(text) - 1:
                sequence.append(space)

        return sequence
    
    def get_dimension(self):
        return len(self.dictionary)
    
    def get_sequence(self, text):
        sequence = self.get_full_sequence(text)

        if len(sequence) > self.sequence_length:
            return sequence[:50]
        
        elif len(sequence) < self.sequence_length:
            padding_token = self.dictionary.index('<padding>')
            padding = [padding_token] * (self.sequence_length - len(sequence))
            sequence.extend(padding)

        return sequence

if __name__ == '__main__':
    path = 'D:/Exider Company/Cyra/dataset_preparing/input_dataset/dataset-002'
    text = get_text_from_folder(path)

    cyra_tokenizer = CyraTokenizer(path='D:/Exider Company/Cyra/trained-models/tokenizer.txt', count_iterations=2**11)
    # cyra_tokenizer.save_dictionary('D:/Exider Company/Cyra/trained-models/tokenizer.txt')

    test_text = text[:20] + '😀'

    sequense = cyra_tokenizer.get_sequence(test_text)
    original_text = cyra_tokenizer.get_text(sequense)

    print(f"Original text: {test_text} | count words: {len(test_text.split())}")
    print(f"Sequense: {sequense} | count tokens: {len(sequense)}")
    print(f"Detokenize: {original_text}")