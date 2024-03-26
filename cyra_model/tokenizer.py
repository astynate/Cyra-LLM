import re, os
import unicodedata

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

    def __init__(self, path: str = None, text: str = None, count_iterations: int = 10, special_tokens=[]) -> None:
        self.dictionary = set()
        
        if path != None and os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.dictionary = f.read().split()
                print(f'Cyra tokenizer was loaded, dimention: {len(self.dictionary)}')
        else:
            self.dictionary = self.create_vocabulary(text, count_iterations)
            self.append_special_tokens(special_tokens=special_tokens)

            print(f'Cyra tokenizer was created, dimention: {len(self.dictionary)}')
    
    def cleaning_dataset(self, text: str) -> str:
        text = text.replace(u'\xa0', u' ')
        text = text.lower()

        return re.sub(r'[^\w]', '', text)

    def append_special_tokens(self, special_tokens: list):
        for i in range(0, 0x10FFFF):
            char = chr(i)
            if unicodedata.category(char)[0] in ('L', 'N', 'P', 'Z'):
                self.dictionary.append(char)
        
        self.dictionary += special_tokens
        
    def create_pairs(self, tokens):
        return [tokens[i:i+2] for i in range(len(tokens) - 1)]

    def create_vocabulary(self, text: str, count_iterations: int) -> list:
        print("Cleaning dataset...")

        text = self.cleaning_dataset(text)

        vocabulary = [char for char in text]
        base_vocabulary = list(set(vocabulary))

        print(f'Starting Byte pair encoding, target_size: {count_iterations}')

        for iteration in range(count_iterations):
            pairs = self.create_pairs(vocabulary)
            print(f"Iteration: {iteration + 1} / {count_iterations} | {int(100 * (iteration + 1) / count_iterations)}%")

            pair_to_merge = [pairs.count(i) for i in pairs]
            max_element = max(pair_to_merge)

            if max_element == 1:
                break

            pair_to_merge = pair_to_merge.index(max_element)

            new_token = ''.join(pairs[pair_to_merge])
            i = 0

            while i < len(vocabulary) - 1:

                if (vocabulary[i] == pairs[pair_to_merge][0] 
                    and vocabulary[i + 1] == pairs[pair_to_merge][1]):

                    vocabulary[i] = new_token
                    del vocabulary[i + 1]

                i += 1
            
            base_vocabulary.append(new_token)

        return list(set(base_vocabulary))

    def save_dictionary(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for token in self.dictionary:
                f.write(f'{token}\n')

    def detokenize(self, sequence: list) -> str:
        return ''.join([self.dictionary[token] for token in sequence])

    def tokenize(self, text: str) -> list:
        text = text.lower().split()
        space, sequence = self.dictionary.index(' '), []

        for i in range(len(text)):
            start, end = 0, len(text[i]) + 1
            separater = 0

            while separater < len(text[i]) and end >= 0:

                if text[i][start:end] in self.dictionary:

                    sequence.append(self.dictionary.index(text[i][start:end]))
                    start, separater = end, end
                    end = len(text[i]) + 1

                end -= 1

            if len(text) >= 1 and i != len(text) - 1:
                sequence.append(space)

        return sequence

if __name__ == '__main__':
    path = 'D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002/'
    text = get_text_from_folder(path)

    cyra_tokenizer = CyraTokenizer(text=text, count_iterations=300)
    cyra_tokenizer.save_dictionary('C:/Users/sicom/OneDrive/Рабочий стол/tokenizer.txt')

    test_text = text[:10]

    sequense = cyra_tokenizer.tokenize(test_text)
    original_text = cyra_tokenizer.detokenize(sequense)

    print(cyra_tokenizer.dictionary)

    print(f"Original text: {test_text}")
    print(f"Sequense: {sequense}")
    print(f"Detokenize: {original_text}")