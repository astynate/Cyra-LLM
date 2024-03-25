import re, os

def load_dataset(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

class CyraTokenizer:
    def __init__(self, path=None, text=None, count_iterations=10, *args, **kwargs) -> None:
        self.dictionary = []
        
        if path != None and os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self.dictionary = f.read().split()
        else:
            self.dictionary = self.create_vocabulary(text, count_iterations)
    
    def cleaning_dataset(self, text: str) -> str:
        text = text.replace(u'\xa0', u' ')
        text = text.lower()

        return re.sub(r'[^\w]', '', text)

    def create_pairs(self, tokens):
        return [tokens[i:i+2] for i in range(len(tokens) - 1)]

    def create_vocabulary(self, text: str, count_iterations: int) -> list:

        print("Cleaning dataset...")

        text = self.cleaning_dataset(text)

        vocabulary = [char for char in text]
        base_vocabulary = list(set(vocabulary))

        print(f'Starting Byte pair encoding, iterations: {count_iterations}')

        for iteration in range(count_iterations):

            pairs = self.create_pairs(vocabulary)

            print(f"Iteration: {iteration}, {100 * (iteration + 1) / count_iterations}%")

            pair_to_merge = [pairs.count(i) for i in pairs]
            pair_to_merge = pair_to_merge.index(max(pair_to_merge))

            new_token = ''.join(pairs[pair_to_merge])

            i = 0

            while i < len(vocabulary):

                if (vocabulary[i] == pairs[pair_to_merge][0] 
                    and vocabulary[i + 1] == pairs[pair_to_merge][1]):

                    vocabulary[i] = new_token
                    vocabulary.pop(i + 1)

                i += 1
            
            base_vocabulary.append(new_token)

        return base_vocabulary
    
    def create_sequence_from_text(self, text: str) -> list:
        text = text.lower().split()
        sequence = []

        for word in text:
            start, end = 0, len(word)
            separater = 0

            while separater < len(word) and end >= 0:
            #     print(word[start:end])
                if word[start:end] in self.dictionary:
                    print(word[start:end])
            #     #     sequence.append(self.dictionary.index(word[start:end]))
            #     #     start = end
            #     #     separater = end

                end -= 1

        return sequence

if __name__ == '__main__':
    path = 'D:/Exider Company/Cyra/dataset_preparing/input_dataset/dataset-001/20150302bionics.txt'

    text = load_dataset(path)
    cyra_tokenizer = CyraTokenizer(text=text)

    print(cyra_tokenizer.dictionary)
    print(cyra_tokenizer.create_sequence_from_text(text[:30]))