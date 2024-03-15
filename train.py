import os
import glob
import tensorflow as tf
import numpy as np
from cyra_model.model import Cyra
from cyra_model.tokenizer import CyraTokenizer

def train(model) -> None:

    print('Launching the training module...')
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002', '*.txt'))
    
    print('Loading a dataset...')

    for txt_file in txt_files:

        print('Reading a file...')

        with open(txt_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        sequence = model.tokenizer.get_full_sequence(text)
        
        input_values = []
        valid_values = []

        separator = 0
        tokenizer = model.tokenizer

        print('Partitioning into training samples...')

        while len(sequence) > separator + 51:

            length = np.random.randint(2, 50)

            if model.tokenizer.get_text([sequence[separator + length]]) != '\ufffd':

                input_values.append(tokenizer.get_sequence(tokenizer.get_text(sequence[separator:separator + length])))
                valid_values.append(sequence[separator + length])

            separator += length + 1

        train_data = np.array(input_values)
        train_labels = tf.keras.utils.to_categorical(np.array(valid_values), num_classes=model.tokenizer.get_dimension())

        print('Starting model training...')
        
        model.model.fit(train_data, train_labels, batch_size=1, epochs=1)
        model.model.save('trained-models/cyra.keras')

        context = model.tokenizer.get_text([np.random.randint(1, 30)])

        print(context)

        for i in range(10):
            
            generated_word = model(context)
            print(str(i) + ' |' + context + '|')

            context += generated_word

if __name__ == '__main__':

    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 12, 1024, 16, 1024)

    train(cyra_model)