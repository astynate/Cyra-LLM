import os
import glob
import tensorflow as tf
import numpy as np
from cyra_model.model import Lexa

def train(project_path, model) -> None:

    print('Launching the training module...')
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Lexa/Lexa AI/Assistant/dataset/original/dataset_01', '*.txt'))

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

                input_values.append(tokenizer.get_sequences(tokenizer.get_text(sequence[separator:separator + length])))
                valid_values.append(sequence[separator + length])

            separator += length + 1

        train_data = np.array(input_values[0])
        train_labels = tf.keras.utils.to_categorical(np.array(valid_values[0]), num_classes=model.tokenizer.get_dimension())

        print('Starting model training...')

        # for i in range(train_data.shape[0]):

        #     print(lexa.tokenizer.get_text(train_data[i]))
        #     print(lexa.tokenizer.get_text(train_labels[i]))

        # print('---------------------------------------')

        # with tf.device('/GPU:0'):

        model.model.fit(train_data.reshape((1, 50)), train_labels.reshape((1, train_labels.shape[0])), batch_size=1, epochs=1)
        print('Saving model...')
        model.model.save(project_path + '/models/lexa.keras')

        context = model.tokenizer.get_text([np.random.randint(1, 30)])

        print(context)

        for i in range(10):
            
            generated_word = model(context)
            print(str(i) + ' |' + context + '|')

            context += generated_word

if __name__ == '__main__':

    train()