import os
import glob
from typing import Tuple
import numpy as np
import random
import math
from cyra_model.model import Cyra
from cyra_model.tokenizer import CyraTokenizer
from dataset_preparing.__download__ import get_text_from_folder
import tensorflow as tf

def separation(func):
    def wrapper(*args, **kwargs):
        print('----------------------------------------')
        result = func(*args, **kwargs)
        print('----------------------------------------')
        return result
    return wrapper

@separation
def print_current_num(text: str) -> None:
    print(text)

def get_training_sequences_1(text: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    sequence = tokenizer.get_full_sequence(text)
    length = 50

    print('0%')
    input_values = [sequence[i:i+length] for i in range(len(sequence) - length) if tokenizer.get_text([sequence[i + length]]) != '\ufffd' and ' ']
    
    print('50%')
    valid_values = [sequence[i + length] for i in range(len(sequence) - length) if tokenizer.get_text([sequence[i + length]]) != '\ufffd' and ' ']
    
    print('90%')
    train_data = np.array(input_values)
    train_labels = np.array(valid_values)

    middle = len(train_labels) // 2



    return train_data[:middle], train_labels[:middle], train_data[middle:], train_labels[middle:]

def get_training_sequences(text: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    sequence = tokenizer.get_full_sequence(text)

    input_values = []
    valid_values = []
    min_length = 2
    max_length = 50

    for i in range(len(sequence) - max_length):

        length = random.randint(min_length, max_length)

        if tokenizer.get_text([sequence[i + length]]) != '\ufffd':

            input_values.append(tokenizer.get_sequence(tokenizer.get_text(sequence[i:i + length])))
            valid_values.append(sequence[i + length])

        print(f'Do: {round(100 * (i / (len(sequence) - max_length)), 2)}%')

    train_data = np.array(input_values, dtype='float16')
    train_labels = np.array(valid_values, dtype='float16')

    middle = len(train_labels) // 2

    return train_data[:middle], train_labels[:middle], train_data[middle:], train_labels[middle:]

def train(cyra_model) -> None:
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002', '*.txt'))
    
    print(f'Files in your dataset: {len(txt_files)}')

    for txt_file in txt_files:

        print_current_num(f'Reading: {txt_files.index(txt_file)}/{len(txt_files)}')

        with open(txt_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        train_data, train_labels, test_data, test_labels = get_training_sequences(text, cyra_model.tokenizer)
        checkpoint_path = "trained-models/cyra_check_point.ckpt"

        # for token in train_labels:
        #     print(cyra_model.tokenizer.get_text([token]))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        save_freq='epoch',
                                                        period=50)

        # if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
        #     print(f'Model weights are loded form: {checkpoint_path}')
        #     cyra_model.model.load_weights(checkpoint_path)

        cyra_model.model.fit(train_data, 
                            train_labels, 
                            batch_size=1, 
                            epochs=1,
                            validation_data=(test_data, test_labels), 
                            callbacks=[cp_callback])

        # if (txt_files.index(txt_file) % 5 == 0):

        #     cyra_model.model.save('D:/Exider Company/Cyra/trained-models/cyra.h5')

@separation
def print_dataset(train_data, train_labels) -> None:
    
        print(cyra_model.tokenizer.get_text(train_data), end='-->')
        print(cyra_model.tokenizer.get_text(train_labels))

def train_with_huge_batch(cyra_model) -> None:

    print('Loading text')
    with open('C:/Users/Atynate/Downloads/ru-2.txt', 'r', encoding='utf-8') as dataset:
        text = dataset.read(10 ** 7)

    print('Creating test and training samples')
    train_data, train_labels, test_data, test_labels = get_training_sequences_1(text, cyra_model.tokenizer)

    # for i in range(len(train_data)):
    #     print_dataset(train_data[i], [train_labels[i]])

    # if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
    #     print(f'Model weights are loded form: {checkpoint_path}')
    #     cyra_model.model.load_weights(checkpoint_path)

    # print(train_data)
    print(train_data.shape, train_labels.shape)

    print('Start training')
    cyra_model.model.fit(train_data, 
                        train_labels, 
                        batch_size=32,
                        epochs=10,
                        validation_data=(test_data, test_labels))
    
    print('Saving model')
    cyra_model.model.save_weights('D:/Exider Company/Cyra/trained-models/cyra.h5')

if __name__ == '__main__':
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 8, 512, 12, 2048, path='D:/Exider Company/Cyra/trained-models/cyra.h5')
    train_with_huge_batch(cyra_model)