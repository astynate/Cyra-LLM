import os
import glob
from typing import Tuple
import numpy as np
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

def get_training_sequences(text: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    sequence = tokenizer.get_full_sequence(text)
    length = 50

    print('0%')
    input_values = [sequence[i:i+length] for i in range(len(sequence) - length) if tokenizer.get_text([sequence[i + length]]) != '\ufffd']
    
    print('50%')
    valid_values = [sequence[i + length] for i in range(len(sequence) - length) if tokenizer.get_text([sequence[i + length]]) != '\ufffd']
    
    print('90%')
    train_data = np.array(input_values)
    train_labels = np.array(valid_values)

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

        for token in train_labels:
            print(cyra_model.tokenizer.get_text([token]))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        save_freq='epoch',
                                                        period=50)

        if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
            print(f'Model weights are loded form: {checkpoint_path}')
            cyra_model.model.load_weights(checkpoint_path)

        cyra_model.model.fit(train_data, train_labels, batch_size=32, epochs=10)
        number_of_correct_predictions = 0

        for i in range(test_data.shape[0]):

            generated_word = cyra_model(cyra_model.tokenizer.get_text(test_data[i]))

            if generated_word == cyra_model.tokenizer.get_text([test_labels[i]]):

                number_of_correct_predictions += 1

        print(f'Final accuracy: {100 * (number_of_correct_predictions / test_labels.shape[0])}%')

        if (txt_files.index(txt_file) % 5 == 0):

            cyra_model.model.save('D:/Exider Company/Cyra/trained-models/cyra.h5')

def train_with_huge_batch(cyra_model) -> None:

    print('Loading text')
    text = get_text_from_folder('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002')

    print('Creating test and training samples')
    train_data, train_labels, test_data, test_labels = get_training_sequences(text, cyra_model.tokenizer)

    checkpoint_path = "D:/Exider Company/Cyra/trained-models/cyra_check_point.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq='epoch',
                                                    period=1)

    if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
        print(f'Model weights are loded form: {checkpoint_path}')
        cyra_model.model.load_weights(checkpoint_path)

    print('Start training')
    cyra_model.model.fit(train_data, train_labels, batch_size=256, epochs=10, validation_data=(test_data, test_labels), callbacks=[cp_callback])
    
    print('Saving model')
    cyra_model.model.save('D:/Exider Company/Cyra/trained-models/cyra.h5')

if __name__ == '__main__':

    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 1, 512, 8, 25, path='trained-models/cyra.h5')
    train_with_huge_batch(cyra_model)