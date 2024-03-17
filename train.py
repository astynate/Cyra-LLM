import os
import glob
from typing import Tuple
import tensorflow as tf
import numpy as np
from cyra_model.model import Cyra
from cyra_model.tokenizer import CyraTokenizer

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

def get_training_squences(text: str, tokenizer) -> Tuple[np.ndarray, np.ndarray]:
    sequence = tokenizer.get_full_sequence(text)
        
    input_values = []
    valid_values = []

    separator = 0

    while len(sequence) > separator + 51:

        length = 51

        if tokenizer.get_text([sequence[separator + length]]) != '\ufffd':

            input_values.append(tokenizer.get_sequence(tokenizer.get_text(sequence[separator:separator + length])))
            valid_values.append(sequence[separator + length])

        separator += 1

    train_data = np.array(input_values)
    train_labels = np.array(valid_values)

    return [train_data, train_labels]

def train(cyra_model) -> None:
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002', '*.txt'))
    
    print(f'Files in your dataset: {len(txt_files)}')

    for txt_file in txt_files:

        print_current_num(f'Reading: {txt_files.index(txt_file)}/{len(txt_files)}')

        with open(txt_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        train_data, train_labels = get_training_squences(text, cyra_model.tokenizer)
        checkpoint_path = "trained-models/cyra_check_point.ckpt"

        # for token in train_labels:
        #     print(cyra_model.tokenizer.get_text([token]))

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                 save_weights_only=True,
        #                                                 verbose=1,
        #                                                 save_freq='epoch',
        #                                                 period=50)

        # if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
        #     print(f'Model weights are loded form: {checkpoint_path}')
        #     cyra_model.model.load_weights(checkpoint_path)

        cyra_model.model.fit(train_data, train_labels, batch_size=32, epochs=30)
        number_of_correct_predictions = 0

        for i in range(train_data.shape[0]):

            generated_word = cyra_model(cyra_model.tokenizer.get_text(train_data[i]))

            if generated_word == cyra_model.tokenizer.get_text([train_labels[i]]):

                number_of_correct_predictions += 1

        print(f'Final accuracy: {100 * (number_of_correct_predictions / train_data.shape[0])}%')

        if (txt_files.index(txt_file) % 3 == 0):

            cyra_model.model.save('trained-models/cyra.keras')

if __name__ == '__main__':

    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 1, 512, 12, 25)
    train(cyra_model)