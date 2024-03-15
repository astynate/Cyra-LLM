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

        length = np.random.randint(2, 50)

        if tokenizer.get_text([sequence[separator + length]]) != '\ufffd':

            input_values.append(tokenizer.get_sequence(tokenizer.get_text(sequence[separator:separator + length])))
            valid_values.append(sequence[separator + length])

        separator += length + 1

    train_data = np.array(input_values)
    train_labels = tf.keras.utils.to_categorical(np.array(valid_values), num_classes=tokenizer.get_dimension())

    return [train_data, train_labels]

def train(model) -> None:
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002', '*.txt'))
    
    print(f'Files in your dataset: {len(txt_files)}')

    for txt_file in txt_files[:1]:

        print_current_num(f'Reading: {txt_files.index(txt_file)}/{len(txt_files)}')

        with open(txt_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        train_data, train_labels = get_training_squences(text, model.tokenizer)

        # Print training dataset

        # for i in range(len(train_labels)):
        #     print_current_num(f'{model.tokenizer.get_text(train_data[i])}\n{model.tokenizer.get_text([[np.argmax(train_labels[i])]][0])}')

        # prediction = model.model.predict(train_data[0].reshape((1, 50)))
        # prediction = prediction.flatten()
    
        # for i in range(prediction.shape[0]):

        #     if (round(prediction[i], 4) > 0):

        #         print(round(prediction[i], 4))

        checkpoint_path = "trained-models/cyra_check_point.ckpt"

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                        save_weights_only=True,
                                                        verbose=1,
                                                        save_freq='epoch',
                                                        period=100)

        if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
            print(f'Model weights are loded form: {checkpoint_path}')
            model.model.load_weights(checkpoint_path)

        model.model.fit(train_data, train_labels, batch_size=1, epochs=300, callbacks=[cp_callback])
        context = model.tokenizer.get_text([np.random.randint(1, 30)])

        for i in range(5):

            generated_word = model(context)
            context += generated_word

            print(f'{i}: |{context}|')

    # model.model.save('trained-models/cyra.keras')

if __name__ == '__main__':

    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 16, 312, 16, 2048)
    train(cyra_model)