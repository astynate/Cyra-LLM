import os
import glob
from typing import Tuple
import tensorflow as tf
import numpy as np
from cyra_model.model import Cyra
from cyra_model.tokenizer import CyraTokenizer
from tensorflow.keras.callbacks import LearningRateScheduler

def scheduler(epoch, lr):
    return lr * np.exp(-0.2)

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

        length = 50

        if tokenizer.get_text([sequence[separator + length]]) != '\ufffd':

            input_values.append(tokenizer.get_sequence(tokenizer.get_text(sequence[separator:separator + length])))
            valid_values.append(sequence[separator + length])

        separator += 1

    train_data = np.array(input_values)
    train_labels = tf.keras.utils.to_categorical(np.array(valid_values), num_classes=tokenizer.get_dimension())

    return [train_data, train_labels]

def train(model) -> None:
    
    txt_files = glob.glob(os.path.join('D:/Exider Company/Cyra/dataset_preparing/output_dataset/dataset-002', '*.txt'))
    
    # print(f'Files in your dataset: {len(txt_files)}')

    for txt_file in txt_files[:1]:

        print_current_num(f'Reading: {txt_files.index(txt_file)}/{len(txt_files)}')

        with open(txt_file, 'r', encoding='utf-8') as infile:
            text = infile.read()

        # text = 'сотрудники венского медицинского университета разработали технологию которая позволяет пациентам травмой плечевого нервного сплетения управлять бионическим протезом работа опубликована журнале lancet плечевое сплетение дает начало нервам которые управляют движением мышц руки травмы этого сплетения приводят функциональной ампутации конечности причем восстановить иннервацию обычно удается вместо попыток наладить иннервацию австрийские медики решили использовать бионический протез управлять должна остаточная активность нервов плеча работа нервов ранее пересаженных пациентам вместе мышцей другой части тела ноги обнаружив слабую активность плечевых нервов ученые снабдили электрическими сенсорами начали тренировать пациентов управлять виртуальной рукой которую показывали экране компьютера после девяти месяцев тренировок электрическая активность нервных окончаний значительно возросла нервам подключили бионический протез первоначально носился одновременно нефункциональной рукой только пациенты научились достаточно ловко ним управляться медики ампутировали бесполезную конечность'

        # callback = LearningRateScheduler(scheduler)
        train_data, train_labels = get_training_squences(text, model.tokenizer)

        # for i in range(len(train_labels)):
        #     print_current_num(f'{model.tokenizer.get_text(train_data[i])} - {model.tokenizer.get_text([[np.argmax(train_labels[i])]][0])}')

        print(train_data)
        prediction = model.model.predict(train_data[0].reshape((1, 50)))
        # prediction = prediction.flatten()

        # print(prediction.shape)
        # print(prediction)

        # for i in range(prediction.shape[0]):

        #     if (round(prediction[i], 4) > 0):

        #         print(round(prediction[i], 4))

        # checkpoint_path = "trained-models/cyra_check_point.ckpt"

        # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
        #                                                 save_weights_only=True,
        #                                                 verbose=1,
        #                                                 save_freq='epoch',
        #                                                 period=100)

        # if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
        #     print(f'Model weights are loded form: {checkpoint_path}')
        #     model.model.load_weights(checkpoint_path)

        # model.model.fit(train_data, train_labels, batch_size=1, epochs=12)

        # for i in range(train_data.shape[0]):
        #     context = model.tokenizer.get_text(train_data[i])
        #     generated_word = model(context)
            
        #     print(f'Context: {context}')
        #     print(f'Prediction: {generated_word}.')

        #     print(f'Data: {model.tokenizer.get_text(train_data[i])}')
        #     print(f'Target: {model.tokenizer.get_text([[np.argmax(train_labels[i])]][0])}.')

    # model.model.save('trained-models/cyra.keras')

if __name__ == '__main__':

    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/cyra_tokenizer.pickle'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path, 50)

    cyra_model = Cyra(cyra_tokenizer, 6, 128, 6, 1024)
    train(cyra_model)