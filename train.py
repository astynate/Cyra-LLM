from typing import Tuple
import numpy as np
from cyra_model.model import Cyra
from cyra_model.tokenizer import CyraTokenizer
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
    incorrect = [' ', '<unknown>']
    length = 50

    print('0%')
    input_values = [sequence[i:i+length] for i in range(len(sequence) - length)
                    if tokenizer.get_text([sequence[i + length]]) not in incorrect]
    
    print('50%')
    valid_values = [sequence[i + length] for i in range(len(sequence) - length)
                    if tokenizer.get_text([sequence[i + length]]) not in incorrect]
    
    print('90%')
    train_data = np.array(input_values)
    train_labels = np.array(valid_values)

    middle = len(train_labels) // 2
    return train_data[:middle], train_labels[:middle], train_data[middle:], train_labels[middle:]

@separation
def print_dataset(train_data, train_labels) -> None:
    print(cyra_model.tokenizer.get_text(train_data), end='-->')
    print(cyra_model.tokenizer.get_text(train_labels))

def train_with_huge_batch(cyra_model) -> None:

    print('Loading text')
    with open('C:/Users/Atynate/Downloads/ru-2.txt', 'r', encoding='utf-8') as dataset:
        text = dataset.read(8 ** 7)

    print('Creating test and training samples')
    train_data, train_labels, test_data, test_labels = get_training_sequences(text, cyra_model.tokenizer)

    space = cyra_model.tokenizer.dictionary.index(' ')

    print(f'Spaces: {100 * train_labels.tolist().count(space) / len(train_labels.tolist())}%')

    # for i in range(len(train_data)):
    #     print_dataset(train_data[i], [train_labels[i]])

    # if os.path.exists('trained-models/cyra_check_point.ckpt_temp'):
    #     print(f'Model weights are loded form: {checkpoint_path}')
    #     cyra_model.model.load_weights(checkpoint_path)

    for i in range(train_data.shape[0]):
        if (np.isnan(train_data[i]).any() or np.isnan(train_labels[i]).any() or np.isnan(test_data[i]).any() or np.isnan(test_labels[i]).any()):
            print("NAN!")

    # print(train_data)
    # print(train_data.shape, train_labels.shape)

    # unique, counts = np.unique(train_labels, return_counts=True)

    # # Находим индекс самого часто встречающегося элемента
    # index = np.argmax(counts)

    # # Самый часто встречающийся элемент
    # most_frequent = unique[index]

    # # Процент от общего числа элементов
    # percentage = counts[index] / len(train_labels.tolist()) * 100

    # print(f'Самый часто встречающийся элемент: {most_frequent}')
    # print(f'Процент от общего числа элементов: {percentage}%')

    # tf.keras.losses.sparse_categorical_crossentropy(np.array([1, 1, 1]), np.array([1, 1, 1]))

    # with open('1.txt', 'w', encoding='utf-8') as f:
    #     f.write(cyra_model.tokenizer.get_text(train_data.tolist()))

    # with open('2.txt', 'w', encoding='utf-8') as f:
    #     f.write(train_data.tolist())

    print('Start training')
    cyra_model.model.fit(train_data, 
                        train_labels, 
                        batch_size=64,
                        epochs=1)

    # print(cyra_model("Hello adadasd"))
    
    print('Saving model')
    cyra_model.model.save_weights('D:/Exider Company/Cyra/trained-models/cyra.h5')

if __name__ == '__main__':
    cyra_tokenizer_path = 'D:/Exider Company/Cyra/trained-models/tokenizer.txt'
    cyra_tokenizer = CyraTokenizer(cyra_tokenizer_path)

    cyra_model = Cyra(cyra_tokenizer, 12, 256, 12, 2048, path='D:/Exider Company/Cyra/trained-models/cyra.h5')
    train_with_huge_batch(cyra_model)