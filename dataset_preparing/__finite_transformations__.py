import os
from __primary_transformations__ import *

def convert_dataset(folder_path: str, target_folder_path: str) -> None:

    print('Start of dataset conversion...')

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = preprocess_text(file.read())

            # if not os.path.exists(os.path.join(target_folder_path, filename)):
            #     os.makedirs(target_folder_path)

            with open(os.path.join(target_folder_path, filename), 'w', encoding='utf-8') as f:
                f.write(text)

            print(f'Write to file: {filename}')

    print('All data processed!')

if __name__ == '__main__':

    folder_path = 'dataset_preparing/input_dataset/dataset-001'
    target_folder_path = 'dataset_preparing/output_dataset/dataset-002'

    convert_dataset(folder_path, target_folder_path)