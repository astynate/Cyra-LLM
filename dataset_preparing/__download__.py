import os

"""
This method is designed to read all text
files in the specified folder and concatenate their contents into one line.

Options:
folder_path (str): Path to the folder from which
need to read text files.

Returns:
combined_text (str): String containing combined
the text of all text files in the specified folder.
"""
def get_text_from_folder(folder_path) -> None:
    combined_text = ""
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                combined_text += file.read() + " "
                file.close()
    
    return combined_text

if __name__ == '__main__':

    dataset = get_text_from_folder('dataset_preparing/input_dataset/dataset-001')
    print(dataset)