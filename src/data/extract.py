import os
import py7zr

# function recursively extracts all 7z files from a directory
def extract_landmarks(data_path):
    '''
    Extracts all 7z files from a directory and its subdirectories
    saves the extracted files in the same directory
    keep in separate folders where zip files are located
    :param data_path: path to the data directory
    '''
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".7z"):
                with py7zr.SevenZipFile(os.path.join(root, file), mode='r') as archive:
                    archive.extractall(root + '\\' + file.split('.')[0])
                # remove the 7z file after extraction
                os.remove(os.path.join(root, file))


if __name__ == '__main__':
    extract_landmarks('..\\data\\raw\\landmarks_classified')

