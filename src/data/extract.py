import os
import shutil
import py7zr


def extract_landmarks(data_path='data\\raw\\dataset'):
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


def archive_landmarks(data_path='data\\raw\\dataset'):
    '''
    Archives all extracted files in a directory and its subdirectories
    :param data_path: path to the data directory
    '''

    for root, dirs, _ in os.walk(data_path):
        for dir in dirs:
            print(f'Archiving folders in {dir}...')
            files = os.listdir(root + '\\' + dir)
            for file in files:
                if file.endswith((".wav", ".7z")):
                    continue
                filename = root + '\\' + dir + '\\' + file
                with py7zr.SevenZipFile(filename + '.7z', mode='w') as archive:
                    archive.writeall(filename, '')
                shutil.rmtree(filename)

if __name__ == '__main__':
    choice = input("Enter '1' to extract landmarks, '2' to archive landmarks: ")
    if choice == '1':
        extract_landmarks()
        print("Extraction complete")
    elif choice == '2':
        archive_landmarks()
        print("Archiving complete")
    else:
        print("Invalid choice")