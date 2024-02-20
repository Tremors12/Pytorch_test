"""
    Select chosen number of each class examples. 
    Do MFCC on this files.
"""

import os
import numpy as np
import librosa

class_name = [
    "airport",
    "shopping_mall",
    "metro_station",
    "street_pedestrian",
    "public_square",
    "street_traffic",
    "tram",
    "bus",
    "metro",
    "park"
]

def find_wav_files(directory):
    wav_files_path = []
    wav_files_name = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a .wav extension
            if file.lower().endswith(".wav"):
                # Construct the full path to the WAV file
                wav_path = os.path.join(root, file)
                # Append a tuple containing both the file name and its full path
                wav_files_path.append(wav_path)
                wav_files_name.append(file)

    return wav_files_path, wav_files_name

def parse_filename(filename):
    # Remove the ".wav" extension and split the filename by hyphen
    parts = filename[:-4].split('-')
    return parts

def main():
    
    class_counter = np.zeros(len(class_name), dtype=np.int64)

    directory_path = "/media/urh/UBUNTU_EXTE/DCASE"

    # Check if the directory exists
    if not os.path.exists(directory_path):
        print("Directory does not exist.")
        exit()

    # Find WAV files in the specified directory
    wav_files_path, wav_files_name = find_wav_files(directory_path)

    wav_files_path_class = []
    wav_files_name_class = []

    for j in range(len(class_name)):
        wav_files_path_class.append([])
        wav_files_name_class.append([])

    for i in range(len(wav_files_name)):
        parse_list = parse_filename(wav_files_name[i])
        for j in range(len(class_name)):
            if(class_name[j] == parse_list[0]):
                class_counter[j] += 1
                wav_files_path_class[j].append(wav_files_path[i])
                wav_files_name_class[j].append(wav_files_name[i])
                break
        #### print result 
    for j in range(len(class_name)):
        print(f"found {class_counter[j]} x class : {class_name[j]}")

    print(f"\ntotal number of samples : {len(wav_files_name)}")

    ################################################################################
    
    LEN_DATA_LEARN = 100
    LEN_DATA_EVAL = 30

    ################################################################################

    # create array for data
    data_path_learn = []
    data_path_eval = []

    for j in range(len(class_name)):
        data_path_learn.append([])
        data_path_eval.append([])

    # save data path
    for j in range(len(class_name)):
        data_path_learn[j] = wav_files_path_class[j][0:LEN_DATA_LEARN]
        data_path_eval[j] = wav_files_path_class[j][LEN_DATA_LEARN:LEN_DATA_LEARN+LEN_DATA_EVAL]

    print(f"\nWe will use {LEN_DATA_LEARN}/class and {LEN_DATA_EVAL}/class number of samples")
    ################################################################################

    # MFCC
    data_learn = []
    data_eval = []

    # add learn data
    for j in range(len(class_name)):
        for i in range(LEN_DATA_LEARN):
            y, sr = librosa.load(data_path_learn[j][i])       
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            data_learn.append([j, mfcc])
    
    print(f"\ncalculating of learn mfcc done, total number of samples : {len(data_learn)}")

    # add eval data
    for j in range(len(class_name)):
        for i in range(LEN_DATA_EVAL):       
            y, sr = librosa.load(data_path_eval[j][i])       
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            data_eval.append([j, mfcc]) 

    print(f"\ncalculating of eval mfcc done, total number of samples : {len(data_eval)}")


if __name__ == "__main__":
    main()