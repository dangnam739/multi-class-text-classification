import os
import numpy as np
import pandas as pd
import re
import csv
import torch
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm

CLASS_NAME_TO_ID = {'Am nhac': 0,
                    'Am thuc': 1,
                    'Bat dong san': 2,
                    'Bong da': 3,
                    'Chung khoan': 4,
                    'Cum ga': 5,
                    'Cuoc song do day': 6,
                    'Du hoc': 7,
                    'Du lich': 8,
                    'Duong vao WTO': 9,
                    'Gia dinh': 10,
                    'Giai tri tin hoc': 11,
                    'Giao duc': 12,
                    'Gioi tinh': 13,
                    'Hackers va Virus': 14,
                    'Hinh su': 15,
                    'Khong gian song': 16,
                    'Kinh doanh quoc te': 17,
                    'Lam dep': 18,
                    'Loi song': 19,
                    'Mua sam': 20,
                    'My thuat': 21,
                    'San khau dien anh': 22,
                    'San pham tin hoc moi': 23,
                    'Tennis': 24,
                    'The gioi tre': 25,
                    'Thoi trang': 26}


def load_data_to_csv(path, train=True):
    """
    load data from original directory to csv file and retrun dataframe

    Parameters:
    -----------
        path: path to original data directory
        train=True: load data from train directory and False is load from test data

    Return:
    -------
        df: dataframe of data.
    """

    if train:
        print("---Start load Training data---")
        csv_path = os.path.join(path, 'train.csv')
    else:
        print("---Start load Testing data---")
        csv_path = os.path.join(path, 'test.csv')

    data = {}
    data["text"] = []
    data["class_id"] = []
    data["length"] = []

    for dirname in CLASS_NAME_TO_ID.keys():
        # print(dirname)
        class_id = CLASS_NAME_TO_ID[dirname]
        category_path = os.path.join(path, dirname)

        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-16') as txt_file:
                    content = txt_file.read()
                    length = len(content)

                    data["text"].append(content)
                    data["class_id"].append(class_id)
                    data["length"].append(length)
            except:
                print('Can not open file: ', filepath)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, sep='\t', encoding='utf-8')
    print('---End load data---')

    return df


# remove special characters
strip_special_chars = re.compile("[^\w0-9 ]+")


def clean_sentences(sentence):
    sentence = sentence.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", sentence.lower())


def text2ids(sentence, max_sen_length, words_list):
    """
    Biến đổi các text trong dataframe thành ma trận index
    Parameters
    ----------
    sentence:
        text cần biến đổi
    max_length: int
        độ dài tối đa của một text
    _word_list: numpy.array
        array chứa các từ trong word vectors
    Returns
    -------
    numpy.array
        len(df) x max_length contains indices of text
    """
    # create word2idx directory
    word2idx = {w: i for i, w in enumerate(words_list)}

    # word segmentation
    sentence = ViTokenizer.tokenize(sentence)
    sentence = clean_sentences(sentence)

    # Split sentence to words
    words = [word.lower() for word in sentence.split()]

    ids = np.zeros((max_sen_length), dtype='int32')

    for idx, word in enumerate(words):
        if idx < max_sen_length:
            if (word in words_list):
                word_idx = word2idx[word]
            else:
                word_idx = word2idx['UNK']
            ids[idx] = word_idx
        else:
            break

    ids = torch.from_numpy(ids).type(torch.LongTensor)
    return ids

def preprocess_RF(sentence, stop_file):
    # clean sentence
    sentence = clean_sentences(sentence)
    # word segmentation
    sentence = ViTokenizer.tokenize(sentence)

    stop_words = []
    with open(stop_file, "r", encoding='utf-8') as f:
        text = f.read()

        for word in text.split():
            stop_words.append(word)
        f.close()
    
    sent = []
    for word in sentence.split(" ") :
        if (word not in stop_words):
            if ("_" in word) or (word.isalpha() == True):
                sent.append(word)
    pre_sentence = " ".join(sent)

    return pre_sentence
