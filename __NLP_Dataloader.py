import os, pickle
import numpy as np
import pandas as pd




from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from __Korean_Package import *

###processed_X는 문장들의 집합
def DEF_Stopwords(processed_X, predict=False) :
    stopwords_path = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
                      '도', '를', '으로', '자', '에', '와', '한', '하다']
    if predict != True:  ### predict가 False이면 진행 할겠다
        for i in range(len(processed_X)) :  ## prcossed_X의 리스트 갯수만 만큼 포문에 들어가겠다.  i는 각문장
            words = []
            for j in range(len(processed_X[i])):  ##j가 의미하는 바는 predict의 리스트를 하나하나씩 받아 오겠다 j는 한문장내에서의 성분
                if processed_X[i][j] not in stopwords_path:
                    words.append(processed_X[i][j])
            processed_X[i] = ' '.join(words)
            return processed_X
    else : ##predict = true
        new_text = []
        for i in processed_X: ##processed_X 문장하나씩 가자옴
            if i not in stopwords_path:
                new_text.append(i)
        return new_text


def DEF_Onehot(Y, korean_package) :  ###
    encoder = LabelEncoder()
    labeled_Y = encoder.fit_transform(Y)  ###
    label = encoder.classes_
    with open(f'./pickled_ones/{korean_package}_encoder.pickle', 'wb') as f:
        pickle.dump(encoder, f)
    onehot_Y = to_categorical(labeled_Y)
    np.save(f'./saved_np/{korean_package}_onehot_data.npy', onehot_Y)
    return onehot_Y


def DEF_Check_Words_Info(processed_X_with_stopped, threshold=3) :
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_X_with_stopped)
    threshold = threshold
    rare_cnt, total_freq, rare_freq = 0, 0, 0
    total_cnt = len(tokenizer.word_index)
    for key, value in tokenizer.word_index.items():
        total_freq = total_freq + value
        if (value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value
    Vocab_size = total_cnt - rare_cnt + 1
    return Vocab_size + 2  # 패딩 토큰과 OVV 토큰을 고려하여 + 2 하자.


def DEF_Check_Padding_Length(X_train, max_len=10, include=99) :  ###X_train은 훈련 시키규 하는 데이터
    while True:
        count = 0
        for sentence in X_train :
            if(len(sentence) <= max_len) :  ##문장이 max_len 길이보다 작으면 실행
                count += 1    #최대길이 보다 데이터 수를 더함
        percent = (count / len(X_train))*100  ###최대 길이보다 작은 수의 비율을  제시
        if percent > include : ## 확률이 99이상일경우   프린트함
            # print('percent : ', percent, 'max_len : ', max_len, )
            return percent, max_len
            break
        else :
            max_len += 1   ### 그렇치 못할 경우 max_len 늘리겠다


def DEF_Tokenizing(processed_X_with_stopped, Vocab_size, korean_package) :
    if f'{korean_package}_tokenizer.pickle' not in os.listdir('./pickled_ones'):
        tokenizer = Tokenizer(Vocab_size, oov_token='OOV')
        tokenizer.fit_on_texts(processed_X_with_stopped)
        with open(f'./pickled_ones/{korean_package}_tokenizer.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
    else :
        with open(f'./pickled_ones/{korean_package}_tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
    tokened_X = tokenizer.texts_to_sequences(processed_X_with_stopped)
    return tokened_X

def DEF_Padding_Sequences(tokened_X, max_len) :
    X_pad = pad_sequences(tokened_X, max_len)
    return X_pad

def DEF_Sample_Split(X_pad, Onehot_Y, max_len, vocab_size, test_size, korean_package):
    X_train, X_test, Y_train, Y_test = train_test_split(X_pad, Onehot_Y,
                                                        test_size=test_size, stratify=Onehot_Y)
    if korean_package :
        XY = X_train, X_test, Y_train, Y_test
        np.save(f'./saved_np/{korean_package}_badlang_dataset_{max_len}_{vocab_size}.npy', XY)
    return X_train, X_test, Y_train, Y_test

# def DEF_Setting_Data_to_DF(file_path='./data/label_text_datasets_Feb17_2.csv') :
def DEF_Setting_Data_to_DF(file_path):

    df = pd.read_csv(file_path, sep='|')
    df.columns = ['비속어','카테고리','text']

    df.info()
    # df['text'] = df['text'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")
    # print(df['text'].isnull().sum())
    # df['카테고리'].value_counts().plot(kind = 'bar')
    # plt.show()
    X, Y = df['text'], df['카테고리']
    # X, Y = df['text'], df['label']
    return X, Y


def DEF_Prepare_Dataset(korean_package, eval = True) :
    korean_package = korean_package

    file_path = './datasets/final_train_test_datasets_normalize_0225.csv'

    X, Y = DEF_Setting_Data_to_DF(file_path)
    Onehot_Y = DEF_Onehot(Y, korean_package)

    if korean_package != 'spm':
        if korean_package == 'okt_morphs' :
            processed_X = DEF_Okt_Morphs(X)
            processed_X_with_stopped = DEF_Stopwords(processed_X)   ## 일단 이것만 스탑워드 함수 사용.
        elif korean_package == 'okt_pos' :
            processed_X = DEF_Okt_Pos(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'jamo' :
            processed_X = DEF_Jamotools(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'char' :
            processed_X = DEF_Char(X)
            processed_X_with_stopped = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        elif korean_package == 'mecab' :
            processed_X = DEF_Mecab(X)
            processed_X_with_stopped = processed_X
        elif korean_package == 'soynlp' :
            processed_X = DEF_Soynlp(X)
            processed_X_with_stopped = processed_X

        Vocab_size = DEF_Check_Words_Info(processed_X_with_stopped, 3)
        tokened_X = DEF_Tokenizing(processed_X_with_stopped, Vocab_size, korean_package)
        percent, Max_len = DEF_Check_Padding_Length(tokened_X)

    elif korean_package == 'spm':
        processed_X = DEF_Sentencepiece(X)
        tokened_X = processed_X      ## 스탑워드하지 않았지만 변수명 맞추기용 코드
        Vocab_size = 10000 + 1
        percent, Max_len = DEF_Check_Padding_Length(tokened_X)



    X_pad = DEF_Padding_Sequences(tokened_X, Max_len)
    X_train, X_test, Y_train, Y_test = DEF_Sample_Split(
        X_pad, Onehot_Y, Max_len, Vocab_size, 0.2, korean_package)

    return X_train, X_test, Y_train, Y_test, Max_len, Vocab_size

if __name__ == '__main__':
    # print(DEF_Prepare_Dataset())



    pass

