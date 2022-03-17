import os, glob
import numpy as np
from Basic_models.__NLP_Dataloader import DEF_Prepare_Dataset
from Basic_models.__NLP_Models import LSTM_model, GRU_model, BiLSTM_model, onedCNN_model


def DEF_Load_Model(model_name, vocab_size, X_train, X_test, Y_train, Y_test, korean_package) :
    if model_name == 'LSTM' :
        LSTM_model(vocab_size, X_train, X_test, Y_train, Y_test, korean_package)
    elif model_name == 'GRU' :
        GRU_model(vocab_size, X_train, X_test, Y_train, Y_test, korean_package)
    elif model_name == 'BiLSTM' :
        BiLSTM_model(vocab_size, X_train, X_test, Y_train, Y_test, korean_package)
    elif model_name == 'onedCNN' :
        onedCNN_model(vocab_size, X_train, X_test, Y_train, Y_test, korean_package)

def DEF_Check_Before_Train(korean_package) :

    if (f'{korean_package}_tokenizer.*' not in os.listdir('./pickled_ones')) and (f'{korean_package}_badlang_dataset_*.npy' not in os.listdir('./saved_np')):
        X_train, X_test, Y_train, Y_test, Max_len, Vocab_size = DEF_Prepare_Dataset(korean_package)
        return X_train, X_test, Y_train, Y_test, Max_len, Vocab_size

    else :
        target = f'./saved_np/{korean_package}_badlang_dataset_*.npy'
        find_vocab_size = glob.glob(target)[0][:-4].split('_')
        Max_len, Vocab_size = int(find_vocab_size[-2]), int(find_vocab_size[-1])

        X_train, X_test, Y_train, Y_test = np.load(
            f'./saved_np/{korean_package}_badlang_dataset_{Max_len}_{Vocab_size}.npy', allow_pickle=True)

        return X_train, X_test, Y_train, Y_test, Max_len, Vocab_size


if __name__ == '__main__':

    korean_package_list = ['okt_morphs', 'okt_pos', 'jamo', 'char', 'spm', 'mecab', 'soynlp']

    korean_package = korean_package_list[0]

    model_name_list = ['LSTM','GRU','BiLSTM','onedCNN']
    model_name = model_name_list[0]

    X_train, X_test, Y_train, Y_test, Max_len, Vocab_size = DEF_Check_Before_Train(korean_package)
    DEF_Load_Model(model_name, Vocab_size, X_train, X_test, Y_train, Y_test, korean_package)

    # model_name_list = ['LSTM', 'GRU', 'BiLSTM', 'onedCNN']
    # korean_package_list = ['okt_morphs', 'okt_pos', 'jamo', 'char', 'spm', 'mecab', 'soynlp']
    #
    # for i in model_name_list :
    #     for j in korean_package_list[:4] :
    #         X_train, X_test, Y_train, Y_test, Max_len, Vocab_size = DEF_Check_Before_Train(j)
    #         DEF_Load_Model(i, Vocab_size, X_train, X_test, Y_train, Y_test, j)

