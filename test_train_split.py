import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/gkghk/PycharmProjects/bad_chatbot/Bert_Gpt_Chatbot/input/datasets/final_datasets_normalize_0225.csv', sep="|")
###C:/Users/gkghk/PycharmProjects/bad_chatbot/Bert_Gpt_Chatbot/input/datasets/final_datasets_normalize_0225.csv
train_df = pd.DataFrame()
test_df = pd.DataFrame()

categories = ['혐오', '섹슈얼', '일반_긍정']

for i in categories:
    filter_df = df[df['카테고리'] == i]
    train_set, test_set = train_test_split(filter_df, test_size=0.2)  # test_size = 비율설정
    train_df = pd.concat([train_df, train_set], ignore_index=True)
    test_df = pd.concat([test_df, test_set], ignore_index=True)

# 행 섞어주기
test_df = test_df.sample(frac=1).reset_index(drop=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)

test_df.to_csv("C:/Users/gkghk/PycharmProjects/bad_chatbot/Bert_Gpt_Chatbot/input/final_evaluation_datasets_normalize_0225_h.csv",sep='|', index=False)
train_df.to_csv("C:/Users/gkghk/PycharmProjects/bad_chatbot/Bert_Gpt_Chatbot/input//final_train_test_datasets_normalize_0225_h.csv", sep='|',index=False)
##C:/Users/gkghk/PycharmProjects/bad_chatbot/Bert_Gpt_Chatbot/input/datasets/final_datasets_normalize_0225.csv