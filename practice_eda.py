import pandas as pd
df = pd.read_csv('C:/Users/gkghk/PycharmProjects/bad_chatbot/Basic_models/datasets/final_datasets_0225.csv', sep='|',index_col = False)
df = df['text']
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

print(DEF_Check_Padding_Length(df, max_len=10, include=99))