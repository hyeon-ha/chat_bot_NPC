import pandas as pd
from pykospacing import Spacing

spacing = Spacing()

df = pd.read_csv('bad_texts_i100_1400.csv')
df.info()

datas = []

for i in range(len(df)) :
    x = df.iloc[i,1]
    x = x.split('</s><s>')
    # x = x.split('\n')
    for j in x :
        j = j.replace('""','')
        j = j.replace('<s>','')
        j = j.replace('</s>', '')
        j = j.replace('\n', '')
        j = j.replace('|', '')
        # j = spacing(j)
        datas.append(j)


for i in range(len(df)) :
    x = df.iloc[i,2]
    x = x.split('</s><s>')
    for j in x :
        j = j.replace('<s>','')
        j = j.replace('</s>', '')
        j = j.replace('\n', '')
        j = j.replace('|', '')
        datas.append(j)
print(len(datas))

df_2 = pd.DataFrame(datas)
df_2.info()
print(df_2.head(10))
df_2.to_csv('bad_reply_4.csv', sep='|', index=False)
