# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# pip install sentencepiece

import torch, os, gc, GPUtil
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

from Bert_Gpt_Chatbot.dataloader.dataloader import kcBERTDataset
from Bert_Gpt_Chatbot.model.kCbert_model import kcBERTClassifier



ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)
root_path= '../input/datasets'

dataset_train = []
dataset_test = []

df = pd.read_csv(f'{root_path}/final_train_test_datasets_normalize_0225.csv',sep='|')


train_df = pd.DataFrame()
test_df = pd.DataFrame()
categories = ['혐오', '섹슈얼', '일반_긍정']
for i in categories:
    filter_df = df[df['카테고리'] == i]
    train_set, test_set = train_test_split(filter_df, test_size=0.2)  # test_size = 비율설정
    train_df = pd.concat([train_df, train_set], ignore_index=True)
    test_df = pd.concat([test_df, test_set], ignore_index=True)


dataset_train = []
dataset_test = []

for i in range(len(train_df)) :
    if train_df.iloc[i,1] == '일반_긍정' :
        train_df.iloc[i,1] = '0'
    elif train_df.iloc[i,1] == '섹슈얼' :
        train_df.iloc[i,1] = '1'
    else : train_df.iloc[i,1] = '2'

    dataset_train.append([train_df.iloc[i,2], train_df.iloc[i,1]])

for i in range(len(test_df)) :
    if test_df.iloc[i,1] == '일반_긍정' :
        test_df.iloc[i,1] = '0'
    elif test_df.iloc[i,1] == '섹슈얼' :
        test_df.iloc[i,1] = '1'
    else : test_df.iloc[i,1] = '2'

    dataset_test.append([test_df.iloc[i,2], test_df.iloc[i,1]])


print(len(dataset_train))
print(len(dataset_test))


max_len = 64
batch_size = 1
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
learning_rate = 5e-5
add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']


train_dataset = kcBERTDataset(dataset_train, add_token=add_token)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

del dataset_train

test_dataset = kcBERTDataset(dataset_test, add_token=add_token)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

del dataset_test

model = kcBERTClassifier(vocab_size=train_dataset.vocab_size, add_token=train_dataset.added_token_num)
model.to(device)


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()


t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


####################################################################################
checkpoint_path = '../output/model'
save_ckpt_path = f'{checkpoint_path}/cp_kCbert.pt'

pre_epoch, pre_loss, train_step = 0, 0, 0
if os.path.isfile(save_ckpt_path):
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    pre_epoch = checkpoint['Epoch']
    model.load_state_dict(checkpoint['State_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"load pretrain from: {save_ckpt_path}, epoch={pre_epoch}")  # , loss={pre_loss}\n")
####################################################################################


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc


best_model = 0

for epoch in range(num_epochs):

    train_acc = 0.0
    test_acc = 0.0

    model.train()
    for batch_id, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(train_dataloader)):
        input_ids = input_ids.long().to(device)
        token_type_ids = token_type_ids.long().to(device)
        attention_mask = attention_mask.long().to(device)
        label = label.long().to(device)

        optimizer.zero_grad()
        out = model(input_ids, token_type_ids, attention_mask)

        del input_ids
        del token_type_ids
        del attention_mask


        loss = loss_fn(out, label)
        del label
        del out
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

    print("epoch {} train acc {}".format(epoch+1, train_acc / (batch_id+1)))

    model.eval()
    with torch.no_grad() :
        for batch_id, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(test_dataloader)):
            input_ids = input_ids.long().to(device)
            token_type_ids = token_type_ids.long().to(device)
            attention_mask = attention_mask.long().to(device)
            label = label.long().to(device)

            optimizer.zero_grad()
            out = model(input_ids, token_type_ids, attention_mask)

            test_acc += calc_accuracy(out, label)

    print("epoch {} test acc {}".format(epoch + 1, test_acc / (batch_id + 1)))


    if best_model < test_acc:
        state = {'Epoch': epoch,
                 'State_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        torch.save(state, f'{checkpoint_path}/cp_kCbert.pt')
        torch.save(model, f'{checkpoint_path}/kCbert_3multi_{test_acc/(batch_id + 1)}.pt')

        best_model = test_acc
