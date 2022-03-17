import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

from Bert_Gpt_Chatbot.dataloader.dataloader import BERTDataset

test_dataset = []
file = open('../input/bad_lang_3.txt', 'r', encoding='utf-8')
while True:
    line = file.readline()
    if not line:
        break
    test_dataset.append([line[:-3], line[-2]])
file.close()
test_dataset = test_dataset[5000:]
print(len(test_dataset))

add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']


ctx = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(ctx)
model = torch.load('../output/model/badword.pt', map_location=device)
model.to(device)

test_dataset = BERTDataset(test_dataset, add_token=add_token)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=0)

labels = []
preds = []
for idx, (input_ids, token_type_ids, attention_mask, label) in enumerate(tqdm(test_dataloader)):
    input_ids = input_ids.long().to(device)
    token_type_ids = token_type_ids.long().to(device)
    attention_mask = attention_mask.long().to(device)

    result = model(input_ids, token_type_ids, attention_mask)
    pred = result.argmax().cpu().item()
    labels.append(label.item())
    preds.append(pred)
f1_score = f1_score(labels, preds)
print(f1_score)

