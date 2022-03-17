# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

import re
import numpy as np
import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import PreTrainedTokenizerFast
# from personal_information.class_check_personal_info import person_info_check_procedure


ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

badword_model = torch.load('./output/model/CSW_kCbert_3multi_0.93.pt', map_location=device)


badword_model.to(device)
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']

tokenizer.add_tokens(add_token)


chat_model = torch.load('./output/model/CSW_KoGPT_chatbot_14.89.pt')
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')


def transform(data):
    data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


with torch.no_grad():  ###torch.no_grad()는 오차 역전파에 사용하는 계산량을 줄여서 처리 속도를 높인다. with문이 끝나면 저절로 객체가 닫힌다.
    while 1:
        q = input("user > ").strip()  ###앞뒤 공백문자제거
        if q == "quit":
            break
        ##########################
        # checked_q = person_info_check_procedure().check(q)
        # if checked_q:
        #     print('개인정보가 포함된 내용입니다.')
        ##########################

        a = ""
        while 1:
            check = transform(q)
            input_ids = torch.tensor([check[0]]).to(device)
            token_type_ids = torch.tensor([check[1]]).to(device)
            attention_mask = torch.tensor([check[2]]).to(device)
            result = badword_model(input_ids, token_type_ids, attention_mask)
            idx = result.argmax().cpu().item()
            if idx == 1 :
                a = f"섹슈얼_{result}"
                break
            elif idx == 2 :
                a = f"혐오_{result}"
                break

            input_ids = torch.LongTensor(
                koGPT2_TOKENIZER.encode("<usr>" + q + '<unused1>' + "<sys>" + a)).unsqueeze(dim=0)
            input_ids = input_ids.to(ctx)
            pred = chat_model(input_ids)
            pred = pred.logits
            pred = pred.cpu()
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == '</s>':
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))

