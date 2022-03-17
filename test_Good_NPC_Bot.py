import telegram
# pip install python-telegram-bot
# https://python.bakyeono.net/chapter-12-2.html


import torch, re
import numpy as np
from transformers import PreTrainedTokenizerFast
from kobert_tokenizer import KoBERTTokenizer

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

badword_model = torch.load('./output/model/Kobert_3multi_0.85.pt', map_location=device)
badword_model.to(device)

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
tokenizer.add_tokens(add_token)


def transform(data):
    data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


chat_model = torch.load('./output/model/KoGPT_chatbot_22.0.pt', map_location=device)
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')

import json
import time  # 추가함
import urllib.parse
import urllib.request

TOKEN = '5215951650:AAFVD-sUEFB-vgb5yvryBOgqnyoiG66QJZU'  # 여러분의 토큰으로 변경


####5215951650:AAFVD-sUEFB-vgb5yvryBOgqnyoiG66QJZU
def request(url):
    """지정한 url의 웹 문서를 요청하여, 본문을 반환한다."""
    response = urllib.request.urlopen(url)
    byte_data = response.read()
    text_data = byte_data.decode()
    return text_data


def build_url(method, query):
    """텔레그램 챗봇 웹 API에 요청을 보내기 위한 URL을 만들어 반환한다."""
    return f'https://api.telegram.org/bot{TOKEN}/{method}?{query}'


def request_to_chatbot_api(method, query):
    """텔레그램 챗봇 웹 API에 요청하고 응답 결과를 사전 객체로 해석해 반환한다."""
    url = build_url(method, query)
    response = request(url)
    return json.loads(response)


def simplify_messages(response):
    """텔레그램 챗봇 API의 getUpdate 메서드 요청 결과에서 필요한 정보만 남긴다."""
    result = response['result']
    if not result:
        return None, []
    last_update_id = max(item['update_id'] for item in result)

    try:
        messages = [item['message'] for item in result]
        simplified_messages = [{'from_id': message['from']['id'],
                                'text': message['text']}
                               for message in messages]
    except:
        for message in messages:
            if 'text' not in list(message.keys()):
                message['text'] = '무시'
                simplified_messages = [{'from_id': message['from']['id'],
                                        'text': message['text']}]
    print(simplified_messages)
    return last_update_id, simplified_messages


def get_updates(update_id):
    """챗봇 API로 update_id 이후에 수신한 메시지를 조회하여 반환한다."""
    query = f'offset={update_id}'
    response = request_to_chatbot_api(method='getUpdates', query=query)
    return simplify_messages(response)


def send_message(chat_id, text):
    """챗봇 API로 메시지를 chat_id 사용자에게 text 메시지를 발신한다."""
    text = urllib.parse.quote(text)
    query = f'chat_id={chat_id}&text={text}'
    response = request_to_chatbot_api(method='sendMessage', query=query)
    return response


def check_messages_and_response(next_update_id):
    """챗봇으로 메시지를 확인하고, 적절히 응답한다."""
    last_update_id, recieved_messages = get_updates(next_update_id)  # ❶
    for message in recieved_messages:  # ❷
        chat_id = message['from_id']
        text = message['text']

        q = text
        a = ''
        while 1:

            check = transform(q)
            input_ids = torch.tensor([check[0]]).to(device)
            token_type_ids = torch.tensor([check[1]]).to(device)
            attention_mask = torch.tensor([check[2]]).to(device)
            result = badword_model(input_ids, token_type_ids, attention_mask)
            idx = result.argmax().cpu().item()
            # print(idx)

            if idx == 1:
                a = "성적표현 입니다. 경고가 누적됩니다."
                break
            elif idx == 2:
                a = "혐오표현 입니다. 경고가 누적됩니다."
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
        send_text = "{}".format(a.strip())  # ❸

        warning_message = Warning_system(send_text, chat_id)
        send_message(chat_id, send_text)  # ❹
        if send_text == '혐오표현 입니다. 경고가 누적됩니다.' or send_text == '성적표현 입니다. 경고가 누적됩니다.':
            send_message(chat_id, warning_message)
        # send_message(chat_id, '당신의 매너수치는 ?')
    return last_update_id  # ❺


def Warning_system(send_text, chat_id):
    if chat_id not in id_dict:
        id_dict[chat_id] = 0
    if send_text == '혐오표현 입니다. 경고가 누적됩니다.' or send_text == '성적표현 입니다. 경고가 누적됩니다.':
        id_dict[chat_id] += 1
    return '총 {}회 경고가 누적되었습니다.'.format(id_dict[chat_id])


if __name__ == '__main__':  # ❶
    next_update_id = 0  # ❷
    id_dict = dict()

    while True:  # ❸
        last_update_id = check_messages_and_response(next_update_id)  # ❹
        if last_update_id:  # ❺
            next_update_id = last_update_id + 1
        time.sleep(2)  # ❻

    # '''
    # https://api.telegram.org/bot5130372966:AAHFNnCZQcNDQvxz8mtvbQlujYyqFH_MusE/getUpdates
    # '''
    # bot_token = "5130372966:AAHFNnCZQcNDQvxz8mtvbQlujYyqFH_MusE"
    # bot = telegram.Bot(token=bot_token)
    # bot.getUpdates("257478101")   ## update_id 넣으시요

