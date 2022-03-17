import random
from transformers import BertTokenizer
import telegram
# pip install python-telegram-bot
# https://python.bakyeono.net/chapter-12-2.html


import torch, re
import numpy as np
from transformers import PreTrainedTokenizerFast
from kobert_tokenizer import KoBERTTokenizer

ctx = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(ctx)

badword_model = torch.load('./output/model/CSW_kCbert_3multi_0.93.pt', map_location=torch.device(ctx))
badword_model.to(device)

# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1') # KoBERT
tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-large')  # KcBERT

add_token = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㄲ', 'ㄸ', 'ㅃ', 'ㅆ', 'ㅉ', 'ㄳ', 'ㄵ', 'ㄶ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅄ']
tokenizer.add_tokens(add_token)


def transform(data):
    # data = re.sub(r"([!@#$%^&*()_+=,./?0-9])", r"", data)
    data = tokenizer(data)
    return np.array(data['input_ids']), np.array(data['token_type_ids']), np.array(data['attention_mask'])


chat_model = torch.load('./output/model/CSW_KoGPT_chatbot_14.89.pt', map_location=torch.device(ctx))
chat_model.to(device)
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                           bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                                                           pad_token='<pad>', mask_token='<unused0>')

import json
import time  # 추가함
import urllib.parse
import urllib.request

TOKEN = '5215471335:AAEl4LYXhaDt9fKAzZUC7f6j83hKyO-fHTc'  # 여러분의 토큰으로 변경


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
                message['text'] = '<<텍스트아님>>'
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
    global baduser_flag
    global notice
    last_update_id, recieved_messages = get_updates(next_update_id)  # ❶
    for message in recieved_messages:  # ❷
        chat_id = message['from_id']
        text = message['text']

        q = text
        a = ''
        while 1:
            if q == '<<텍스트아님>>':
                imoji = ["😂", "😍", "😚", "🤩", "🤔", "😉", "😻"]
                a = random.choice(imoji)
                break
            elif q == "잘못했습니다":
                _, _, _, a = Warning_system("잘못했습니다", chat_id)
                break
            elif q == "/gauge":
                baduser, bad_bar, max_badbar, _ = Warning_system("/gauge", chat_id)
                if not bad_bar:
                    a = f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
                    break
                else:
                    a = f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
                    break
            elif q == "/reset":
                baduser, bad_bar, max_badbar, a = Warning_system("/reset", chat_id)
                break
            else:
                check = transform(q)
                input_ids = torch.tensor([check[0]]).to(device)
                token_type_ids = torch.tensor([check[1]]).to(device)
                attention_mask = torch.tensor([check[2]]).to(device)
                result = badword_model(input_ids, token_type_ids, attention_mask)
                idx = result.argmax().cpu().item()
                # print(idx)

                if idx == 1:
                    a = "성적표현 입니다. 112번호가 몇번이죠?"
                    break
                elif idx == 2:
                    a = "혐오표현 입니다. 저도 귀한 집 자식이예요~"
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
        if a == "게이지가 초기화됐습니다":
            send_text = a
        else:
            send_text = "{}".format(a.strip())  # ❸

        baduser, bad_bar, max_badbar, warning_message = Warning_system(send_text, chat_id)
        if baduser:
            if q == "잘못했습니다":
                _, _, _, warning_message = Warning_system("잘못했습니다", chat_id)
                send_message(chat_id, warning_message)
                print(f"챗봇>> {warning_message}")

            else:
                if baduser_flag:
                    if notice:
                        send_message(chat_id, "비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                        print("챗봇>> 비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                        notice = False
                    else:
                        send_message(chat_id, "[응답없음]")

                else:
                    send_message(chat_id, send_text)
                    print(f"챗봇>> {send_text}")
                    send_message(chat_id, warning_message)
                    print(f"챗봇>>{warning_message}")
                    baduser_flag = True
                    send_message(chat_id, "비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                    print("챗봇>> 비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                    notice = False

        else:
            send_message(chat_id, send_text)  # ❹
            print(f"챗봇>> {send_text}")
            if send_text == '혐오표현 입니다. 저도 귀한 집 자식이예요~' or send_text == '성적표현 입니다. 112번호가 몇번이죠?':
                send_message(chat_id, warning_message)
                print(f"챗봇>> {warning_message}")
                if baduser_flag:
                    if notice:
                        send_message(chat_id, "비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                        print("챗봇>> 비매너 사용자에게는 응답하지 않습니다\n '잘못했습니다' 라고 빌면 용서해드리죠")
                        notice = False

        # send_message(chat_id, '당신의 매너수치는 ?')
    return last_update_id  # ❺


def Warning_system(send_text, chat_id):
    global bad_bar, baduser_flag, notice

    max_sorry = 1  # 봐주는 횟수
    max_badbar = 3  # 최대욕설 횟수
    if chat_id not in id_dict:
        id_dict[chat_id] = 0  # 욕설 카운트
    if chat_id not in sorry_dict:
        sorry_dict[chat_id] = max_sorry  # 봐주는 횟수

    if send_text == '혐오표현 입니다. 저도 귀한 집 자식이예요~' or send_text == '성적표현 입니다. 112번호가 몇번이죠?':
        if id_dict[chat_id] >= max_badbar:
            id_dict[chat_id] = max_badbar
        else:
            id_dict[chat_id] += 1
        print(id_dict[chat_id])

    elif send_text == "잘못했습니다":
        if (id_dict[chat_id] == 0) or (sorry_dict[chat_id] <= 0):
            print(id_dict[chat_id], sorry_dict[chat_id])
            return False, bad_bar, max_badbar, "더이상 봐드릴 수 없습니다"
        elif id_dict[chat_id] > 0:
            id_dict[chat_id] -= 1
            sorry_dict[chat_id] -= 1
            print(id_dict[chat_id], sorry_dict[chat_id])
            notice = True
            baduser_flag = False
            return False, bad_bar, max_badbar, "게이지 한칸만큼만 봐드리겠습니다."

    elif send_text == "/reset":
        id_dict[chat_id] = 0
        sorry_dict[chat_id] = max_sorry
        print(id_dict[chat_id], sorry_dict[chat_id])
        notice = True
        baduser_flag = False
        return False, bad_bar, max_badbar, "게이지가 초기화됐습니다"

    bad_bar = f"{int(id_dict[chat_id]) * '■'}{(max_badbar - int(id_dict[chat_id])) * '□'}"

    if id_dict[chat_id] > max_badbar:
        id_dict[chat_id] = max_badbar
        if id_dict[chat_id] < max_badbar:
            baduser = False
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
        elif id_dict[chat_id] == max_badbar:
            baduser = True
            print(baduser)
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {max_badbar}/{max_badbar}"
    else:
        if id_dict[chat_id] < max_badbar:
            baduser = False
            return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {id_dict[chat_id]}/{max_badbar}"
        elif id_dict[chat_id] == max_badbar:
            if baduser_flag:
                baduser = True
                return baduser, bad_bar, max_badbar, "비매너사용자"
            else:
                baduser = True
                return baduser, bad_bar, max_badbar, f"비속어 게이지 : {bad_bar} {max_badbar}/{max_badbar}"

    # if id_dict[chat_id] == 1:
    #     bad_bar = "■□□□□"
    # elif id_dict[chat_id] == 2:
    #     bad_bar = "■■□□□"
    # elif id_dict[chat_id] == 3:
    #     bad_bar = "■■■□□"
    # elif id_dict[chat_id] == 4:
    #     bad_bar = "■■■■□"
    # elif id_dict[chat_id] >= 5:
    #     bad_bar = "■■■■■"


if __name__ == '__main__':  # ❶
    next_update_id = 0  # ❷
    id_dict = dict()
    sorry_dict = dict()
    baduser_flag = False
    notice = True

    while True:  # ❸
        last_update_id = check_messages_and_response(next_update_id)  # ❹
        if last_update_id:  # ❺
            next_update_id = last_update_id + 1
        time.sleep(1)  # ❻

    '''초기화'''
    bot = telegram.Bot(token=TOKEN)
    # https://api.telegram.org/bot[자신의토큰]/getUpdates
    bot.getUpdates("업데이트ID")

    '''원하는말 보내기'''
    # chat_id = 5104167196
    # bot.sendMessage(chat_id=chat_id, text="너무 이뻐요")
