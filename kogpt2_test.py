import torch

from transformers import PreTrainedTokenizerFast


######   transformers==4.2.2
#####pip install --user transformers==4.16.2

ctx = "cuda" if torch.cuda.is_available() else "cpu"
print(ctx)
device = torch.device(ctx)
model = torch.load('../output/model/Mar06_KoGPT_chatbot_28.pt')
model.to(device)
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                            bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>',)
q_token = "<usr>"
a_token = "<sys>"
sent_token = '<unused1>'
sent= '0'

with torch.no_grad():
    while 1:
        q = input("user > ").strip()
        if q == "quit":
            break
        a = ""
        while 1:
            input_ids = torch.LongTensor(
                tokenizer.encode(q_token + q + sent_token + a_token + a)).unsqueeze(dim=0)
            input_ids = input_ids.to(device)
            # if ctx == 'cuda:0':
            #     input_ids = input_ids.to(ctx)
            pred = model(input_ids)
            pred = pred.logits
            # if ctx == 'cuda:0':
            #     pred = pred.cpu()
            gen = tokenizer.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
            if gen == tokenizer.eos_token:
                break
            a += gen.replace("▁", " ")
        print("Chatbot > {}".format(a.strip()))