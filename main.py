from fastapi import FastAPI
from BaseModel import *
from load_dataset_viet import train_val_load
import numpy as np
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient
import os

UNK_IDX = 2
PAD_IDX = 3
SOS_token = 0
EOS_token = 1

connection_str = os.environ.get('AZURE_CONNECTION_STRING') or ''
container_name = 'model'

app = FastAPI()

def read_model(model, file):
    # blob_client = blob_service_client.get_blob_client(container=container_name, blob=file)
    # model_blog = blob_client.download_blob()
    # with open(file, 'wb') as f:
    #     f.write(model_blog.readall())
    # model.load_state_dict(torch.load(file))
    pass

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# train,val,test,en_lang,vi_lang = train_val_load(48, "")
# blob_service_client = BlobServiceClient.from_connection_string(connection_str)

# encoder_w_att = EncoderRNN(vi_lang.n_words, 512, 512, 2).to(device)
# decoder_w_att = AttentionDecoderRNN(en_lang.n_words, 512, 1024, n_layers=2, attention=True).to(device)
# read_model(encoder_w_att, 'lstm_att_enc_2_layer.pth')
# read_model(decoder_w_att, 'lstm_att_dec_2_layer.pth')

# def translate(sentence: str):
#     sentence = sentence.split(' ')
#     idx = []
#     for i in sentence:
#         try:
#             idx.append(vi_lang.word2index[i])
#         except:
#             idx.append(UNK_IDX)
#     a = torch.from_numpy(np.array(idx)).unsqueeze(0).to(device)
#     encoder_i = a.to(device)
#     src_len = torch.tensor([a.size(1)]).to(device)
#     bs,sl = encoder_i.size()[:2]
#     en_out,en_hid,en_c = encoder_w_att(encoder_i,src_len)
#     max_src_len_batch = max(src_len).item()
#     prev_hiddens = en_hid
#     prev_cs = en_c
#     decoder_input = torch.tensor([[SOS_token]]*bs).to(device)
#     prev_output = torch.zeros((bs, en_out.size(-1))).to(device)
#     d_out = []
#     for i in range(sl*2):
#         out_vocab, prev_output,prev_hiddens, prev_cs, attention_score = decoder_w_att(decoder_input,prev_output, \
#                                                                                 prev_hiddens,prev_cs, en_out,\
#                                                                                 src_len)
#         topv, topi = out_vocab.topk(1)
#         d_out.append(topi.item())
#         decoder_input = topi.squeeze().detach().view(-1,1)
#         if topi.item() == EOS_token:
#             break

#     pred_sent = convert_id_list_2_sent(d_out,en_lang)
#     return pred_sent

class Sentence(BaseModel):
    key: str

@app.get("/")
async def hello():
    return {"message": "Hello World!"}

@app.post("/api/translate")
async def hello(sentence: Sentence):
    sentence = sentence.key
    # pred_sent = translate(sentence)
    # return {"translated_sentence": pred_sent}


