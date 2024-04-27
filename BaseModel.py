import torch
from torch import nn
from utils import *
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_size, embed_dim, hidden_size,n_layers, rnn_type = 'lstm', device = 'cuda'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = Embedding(input_size, embed_dim, PAD_IDX)
        self.rnn_type =  rnn_type
        self.dropout_in = nn.Dropout(p = 0.1)
        self.n_layers = n_layers
        self.device = device
        if rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_size,batch_first=True,bidirectional=True, num_layers = self.n_layers, dropout = 0.2)
        elif rnn_type == 'lstm':
            self.rnn = LSTM(embed_dim, hidden_size, batch_first=True,bidirectional=True, num_layers = n_layers,dropout = 0.2)

    def forward(self, enc_inp, src_len):
        sorted_idx = torch.sort(src_len, descending=True)[1]
        orig_idx = torch.sort(sorted_idx)[1]
        embedded = self.embedding(enc_inp)
        bs = embedded.size(0)
        output = self.dropout_in(embedded)
        if self.rnn_type == 'gru':
            hidden =  self.initHidden(bs)
            sorted_output = output[sorted_idx]
            sorted_len = src_len[sorted_idx]
            packed_output = nn.utils.rnn.pack_padded_sequence(sorted_output, sorted_len.data.tolist(), batch_first = True)
            packed_outs, hiddden = self.rnn(packed_output,(hidden, c))
            hidden = hidden[:,orig_idx,:]
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=PAD_IDX, batch_first = True)
            output = output[orig_idx]
            hidden = hidden.view(self.n_layers, 2, bs, -1).transpose(1, 2).contiguous().view(self.n_layers, bs, -1)
            return output, hidden, hidden
        elif self.rnn_type == 'lstm':
            hidden, c = self.initHidden(bs)
            sorted_output = output[sorted_idx]
            sorted_len = src_len[sorted_idx]
            packed_output = nn.utils.rnn.pack_padded_sequence(sorted_output, sorted_len.data.tolist(), batch_first = True)
            packed_outs, (hiddden, c) = self.rnn(packed_output,(hidden, c))
            hidden = hidden[:,orig_idx,:]
            c = c[:,orig_idx,:]
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=PAD_IDX, batch_first = True)
            output = output[orig_idx]
            c = c.view(self.n_layers, 2, bs, -1).transpose(1, 2).contiguous().view(self.n_layers, bs, -1)
            hidden = hidden.view(self.n_layers, 2, bs, -1).transpose(1, 2).contiguous().view(self.n_layers, bs, -1)
            return output, hidden, c
        
    def initHidden(self,bs):
        if self.rnn_type == 'gru' :
            return torch.zeros(self.n_layers*2, bs, self.hidden_size).to(self.device)
        elif self.rnn_type == 'lstm':
            return torch.zeros(self.n_layers*2,bs,self.hidden_size).to(self.device),torch.zeros(self.n_layers*2,bs,self.hidden_size).to(self.device)

class Attention_Module(nn.Module):
    def __init__(self, hidden_dim, output_dim, device = 'cuda'):
        super(Attention_Module, self).__init__()
        self.l1 = Linear(hidden_dim, output_dim, bias = False)
        self.l2 = Linear(hidden_dim+output_dim, output_dim, bias =  False)
        self.device = device
        
    def forward(self, hidden, encoder_outs, src_lens):
        x = self.l1(hidden)
        att_score = (encoder_outs.transpose(0,1) * x.unsqueeze(0)).sum(dim = 2)
        seq_mask = sequence_mask(src_lens, max_len = max(src_lens).item(), device = self.device).transpose(0,1)
        masked_att = seq_mask*att_score
        masked_att[masked_att==0] = -1e10
        attn_scores = F.softmax(masked_att, dim=0)
        x = (attn_scores.unsqueeze(2) * encoder_outs.transpose(0,1)).sum(dim=0)
        x = torch.tanh(self.l2(torch.cat((x, hidden), dim=1)))
        return x, attn_scores
        
class AttentionDecoderRNN(nn.Module):
    def __init__(self, output_size, embed_dim, hidden_size, n_layers = 1, attention = True, device = 'cuda'):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        encoder_output_size = hidden_size
        self.embedding = Embedding(output_size, embed_dim, PAD_IDX)
        self.dropout = nn.Dropout(p=0.1)
        self.n_layers = n_layers
        self.device = device
        self.att_layer = Attention_Module(self.hidden_size, encoder_output_size,self.device) if attention else None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=self.hidden_size + embed_dim if ((layer == 0) and attention) else embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(self.n_layers)
        ])
        self.fc_out = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, dec_input,context_vector, prev_hiddens,prev_cs,encoder_outputs,src_len):
        bsz = dec_input.size(0)
        output = self.embedding(dec_input)
        output = self.dropout(output)
        if self.att_layer is not None:
            cated_input = torch.cat([output.squeeze(1),context_vector], dim = 1)
        else:
            cated_input = output.squeeze(1)
        new_hiddens = []
        new_cs = []
        for i, rnn in enumerate(self.layers):
            hidden, c = rnn(cated_input, (prev_hiddens[i], prev_cs[i]))
            cated_input = self.dropout(hidden)
            new_hiddens.append(hidden.unsqueeze(0))
            new_cs.append(c.unsqueeze(0))
        new_hiddens = torch.cat(new_hiddens, dim = 0)
        new_cs = torch.cat(new_cs, dim = 0)

        # apply attention using the last layer's hidden state
        if self.att_layer is not None:
            out, attn_score = self.att_layer(hidden, encoder_outputs, src_len)
        else:
            out = hidden
            attn_score = None
        context_vec = out
        out = self.dropout(out)
        out_vocab = self.softmax(self.fc_out(out))

        return out_vocab, context_vec, new_hiddens, new_cs, attn_score