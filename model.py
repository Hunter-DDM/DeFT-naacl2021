import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import math
from util import cudaw
import random


class Gate(nn.Module):
    def __init__(self, dhid, dfeature, init_range=0.1, init_dist='uniform', dropout=0.5):
        super(Gate, self).__init__()
        self.dhid = dhid
        self.dfeature = dfeature
        self.linear_z = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.linear_r = nn.Linear(self.dhid + self.dfeature, self.dfeature)
        self.linear_h_tilde = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights(init_range, init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.linear_z.weight.data, init_dist)
        init_w(self.linear_r.weight.data, init_dist)
        init_w(self.linear_h_tilde.weight.data, init_dist)
        self.linear_z.bias.data.fill_(0)
        self.linear_r.bias.data.fill_(0)
        self.linear_h_tilde.bias.data.fill_(0)

    def forward(self, h, features):
        z = self.sigmoid(self.linear_z(torch.cat((features, h), dim=1)))
        r = self.sigmoid(self.linear_r(torch.cat((features, h), dim=1)))
        h_tilde = self.tanh(self.linear_h_tilde(torch.cat((torch.mul(r, features), h), dim=1)))
        h_new = torch.mul((1 - z), h) + torch.mul(z, h_tilde)
        h_new = self.drop(h_new)
        return h_new


class NORASET(nn.Module):
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, dropout=0.5, tie_weights=False, init_range=0.1,
                 init_dist='uniform', seed_feeding=True, CH=False, nchar=-1, max_char_len=-1, use_morpheme=False, use_formation=False, use_wordvec=False, corpus=None):
        super(NORASET, self).__init__()
        self.seed_feeding = seed_feeding
        self.CH = CH

        self.drop = nn.Dropout(dropout)
        self.embedding = nn.Embedding(ntoken, dinp, padding_idx=corpus.word2id['<pad>'])
        self.nchar = nchar
        self.dinp = dinp
        self.dhid = dhid
        self.dfeature = dfeature
        self.use_morpheme = use_morpheme
        self.use_formation = use_formation
        self.use_wordvec = use_wordvec
        self.corpus = corpus
        if use_wordvec and use_morpheme: 
            # self.dfeature += dinp + 160 + dhid + dhid * 2  # d_att + d_emb + 160 + d_hid + d_mor_att
            self.dfeature += dinp + dinp + dhid + dhid  # d_att + d_emb + d_char_emb + d_hid + d_mor_att
        elif use_wordvec and not use_morpheme: 
            # self.dfeature += dinp + 160  # d_att + d_emb + 160 
            self.dfeature += dinp + dinp  # d_att + d_emb + d_char_emb 
        elif not use_wordvec and use_morpheme: 
            self.dfeature += dhid + dhid  # d_att + d_hid + d_mor_att

        if rnn_type in ['LSTM', 'GRU']:
            rnn = []
            for i in range(nlayers):
                if i == 0:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(dinp, self.dhid))
                else:
                    rnn.append(getattr(nn, rnn_type + 'Cell')(self.dhid, self.dhid))
            self.rnn = nn.ModuleList(rnn)
        else:
            raise ValueError("""An invalid option for `--model` was supplied, options are ['LSTM' or 'GRU']""")
        
        self.readout = nn.Linear(self.dhid, ntoken)

        if tie_weights:
            if self.dhid != dinp:
                raise ValueError('When using the tied flag, dhid must be equal to emsize')
            self.readout.weight = self.embedding.weight

        # Functions and parameters for Gated functions
        self.gate = Gate(self.dhid, self.dfeature)

        # Functions and parameters for Char CNNs
        if CH:
            if use_formation:
                self.ch_linear_combine = nn.ModuleList([nn.Linear(self.dhid * 2, self.dhid) for i in range(len(self.corpus.id2fm))])
            else:
                self.ch_linear_combine = nn.Linear(self.dhid * 2, self.dhid)

        self.init_weights(init_range=init_range, init_dist=init_dist, CH=CH)
        self.rnn_type = rnn_type + 'Cell'
        self.nlayers = nlayers

    def init_weights(self, init_range=0.1, init_dist='uniform', CH=True):
        if init_dist == 'uniform':
            self.embedding.weight.data.uniform_(-init_range, init_range)
            self.readout.weight.data.uniform_(-init_range, init_range)
            if CH:
                if self.use_formation:
                    for net in self.ch_linear_combine:
                        net.weight.data.uniform_(-init_range, init_range)
                else:
                    self.ch_linear_combine.weight.data.uniform_(-init_range, init_range)
        elif init_dist == 'xavier':
            nn.init.xavier_uniform(self.embedding.weight.data)
            nn.init.xavier_uniform(self.readout.weight.data)
            if CH:
                if self.use_formation:
                    for net in self.ch_linear_combine:
                        nn.init.xavier_uniform(net.weight.data)
                else:
                    nn.init.xavier_uniform(self.ch_linear_combine.weight.data)
        else:
            return False

        self.readout.bias.data.fill_(0)
        if CH:
            if self.use_formation:
                for net in self.ch_linear_combine:
                    net.bias.data.fill_(0)
            else:
                self.ch_linear_combine.bias.data.fill_(0)

    def stacked_rnn(self, input, hidden):
        h, c = hidden
        for layer in range(self.nlayers):
            if layer == 0:  # the first layer
                h[layer], c[layer] = self.rnn[layer](input, (h[layer], c[layer]))
            else:
                h[layer], c[layer] = self.rnn[layer](h[layer - 1], (h[layer], c[layer]))
            h[layer] = self.drop(h[layer])
        return (h, c)

    def init_hidden(self, bsz, dhid, bidirectional=False):
        weight = next(self.parameters()).data
        if bidirectional:
            hidden = weight.new((self.nlayers * 2), bsz, dhid).zero_()  # nlayer * 2, batch, d_hid
        else:
            hidden = [weight.new(bsz, dhid).zero_()] * self.nlayers  # nlayer, batch, d_hid

        if self.rnn_type == 'LSTM' or self.rnn_type == 'LSTMCell':
            if bidirectional:
                cell = weight.new((self.nlayers * 2), bsz, dhid).zero_()  # nlayer * 2, batch, d_hid
            else:
                cell = [weight.new(bsz, dhid).zero_()] * self.nlayers  # nlayer, batch, d_hid
            return hidden, cell
        else:
            return hidden

    def init_embedding(self, corpus, fix_embedding=False):
        print('Initializing word embedding layer...')
        word2vec = corpus.word2vec

        init_emb = []
        for id, word in enumerate(corpus.id2word):
            if word in word2vec:
                init_emb.append(word2vec[word])
            else:  # use random initialized embeddings
                init_emb.append([val for val in self.embedding.weight[id].data])
        self.embedding.weight.data = torch.FloatTensor(init_emb)

        if fix_embedding:
            self.embedding.weight.requires_grad = False

        print('  Done')

    def get_char_embedding(self, char_vecs, fms=None):
        # batch, d_emb * 2;
        if self.use_formation:
            char_emb = cudaw(torch.zeros(fms.shape[0], self.dhid))
            for i in range(fms.shape[0]):
                char_emb[i, :] = self.ch_linear_combine[fms[i].item()](char_vecs[i, :])  # batch, d_emb;
        else:
            # 使用统一的矩阵
            char_emb = self.ch_linear_combine(char_vecs)  # batch, d_emb;
            # average, used for analysis
            # char_emb = (char_vecs[:, 0:self.dhid] + char_vecs[:, self.dhid:]) / 2.0  # batch, d_emb;
        char_emb = F.relu(char_emb)
        char_emb = self.drop(char_emb)
        return char_emb


class Attention(nn.Module):
    def __init__(self, denc, ddec, datt, init_range=0.1, init_dist='uniform', dropout=0.0):
        super(Attention, self).__init__()
        self.denc, self.ddec, self.datt = denc, ddec, datt
        self.drop = nn.Dropout(dropout)
        self.W_enc = nn.Linear(self.denc, self.datt, bias=True)
        self.W_dec = nn.Linear(self.ddec, self.datt, bias=True)
        self.init_weights(init_range=init_range, init_dist=init_dist)

    def init_weights(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.W_enc.weight.data, init_dist)
        init_w(self.W_dec.weight.data, init_dist)
        self.W_enc.bias.data.fill_(0)
        self.W_dec.bias.data.fill_(0)

    def map_enc_states(self, h_enc):
        return self.W_enc(h_enc).permute(1, 0, 2) # (batch, len, dim)

    def att_score(self, h_dec, Wh_enc, topk=1):
        Wh_dec = self.W_dec(h_dec).repeat(1, topk).view(h_dec.size(0) * topk, -1).unsqueeze(2)  # (batch, d_hid) -> (batch, d_hid, 1)
        return torch.bmm(Wh_enc, Wh_dec).squeeze(2)  # (batch, max_len_eg, d_hid) * (batch, d_hid, 1) -> (batch, max_len_eg)

    def forward(self, h_dec, h_enc, h_enc_mask=None, Wh_enc=None, topk=1, flatten_enc_states=False, return_att_score=False, att_score=None):
        # h_enc: (len, k*b, 2dim)
        # Wh_enc: (k*b, len, dim)
        # h_dec:  (b, dim)
        if isinstance(Wh_enc, type(None)):
            Wh_enc = self.map_enc_states(h_enc)  # (batch, max_len_eg, d_hid)

        if isinstance(att_score, type(None)):
            att_score = self.att_score(h_dec, Wh_enc, topk)  # (batch, max_len_eg)

        if not isinstance(h_enc_mask, type(None)):
            att_score = att_score + h_enc_mask.permute(1, 0)  # (batch, max_len_eg)

        if return_att_score:
            return att_score

        if not flatten_enc_states:
            att_prob = F.softmax(att_score, dim=1)  # (batch, max_len_eg)
            att_result = torch.bmm(att_prob.unsqueeze(1), h_enc.permute(1, 0, 2)).squeeze(1)  # (batch, 1, max_len_eg), (batch, max_len_eg, d_hid * 2) -> (batch, d_hid * 2)
        else: # softmax over all words in top-k NNs
            batch = h_dec.size(0)
            h_enc_flat = h_enc.permute(1, 0, 2).contiguous().view(batch, -1, h_enc.size(2))  # (len, k*b, 2dim) -> (batch, len*k, 2dim)
            att_score_flat = att_score.view(batch, -1)  # (batch, len*topk)
            att_prob = F.softmax(att_score_flat, dim=1)  # softmax over all words in top-k NNs
            att_result = torch.bmm(att_prob.unsqueeze(1), h_enc_flat).squeeze(1)  # (batch, dim*2)

        att_result = self.drop(att_result)
        return att_result, att_prob


class EDIT(NORASET):
    def __init__(self, rnn_type, ntoken, dinp, dhid, dfeature, nlayers, max_len, corpus, dropout=0.5, tie_weights=False, init_range=0.1, init_dist='uniform', seed_feeding=True,  CH=False, nchar=-1, max_char_len=-1, use_morpheme=False, use_formation=False, use_wordvec=False):
        super(EDIT, self).__init__(rnn_type, ntoken, dinp, dhid, dfeature, nlayers, dropout, tie_weights, init_range, init_dist, seed_feeding=seed_feeding, CH=CH, nchar=nchar, max_char_len=max_char_len, use_morpheme=use_morpheme, use_formation=use_formation, use_wordvec=use_wordvec, corpus=corpus)
        self.name = 'EDIT'
        self.dhid_nn = dhid
        self.max_len = max_len
        self.attention = Attention(self.dhid * 2, self.dhid, self.dhid, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.attention_mor1 = Attention(self.dhid * 2, self.dhid, self.dhid, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.attention_mor2 = Attention(self.dhid * 2, self.dhid, self.dhid, init_range=init_range, init_dist=init_dist, dropout=dropout)
        self.softmax = nn.Softmax()

        # Encoder for descriptions of nearest neighbor words
        if rnn_type in ['LSTM', 'GRU']:
            self.nnEncoder = getattr(nn, rnn_type)(dinp, self.dhid_nn, nlayers, dropout=dropout, bidirectional=True)
            if use_morpheme:
                self.morEncoder = getattr(nn, rnn_type)(dinp, self.dhid_nn, nlayers, dropout=dropout, bidirectional=True)
                if use_formation:
                    self.fm_resizer = nn.ModuleList([nn.Linear(4 * dhid, dhid) for i in range(len(corpus.id2fm))])
                else:
                    self.fm_resizer = nn.Linear(4 * dhid, dhid)
    
    def get_nn_embedding(self, desc_nn):
        # encode description of the nearest neighbor word
        emb_nn = self.drop(self.embedding(desc_nn))  # maxlen, batch, d_emb
        hidden_nn = self.init_hidden(desc_nn.size(1), self.dhid_nn, bidirectional=True)  # nlayer * 2, batch, d_hid; 2*layer, batch, d_hid, 全 zeros
        nn_emb, hidden_nn = self.nnEncoder(emb_nn, hidden_nn)  # maxlen, batch, 2 * d_hid; nlayer * 2, batch, d_hid
        nn_emb = self.drop(nn_emb)  # maxlen, batch, 2 * d_hid; 
        return nn_emb
    
    def get_mor_embedding(self, mor, mor_len):
        # encode description of the nearest neighbor word
        emb_nn = self.drop(self.embedding(mor))  # maxlen, batch, d_emb
        hidden_nn = self.init_hidden(mor.size(1), self.dhid_nn, bidirectional=True)  # nlayer * 2, batch, d_hid; 2*layer, batch, d_hid, 全 zeros
        nn_emb, hidden_nn = self.morEncoder(emb_nn, hidden_nn)  # maxlen, batch, 2 * d_hid; nlayer * 2, batch, d_hid
        nn_emb = self.drop(nn_emb)  # maxlen, batch, 2 * d_hid; 
        review_nn_emb = nn_emb.view(nn_emb.shape[0], nn_emb.shape[1], 2, self.dhid)  # maxlen, batch, 2, d_hid; 
        hid_head = review_nn_emb[0, :, 1, :]
        hid_tail = cudaw(torch.zeros(hid_head.shape[0], hid_head.shape[1]))
        for i in range(hid_head.shape[0]):
            hid_tail[i, :] = review_nn_emb[mor_len[i] - 1, i, 0, :]
        hid_head_tail = torch.cat((hid_head, hid_tail), dim=1)  # batch, 2 * d_hid; 
        return nn_emb, hid_head_tail


class MORPHEME_EXT(EDIT):
    def __init__(self, args, corpus):
        if args.use_eg: 
            self.feature_num = args.dhid
        else:
            self.feature_num = 0
        if args.use_scheme: 
            self.feature_num += args.dhid
        super(MORPHEME_EXT, self).__init__(args.model,
                                              len(corpus.id2word),
                                              args.emsize,
                                              args.dhid,
                                              self.feature_num,
                                              args.nlayers,
                                              corpus.max_len,
                                              corpus, 
                                              dropout=args.dropout,
                                              tie_weights=args.tied,
                                              init_range=args.init_range,
                                              init_dist=args.init_dist,
                                              seed_feeding=args.seed_feeding,
                                              CH=args.char,
                                              nchar=len(corpus.id2char),
                                              max_char_len=corpus.max_char_len,
                                              use_morpheme=args.use_morpheme,
                                              use_formation=args.use_formation,
                                              use_wordvec=args.use_wordvec)

        random.seed(args.seed)
        self.name = 'MORPHEME_EXT'
        self.topk = 1
        self.reshape_att = nn.Linear(args.dhid * 2, args.dhid)
        self.reshape_att_mor1 = nn.Linear(args.dhid * 2, args.dhid)
        self.reshape_att_mor2 = nn.Linear(args.dhid * 2, args.dhid)
        self.use_morpheme = args.use_morpheme
        self.use_formation = args.use_formation
        self.use_eg = args.use_eg
        self.use_wordvec = args.use_wordvec
        self.use_scheme = args.use_scheme

        if self.use_wordvec and self.use_morpheme: 
            self.linear_combine = nn.Linear(self.dhid * 2, self.dhid)
        
        if self.use_morpheme:
            self.mor_gate = nn.Linear(self.dhid * 3, self.dhid)
        
        self.logsoftmax = nn.LogSoftmax()
        self.init_weights_MORPHEME_EXT(init_range=args.init_range, init_dist=args.init_dist)

    def init_weights_MORPHEME_EXT(self, init_range, init_dist):
        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.reshape_att.weight.data, init_dist)
        init_w(self.reshape_att_mor1.weight.data, init_dist)
        init_w(self.reshape_att_mor2.weight.data, init_dist)
        self.reshape_att.bias.data.fill_(0)
        self.reshape_att_mor1.bias.data.fill_(0)
        self.reshape_att_mor2.bias.data.fill_(0)
        if self.use_wordvec and self.use_morpheme: 
            init_w(self.linear_combine.weight.data, init_dist)
            self.linear_combine.bias.data.fill_(0)
        if self.use_morpheme:
            init_w(self.mor_gate.weight.data, init_dist)
            self.mor_gate.bias.data.fill_(0)
        

    def forward(self, input, hidden, vec, eg_emb, eg_mask, Wh_enc, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs, cuda, seed_feeding=True, char_emb=None,
                batch_ensemble=False, teacher_ratio=1.0):
        
        if self.use_morpheme:
            mor1_emb, mor1_repr = self.get_mor_embedding(mor1, mor1_len)  # maxlen_mor1, batch, 2 * d_hid; batch, d_hid * 2
            mor2_emb, mor2_repr = self.get_mor_embedding(mor2, mor2_len)  # maxlen_mor2, batch, 2 * d_hid; batch, d_hid * 2
            mor_concat_repr = torch.cat((mor1_repr, mor2_repr), dim=1)  # batch, d_hid * 4
            if self.use_formation:
                mor_repr = cudaw(torch.zeros(fms.shape[0], self.dhid))  # batch, d_hid
                for i in range(fms.shape[0]):
                    mor_repr[i, :] = self.fm_resizer[fms[i].item()](mor_concat_repr[i, :])
            else:
                mor_repr = self.fm_resizer(mor_concat_repr)  # batch, d_hid
                # average, used for analysis
                # mor_repr = (mor_concat_repr[:, self.dhid * 0:self.dhid * 1] + mor_concat_repr[:, self.dhid * 1:self.dhid * 2] + mor_concat_repr[:, self.dhid * 2:self.dhid * 3] + mor_concat_repr[:, self.dhid * 3:self.dhid * self.dhid]) / 4.0  # batch, d_hid
            mor_repr = self.drop(F.relu(mor_repr))
        
        # We must use at least one of wordvec and morpheme
        assert (self.use_wordvec or self.use_morpheme)
        if self.use_wordvec and self.use_morpheme: 
            # seed_vec = vec + mor_repr  # batch, d_hid
            seed_vec = self.linear_combine(torch.cat((vec, mor_repr), dim=1))
            seed_vec = self.drop(F.relu(seed_vec))
        elif self.use_wordvec and not self.use_morpheme: 
            seed_vec = vec  # batch, d_hid
        elif not self.use_wordvec and self.use_morpheme: 
            seed_vec = mor_repr  # batch, d_hid

        # prepend seed vectors to sentence vectors: [w*, <bos>, w1, w2, w3, ...]
        if seed_feeding:
            emb = self.drop(torch.cat((seed_vec.unsqueeze(0), self.embedding(input)), dim=0))  # 1 + maxlen, batch, d_emb
        else:
            emb = self.drop(self.embedding(input))  # maxlen, batch, d_emb
        batch = emb.size(1)

        # build feature vector
        if self.use_wordvec and self.use_morpheme: 
            features = torch.cat([vec, char_emb, mor_repr], dim=1)  # batch, d_emb + d_char_emb + d_hid
        elif self.use_wordvec and not self.use_morpheme: 
            features = torch.cat([vec, char_emb], dim=1)  # batch, d_emb + d_char_emb
        elif not self.use_wordvec and self.use_morpheme: 
            features = mor_repr  # batch, d_hid
        
        # scheme features
        if self.use_scheme:
            features = torch.cat([features, sm_vecs], dim=1)

        # decode words
        h, c = hidden  # nlayer, batch, d_hid; nlayer, batch, d_hid
        outputs = cudaw(torch.FloatTensor(emb.size(0), h[-1].size(0), h[-1].size(1)).zero_())  # 1 + maxlen, batch, d_hid

        for i in range(emb.size(0)):
            # teacher forcing
            use_teacher = random.random() < teacher_ratio
            if use_teacher or i == 0:
                current_input = emb[i]
            else:
                current_decoded = self.readout(outputs[i - 1])  # (batch, d_out)
                _, dec_word_id = torch.max(current_decoded, dim=1)  # (batch)
                current_input = self.embedding(dec_word_id)  # (batch, d_emb)
            
            # computes h, c in LSTM layers
            h, c = self.stacked_rnn(current_input, (h, c))  # nlayer, batch, d_hid; nlayer, batch, d_hid;

            if self.use_eg:
                # attention over source words
                att_result, att_prob = self.attention(h[-1], eg_emb, eg_mask, Wh_enc, topk=self.topk)  # (batch, d_hid * 2), (batch, max_len_eg)
                # reshape
                att_result = self.reshape_att(att_result)  # (batch, d_hid)
            
            if self.use_morpheme:
                # attention for mor1 and mor2
                Wh_enc_mor1 = self.attention_mor1.map_enc_states(mor1_emb)  # maxlen_mor1, batch, 2 * d_hid -> batch, maxlen_mor1, d_hid; 
                att_result_mor1, att_prob_mor1 = self.attention_mor1(h[-1], mor1_emb, mor1_mask, Wh_enc_mor1, topk=self.topk)  # (batch, d_hid * 2), (batch, max_len_mor1)
                att_result_mor1 = self.reshape_att_mor1(att_result_mor1)  # (batch, d_hid)

                Wh_enc_mor2 = self.attention_mor2.map_enc_states(mor2_emb)  # maxlen_mor2, batch, 2 * d_hid -> batch, maxlen_mor2, d_hid; 
                att_result_mor2, att_prob_mor2 = self.attention_mor2(h[-1], mor2_emb, mor2_mask, Wh_enc_mor2, topk=self.topk)  # (batch, d_hid * 2), (batch, max_len_mor2)
                att_result_mor2 = self.reshape_att_mor2(att_result_mor2)  # (batch, d_hid)

                # morpheme attention gate
                mor_gate = torch.sigmoid(self.mor_gate(torch.cat((h[-1], att_result_mor1, att_result_mor2), dim=1)))
                att_result_mor = att_result_mor1 * mor_gate + att_result_mor2 * (1 - mor_gate)
                # no gate, used for analysis
                # att_result_mor = (att_result_mor1 + att_result_mor2) / 2.0

            # gated function
            gate_features = features
            if self.use_eg:
                gate_features = torch.cat((gate_features, att_result), dim=1)
            if self.use_morpheme:
                gate_features = torch.cat((gate_features, att_result_mor), dim=1)
            h[-1] = self.gate(h[-1], gate_features)  # (batch, d_hid)
            outputs[i] = h[-1]  # (batch, d_hid)
        
        # outputs: (maxlen + 1, batch, d_hid)
        decoded = self.readout(outputs)
        # decoded: (maxlen + 1, batch, vocab)

        return decoded, (h, c)
