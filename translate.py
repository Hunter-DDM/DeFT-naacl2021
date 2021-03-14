import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
import random
import os
from subprocess import Popen, PIPE
from nltk.translate import bleu_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from util import cudaw


class Translator(object):
    def __init__(self, corpus, sentence_bleu=None, valid_all=False, mode='train'):
        self.corpus = corpus
        self.test = corpus.test
        self.ref = {"test": corpus.ref_test}

        if mode == 'train':
            self.valid = corpus.valid
            self.ref['valid'] = corpus.ref_valid
            self.train = corpus.train
            self.ref['train'] = corpus.ref_train

        random.seed('masaru')
        if mode == 'train':
            if not valid_all:
                self.random_samples = random.sample(range(min(len(self.valid), len(self.test))), 10)
            else:
                self.random_samples = list(range(len(self.valid)))

    def bleu(self, hyp, mode, nltk=None):
        """mode can be 'valid' or 'test' """
        if nltk == 'sentence': 
            score = 0
            num_hyp = 0
            for (word, desc) in hyp:
                bleu = bleu_score.sentence_bleu([ref.split() for ref in self.ref[mode][word]],
                                                desc,
                                                smoothing_function=bleu_score.SmoothingFunction().method2)
                score += bleu
                num_hyp += 1
            return score / num_hyp
        elif nltk == 'corpus':
            refs = []
            for (word, desc) in hyp:
                refs.append([ref.split() for ref in self.ref[mode][word]])
            bleu = bleu_score.corpus_bleu(refs,
                                          [word_desc[1] for word_desc in hyp],
                                          smoothing_function=bleu_score.SmoothingFunction().method2)
            return bleu
        else: 
            return -1

    def top1_copy(self, mode="valid", ignore_duplicates=False):
        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test

        if ignore_duplicates:
            data_new = []
            words = set()
            for i in range(len(data)):
                word = data[i][0].split('%', 1)[0]
                if word not in words:
                    data_new.append(data[i])
                    words.add(word)
            data = data_new

        top1_copied = [(data[i][0], data[i][4][0][1]) for i in range(len(data))]
        return top1_copied

    def draw_att_weights(self, att_mat_np, word, hyp, nns, path):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(att_mat_np, cmap='bone', vmin=0, vmax=1)
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + hyp, rotation=90, fontdict={'size': 8})
        ax.set_yticklabels([''] + nns, fontdict={'size': 8})

        # Show label at every tick
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlabel('Generated definition')
        ax.set_ylabel('Retrieved definitions')

        # plt.show()
        plt.savefig(path + '/' + word + '.pdf', bbox_inches='tight')
        return

    def visualize_att_weights(self, att_weights_batches, mode, topk, results, path):

        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test
        # remove duplicates
        data_new = []
        words = set()
        for i in range(len(data)):
            word = data[i][0].split('%', 1)[0]
            if word not in words:
                data_new.append(data[i])
                words.add(word)
        data = data_new

        i = 0
        for batch in att_weights_batches:
            max_src_len = batch.size(2) / topk
            for att_mat in batch:
                word, hyp = results[i]
                hyp_buf = hyp + ['<eos>']
                for j in range(len(hyp_buf)):
                    if hyp_buf[j] not in self.corpus.word2id:
                        hyp_buf[j] = '[' + hyp_buf[j] + ']'
                nns = []
                sliced_att_mat = []
                for k in range(topk):
                    nn = data[i][4][k][3]
                    nns.append(nn)
                    sliced_att_mat.append(att_mat[:(len(hyp_buf)), int(max_src_len * k): int(max_src_len * k + len(nn))])
                att_mat_np = torch.cat(sliced_att_mat, dim=1).permute(1, 0).cpu().data.numpy()
                nns_concat = []
                for nn in nns:
                    for w in nn:
                        if w not in self.corpus.word2id:
                            nns_concat.append('[' + w + ']')
                        else:
                            nns_concat.append(w)
                self.draw_att_weights(att_mat_np, word, hyp_buf, nns_concat, path)
                i += 1
        return 0

    def greedy(self, model, mode="valid", max_batch_size=128, cuda=True, max_len=60):
        if mode == "valid":
            data = self.valid
        elif mode == "test":
            data = self.test
        elif mode == "train":
            # test 1/5 of training examples，for saving time
            data = self.train[:len(self.train) // 5]
        
        model.eval()

        results = []
        batch_iter = self.corpus.batch_iterator(data, max_batch_size, cuda=cuda, mode=mode, seed_feeding=model.seed_feeding)
        with torch.no_grad():
            for i, elems in enumerate(batch_iter):
                batch_size = len(elems[0])
                decoded_words = [[] for x in range(batch_size)]

                hidden = model.init_hidden(batch_size, model.dhid)
                char_emb = None
                if model.CH:
                    if model.use_formation:
                        char_emb = model.get_char_embedding(elems[1], elems[7])  # batch, d_char_emb; 
                    else:
                        char_emb = model.get_char_embedding(elems[1])  # batch, d_char_emb; 

                input_word = torch.LongTensor([[self.corpus.word2id['<bos>']] * batch_size])  # 1, batch
                if cuda:
                    input_word = input_word.cuda()

                # Decode the first word
                (word, chars, vec, src, trg, eg, eg_mask, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs) = elems
                eg_emb = model.get_nn_embedding(eg)
                Wh_enc = model.attention.map_enc_states(eg_emb)  # (batch, len, dim)
                output, hidden = model(input_word, hidden, vec, eg_emb, eg_mask, Wh_enc, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs, cuda, seed_feeding=model.seed_feeding, char_emb=char_emb)  # 2, batch, vocab
                max_ids = output[-1].max(-1)[1]  # (batch); there may be two outputs if we use seed. Here we want the newest one
                input_word[0] = max_ids  # 1, batch
                keep_decoding = [1] * batch_size  # batch

                for k in range(len(keep_decoding)):
                    word_id = max_ids[k].data.item()
                    if self.corpus.id2word[word_id] != '<eos>':
                        decoded_words[k].append(self.corpus.id2word[word_id])
                    else:
                        keep_decoding[k] = 0

                # decode the subsequent words batch by batch
                for j in range(max_len):
                    output, hidden = model(input_word, hidden, vec, eg_emb, eg_mask, Wh_enc, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs, cuda, seed_feeding=False, char_emb=char_emb)
                    # map id to word
                    max_ids = output[0].max(-1)[1]  # (batch);
                    for k in range(len(keep_decoding)):
                        word_id = max_ids[k].data.item()
                        if keep_decoding[k]:
                            if self.corpus.id2word[word_id] != '<eos>':
                                decoded_words[k].append(self.corpus.id2word[word_id])
                            else:
                                keep_decoding[k] = 0
                        else:
                            pass
                    # feedback to the next step
                    input_word[0] = max_ids  # 1, batch
                    if max(keep_decoding) == 0:
                        break

                for k in range(len(decoded_words)):
                    results.append((word[k], decoded_words[k]))

        print('Decoded ' + mode + ' samples:')
        for sample in self.random_samples:
            sentence = ' '.join(results[sample][1])
            print(results[sample][0] + '\t' + sentence)

        return results

    def eval_loss(self, model, mode="valid", max_batch_size=128, cuda=True):
        if mode == 'valid':
            ntoken = self.corpus.valid_ntoken
        elif mode == 'test':
            ntoken = self.corpus.test_ntoken

        # Turn on evaluation mode which disables dropout.
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=model.corpus.word2id['<pad>'])

        total_loss = 0
        batch_iter = self.corpus.batch_iterator(self.valid if mode == 'valid' else self.test, max_batch_size, cuda=cuda, mode='valid', seed_feeding=model.seed_feeding)
        with torch.no_grad():
            for i, elems in enumerate(batch_iter):
                hidden = model.init_hidden(len(elems[0]), model.dhid)

                char_emb = None
                if model.CH:
                    if model.use_formation:
                        char_emb = model.get_char_embedding(elems[1], elems[7])  # batch, d_char_emb; 
                    else:
                        char_emb = model.get_char_embedding(elems[1])  # batch, d_char_emb; 

                (word, chars, vec, src, trg, eg, eg_mask, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs) = elems
                eg_emb = model.get_nn_embedding(eg)
                Wh_enc = model.attention.map_enc_states(eg_emb)  # (topk*batch, len, dim)
                output, hidden = model(src, hidden, vec, eg_emb, eg_mask, Wh_enc, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs, cuda, seed_feeding=model.seed_feeding, char_emb=char_emb)

                output_flat = output.view(output.size(0) * output.size(1), -1) # (trgLen, vocab)
                total_loss += criterion(output_flat, trg).data

        returns = [total_loss.item() / ntoken]

        return returns

    def print_log_loss(self, word, logloss, ref):
        # logloss: (trgLen), ref: [w1, w2, ..., <eos>]
        print(word + ':', end='')
        ref_buf = []
        for w in ref:
            if w not in self.corpus.word2id:
                ref_buf.append('[' + w + ']')
            else:
                ref_buf.append(w)
            print('\t{:>6s}'.format(ref_buf[-1]), end='')
        print('\n' + ' ' * (len(word) + 1), end='')
        for j, loss in enumerate(logloss):
            format = '\t{:>' + str(max(len(ref_buf[j]), 6)) + '.2f}'
            print(format.format(loss), end='')
        print()

    def print_log_loss_matrix(self, logloss_matrix, mode):
        # logloss_matrix (dataSize, trgLen)
        ref = [(self.corpus.test[k][0], self.corpus.test[k][5]) for k in range(len(self.corpus.test))]

        # logloss_matrix: (dataSize, trgLen)
        print('\n' + '-' * 150)
        for i, data in enumerate(logloss_matrix):
            words = ref[i][1][1:]
            logloss = data[1:len(ref[i][1])].tolist()
            self.print_log_loss(ref[i][0], logloss, words)
            print('-' * 150)
