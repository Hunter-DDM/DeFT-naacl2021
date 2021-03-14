import os
import sys
import torch
import random
import numpy as np
np.random.seed(505)
from util import cudaw
from gensim.models import KeyedVectors
import pickle
import jieba.posseg as pseg


class MorphemeCorpus(object):
    def __init__(self, args, mode='train'):
        # Builds a vocabulary set from train data
        self.special_tokens = ['<TRG>', '<unk>', '<bos>', '<eos>', '<pad>']
        self.args = args
        self.vec_path = args.vec
        self.max_char_len = 0
        self.max_vocab_size = args.vocab_size
        self.data_path = args.data
        try:
            self.word2id, self.id2word = self.load_vocab(os.path.join(self.data_path, 'train.txt.vocab'), self.max_vocab_size)
            print('Loaded vocab file from ' + os.path.join(self.data_path, 'train.txt.vocab'))
            self.char2id, self.id2char = self.load_vocab(os.path.join(self.data_path, 'train.txt.char'), -1)
            print('Loaded char vocab file from ' + os.path.join(self.data_path, 'train.txt.char'))
        except:
            print('No vocab files in ' + os.path.join(self.data_path, 'train.txt.vocab'))
            print('               or ' + os.path.join(self.data_path, 'train.txt.char'))
            self.word2id, self.id2word, self.char2id, self.id2char = self.build_vocab(os.path.join(self.data_path, 'train.txt'), os.path.join(self.data_path, 'train.ext'), self.max_vocab_size)
            print('Output vocab file to ' + os.path.join(self.data_path, 'train.txt.vocab'))
            print('                 and ' + os.path.join(self.data_path, 'train.txt.char'))

        # Tokenize tokens in text data
        self.max_len = 0
        print('Reading corpora...')
        if mode == 'train':
            self.train, self.train_ntoken, self.ref_train, self.train_orig = self.tokenize(os.path.join(self.data_path, 'train.txt'))
            self.valid, self.valid_ntoken, self.ref_valid, self.valid_orig = self.tokenize(os.path.join(self.data_path, 'valid.txt'))
        self.test, self.test_ntoken, self.ref_test, self.test_orig = self.tokenize(os.path.join(self.data_path, 'test.txt'))
        print('class NorasetCorpus: Read Data Done')

        # Reads vectors
        print('Reading vector file...')
        self.word2vec = self.read_vector(self.vec_path)
        if mode == 'train':
            self.train = self.add_vector(self.train)
            self.valid = self.add_vector(self.valid)
        self.test = self.add_vector(self.test)
        print('class NorasetCorpus: Read Vectors Done')

        # Convert words into char-ids
        if mode == 'train':
            self.train = self.add_chars(self.train)
            self.valid = self.add_chars(self.valid)
        self.test = self.add_chars(self.test)
        
        # =============================================================================================

        self.fm2id = {}
        self.id2fm = []
        print('Reading extend files...')
        if args.use_scheme:
            self.schemes = self.read_schemes(os.path.join(self.data_path, 'sememes.txt'))
        if mode == 'train':
            self.train = self.add_ext_info(self.train, os.path.join(self.data_path, 'train.ext'))
            self.valid = self.add_ext_info(self.valid, os.path.join(self.data_path, 'valid.ext'))
        self.test = self.add_ext_info(self.test, os.path.join(self.data_path, 'test.ext'))
    
    def read_schemes(self, path):
        assert os.path.exists(path)
        schemes = {}
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                items = line.strip().split('|||')
                word = items[0].strip()
                scheme_words = items[1].strip().split(' ')
                if word not in schemes:
                    schemes[word] = set()
                schemes[word].update(scheme_words)
        return schemes
    
    def update_vocab(self, word2freq, char2freq, word_list):
        for w in word_list:
            if w in word2freq:
                word2freq[w] += 1
            elif w not in self.special_tokens:  # ignore special tokens included in training data
                word2freq[w] = 1

            if w not in self.special_tokens:
                for c in w: 
                    if c in char2freq:
                        char2freq[c] += 1
                    else:
                        char2freq[c] = 1

    def build_vocab(self, path_txt, path_ext, max_vocab_size):
        assert os.path.exists(path_txt)
        assert os.path.exists(path_ext)
        word2freq = {}
        char2freq = {}
        with open(path_txt, 'r') as f:
            for line in f:
                # word = line.strip().split('\t')[0]
                description = line.strip().split('\t')[1].split()
                # self.update_vocab(word2freq, char2freq, list(word))
                self.update_vocab(word2freq, char2freq, description)
                
        with open(path_ext, 'r') as f:
            for line in f:
                cxt = line.strip().split('\t')[1].split()
                mor1 = line.strip().split('\t')[3].split()
                mor2 = line.strip().split('\t')[4].split()
                self.update_vocab(word2freq, char2freq, cxt)
                self.update_vocab(word2freq, char2freq, mor1)
                self.update_vocab(word2freq, char2freq, mor2)

        # sort vocabularies in order of frequency and prepend special tokens
        sorted_word2freq = sorted(word2freq.items(), key=lambda x: -x[1])
        for special_token in self.special_tokens:
            sorted_word2freq.insert(0, (special_token, 0))
        id2word = []
        word2id = {}
        with open(path_txt + '.vocab', 'w') as f:
            for i, (word, freq) in enumerate(sorted_word2freq):
                f.write(word + '\t' + str(i) + '\t' + str(freq) + '\n')
                if max_vocab_size == -1 or i < max_vocab_size + len(self.special_tokens):
                    word2id[word] = i
                    id2word.append(word)

        # Do the same things to char vocabs
        sorted_char2freq = sorted(char2freq.items(), key=lambda x: -x[1])
        for special_token in ['<bow>', '<eow>']:
            sorted_char2freq.insert(0, (special_token, 0))
        id2char = []
        char2id = {}
        with open(path_txt + '.char', 'w') as f:
            for i, (char, freq) in enumerate(sorted_char2freq):
                f.write(char + '\t' + str(i) + '\t' + str(freq) + '\n')
                char2id[char] = i
                id2char.append(char)

        return word2id, id2word, char2id, id2char

    def load_vocab(self, path, max_vocab_size):
        assert os.path.exists(path)
        id2word = []
        word2id = {}
        for line in open(path, 'r'):
            word_id = line.strip().split('\t', 2)[:2]
            if max_vocab_size == -1 or int(word_id[1]) < max_vocab_size + len(self.special_tokens):
                word2id[word_id[0]] = int(word_id[1])
                id2word.append(word_id[0])
            else:
                break
        return word2id, id2word

    def tokenize(self, path):
        assert os.path.exists(path)
        word_desc = []   # [(srcWord0, [trgId0, trgId1, ...]), (srcWord1, [trgId0, trgId1, ...])]
        word_desc_orig = []  # [(srcWord0, [trgWord0, trgWord1, ...]), ... ]
        ntoken = 0
        ref = {}
        with open(path, 'r') as f:
            for line in f:
                elems = line.strip().split('\t')
                word = elems[0]
                if word not in ref:
                    ref[word] = []
                ref[word].append(elems[1])

                if len(word) + 2 > self.max_char_len:
                    self.max_char_len = len(word) + 2
                
                word_desc.append((word, []))
                description = ['<bos>'] + elems[1].split() + ['<eos>']
                word_desc_orig.append((word, description))
                for w in description:
                    if w in self.word2id:
                        word_desc[-1][1].append(self.word2id[w])
                    else:
                        word_desc[-1][1].append(self.word2id['<unk>'])

                ntoken += (len(description) - 1)  # including <eos>, not including <bos>
                if len(description) - 1 > self.max_len:
                    self.max_len = len(description) - 1

        return word_desc, ntoken, ref, word_desc_orig

    def read_vector(self, path):
        """Reads word2vec file."""
        assert os.path.exists(path)
        ckpt_path = os.path.join(self.data_path, 'word2vec_fast_checkpoint.pkl')
        if os.path.exists(ckpt_path): 
            with open(ckpt_path, 'rb') as f:
                word2vec = pickle.load(f)
                return word2vec
        word2vec = {}  # {srdWord0: [dim0, dim1, ..., dim299], srcWord1: [dim0, dim1, ..., dim299]}

        vecs = KeyedVectors.load_word2vec_format(path)
        vec_vocab = vecs.vocab
        print(len(vec_vocab))
        for item in vec_vocab:
            word2vec[item] = [float(dimension) for dimension in vecs[item]]

        if '<unk>' not in word2vec:
            word2vec['<unk>'] = np.random.uniform(-0.05, 0.05, len(word2vec[list(word2vec.keys())[0]])).tolist()

        with open(ckpt_path, 'wb') as f:
            pickle.dump(word2vec, f)
        
        return word2vec

    def add_vector(self, word_desc):
        word_vec_desc = []

        for word, description in word_desc:
            if word in self.word2vec:
                word_vec_desc.append((word, self.word2vec[word], description))
            else:
                word_lemma = word.split('%', 1)[0]
                if word_lemma in self.word2vec:
                    word_vec_desc.append((word, self.word2vec[word_lemma], description))
                else:
                    word_vec_desc.append((word, self.word2vec['<unk>'], description))

        return word_vec_desc

    def add_chars(self, word_vec_desc):
        """convert words ([word1, word2, ...]) into char-ids """
        word_char_vec_desc = []
        for word, vec, description in word_vec_desc:
            # char_ids = [self.char2id['<bow>']]
            # for c in word: 
            #     if c in self.char2id:
            #         char_ids.append(self.char2id[c])
            # char_ids.append(self.char2id['<eow>'])
            char_vecs = []
            for c in word: 
                if c in self.word2vec:
                    char_vecs.append(self.word2vec[c])
                else:
                    char_vecs.append(self.word2vec['<unk>'])
            # word_char_vec_desc.append((word, char_ids, vec, description))
            word_char_vec_desc.append((word, char_vecs, vec, description))

        return word_char_vec_desc

    def sample_train_data(self, data_usage):
        sample_size = int(len(self.train) * data_usage / 100)
        self.train = self.train[:sample_size]
        return
    
    def build_rule_based_eg(self, word): 
        posseg_list = list(pseg.cut(word))
        pos = list(posseg_list[-1])[1]
        if pos == 'n':
            return ['这', '是', '<TRG>']
        elif pos == 'nr':
            return ['这', '是', '<TRG>']
        elif pos == 'nz':
            return ['这', '是', '<TRG>']
        elif pos == 'a':
            return ['这', '很', '<TRG>']
        if pos == 'm':
            return ['三', '<TRG>']
        elif pos == 'c':
            return ['<TRG>']
        elif pos == 'f':
            return ['这', '在', '<TRG>']
        elif pos == 'ns':
            return ['这', '在', '<TRG>']
        if pos == 'v':
            return ['我们', '<TRG>']
        elif pos == 'ad':
            return ['这', '<TRG>']
        elif pos == 'q':
            return ['三', '<TRG>']
        elif pos == 'u':
            return ['<TRG>']
        if pos == 's':
            return ['这', '是', '<TRG>']
        elif pos == 'nt':
            return ['这', '是', '<TRG>']
        elif pos == 'vd':
            return ['<TRG>']
        elif pos == 'an':
            return ['<TRG>']
        if pos == 'r':
            return ['<TRG>', '很', '好']
        elif pos == 'xc':
            return ['<TRG>']
        elif pos == 't':
            return ['现在', '是', '<TRG>']
        elif pos == 'nw':
            return ['我们的', '<TRG>']
        if pos == 'vn':
            return ['<TRG>']
        elif pos == 'd':
            return ['<TRG>']
        elif pos == 'p':
            return ['这', '<TRG>', '这里']
        elif pos == 'w':
            return ['<TRG>']
        if pos == 'PER':
            return ['这', '是', '<TRG>']
        elif pos == 'LOC':
            return ['这', '在', '<TRG>']
        elif pos == 'ORG':
            return ['这', '在', '<TRG>']
        elif pos == 'TIME':
            return ['现在', '是', '<TRG>']
        else:
            return ['<TRG>']

    def add_ext_info(self, word_char_vec_desc, path):
        assert os.path.exists(path)
        word_char_vec_desc_eg_fm_mor1_mor2_sm = []
        with open(path, 'r') as f:
            for i, line in enumerate(f):
                word, eg, fm, mor1, mor2 = line.strip().split('\t')
                (word, char, vec, desc) = word_char_vec_desc[i]
                # formation
                if fm not in self.fm2id:
                    self.fm2id[fm] = len(self.id2fm)
                    self.id2fm.append(fm)
                fm_id = self.fm2id[fm]
                # word example
                eg = eg.split(' ')
                if self.args.simple_eg:
                    eg = self.build_rule_based_eg(word)
                eg_id = []
                for w in ['<bos>'] + eg + ['<eos>']:
                    if w in self.word2id:
                        eg_id.append(self.word2id[w])
                    else:
                        eg_id.append(self.word2id['<unk>'])
                # morpheme 1
                mor1 = mor1.split(' ')
                mor1_id = []
                for w in ['<bos>'] + mor1 + ['<eos>']:
                    if w in self.word2id:
                        mor1_id.append(self.word2id[w])
                    else:
                        mor1_id.append(self.word2id['<unk>'])
                # morpheme 2
                mor2 = mor2.split(' ')
                mor2_id = []
                for w in ['<bos>'] + mor2 + ['<eos>']:
                    if w in self.word2id:
                        mor2_id.append(self.word2id[w])
                    else:
                        mor2_id.append(self.word2id['<unk>'])
                # scheme
                sm_word = []
                if self.args.use_scheme and word in self.schemes:
                    schemes = list(self.schemes[word])
                else: 
                    schemes = ['<unk>']
                for w in schemes:
                    if w in self.word2vec:
                        sm_word.append(w)
                    else:
                        sm_word.append('<unk>')
                word_char_vec_desc_eg_fm_mor1_mor2_sm.append((word, char, vec, desc, eg_id, fm_id, mor1_id, mor2_id, sm_word))

        return word_char_vec_desc_eg_fm_mor1_mor2_sm

    def batch_iterator(self, word_char_vec_desc_eg_fm_mor1_mor2, max_batch_size, cuda=False, shuffle=False, mode='train',
                       ignore_index=-100, seed_feeding=True):
        """
        mode: {train (use whole data; volatile=False) | valid (use whole data; volatile=True) | test (don't use duplicate data; volatile=True)}
        """
        data = word_char_vec_desc_eg_fm_mor1_mor2
        ignore_index = self.word2id['<pad>']
        if shuffle:
            random.shuffle(data)
        batch_num = -(-len(data) // max_batch_size)
        batch_size = max_batch_size
        for i in range(batch_num):
            # for the last remainder batch, set the batch size smaller than others
            if i == batch_num - 1 and len(data) % max_batch_size != 0:
                batch_size = len(data) % max_batch_size

            batch = data[i * max_batch_size:(i+1) * max_batch_size]
            words = []
            # chars = torch.FloatTensor(batch_size, self.max_char_len, len(self.id2char)).zero_()
            char_vecs = torch.FloatTensor(batch_size, len(data[0][2]) * 2)
            vecs = torch.FloatTensor(batch_size, len(data[0][2]))
            srcs_T = torch.LongTensor(batch_size, self.max_len)
            fms = torch.LongTensor(batch_size)
            sm_vecs = torch.FloatTensor(batch_size, len(data[0][2])).zero_()

            # tensors for eg
            max_len_eg = max([len(batch[i][4]) for i in range(len(batch))])
            eg_T = torch.LongTensor(batch_size, max_len_eg)
            eg_mask_T = torch.FloatTensor(batch_size, max_len_eg).fill_(-float('inf'))
            # tensors for mor1 and mor2
            max_len_mor1 = max([len(batch[i][6]) for i in range(len(batch))])
            mor1_T = torch.LongTensor(batch_size, max_len_mor1)
            mor1_len = torch.LongTensor(batch_size)
            mor1_mask_T = torch.FloatTensor(batch_size, max_len_mor1).fill_(-float('inf'))
            max_len_mor2 = max([len(batch[i][7]) for i in range(len(batch))])
            mor2_T = torch.LongTensor(batch_size, max_len_mor2)
            mor2_len = torch.LongTensor(batch_size)
            mor2_mask_T = torch.FloatTensor(batch_size, max_len_mor2).fill_(-float('inf'))

            if seed_feeding:
                trgs_T = torch.LongTensor(batch_size, self.max_len + 1) # src is 1-element less than trg because we will feed Seed vector later
            else:
                trgs_T = torch.LongTensor(batch_size, self.max_len)

            for j, (word, char, vec, description, example, fm, mor1, mor2, sm) in enumerate(batch):
                words.append(word)
                fms[j] = fm
                # one-hot repl. of words
                # for k, c in enumerate(char):
                #     chars[j, k, c] = 1.0
                # concatenated char vectors
                char_vecs[j, :len(data[0][2])] = torch.tensor(char[0]).float()
                char_vecs[j, len(data[0][2]):] = torch.tensor(char[1]).float()

                # scheme vectors
                for k in range(len(sm)):
                    sm_vecs[j] += torch.FloatTensor(self.word2vec[sm[k]])
                sm_vecs[j] /= len(sm)

                # padding the source id sequence with <eos>
                padded_desc = description[:-1] + [self.word2id['<eos>']] * (self.max_len - len(description[:-1]))
                # padding the target id sequence with ignore_indices
                if seed_feeding:
                    shifted_desc = [ignore_index] + description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                else:
                    shifted_desc = description[1:] + [ignore_index] * (self.max_len - len(description[:-1]))
                vecs[j] = torch.FloatTensor(vec)
                srcs_T[j] = torch.Tensor(padded_desc)
                trgs_T[j] = torch.Tensor(shifted_desc)

                # padding the examples
                padded_eg = example + [ignore_index] * (max_len_eg - len(example))
                eg_T[j] = torch.Tensor(padded_eg)
                eg_mask_T[j][:len(example)].fill_(0)
                # padding the mor1 and mor2
                padded_mor1 = mor1 + [ignore_index] * (max_len_mor1 - len(mor1))
                mor1_len[j] = len(mor1)
                mor1_T[j] = torch.Tensor(padded_mor1)
                mor1_mask_T[j][:len(mor1)].fill_(0)
                padded_mor2 = mor2 + [ignore_index] * (max_len_mor2 - len(mor2))
                mor2_len[j] = len(mor2)
                mor2_T[j] = torch.Tensor(padded_mor2)
                mor2_mask_T[j][:len(mor2)].fill_(0)

            # reshape the tensors
            srcs = torch.transpose(srcs_T, 0, 1).contiguous()  # maxlen, batch
            trgs = torch.transpose(trgs_T, 0, 1).contiguous().view(-1)  # maxlen, batch
            eg = torch.transpose(eg_T, 0, 1).contiguous()  # maxlen_eg, batch
            eg_mask = torch.transpose(eg_mask_T, 0, 1).contiguous()  # maxlen_eg, batch
            mor1 = torch.transpose(mor1_T, 0, 1).contiguous()  # maxlen_mor1, batch
            mor1_mask = torch.transpose(mor1_mask_T, 0, 1).contiguous()  # maxlen_mor1, batch
            mor2 = torch.transpose(mor2_T, 0, 1).contiguous()  # maxlen_mor2, batch
            mor2_mask = torch.transpose(mor2_mask_T, 0, 1).contiguous()  # maxlen_mor2, batch

            yield (
                words, 
                # cudaw(chars.unsqueeze(1)), 
                cudaw(char_vecs), 
                cudaw(vecs),
                cudaw(srcs),
                cudaw(trgs),
                cudaw(eg),
                cudaw(eg_mask),
                cudaw(fms),
                cudaw(mor1),
                cudaw(mor1_len),
                cudaw(mor1_mask),
                cudaw(mor2),
                cudaw(mor2_len),
                cudaw(mor2_mask), 
                cudaw(sm_vecs)
            )
