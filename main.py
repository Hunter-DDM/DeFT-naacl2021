import argparse
import time
import math
from builtins import enumerate

import torch
import torch.nn as nn
import numpy as np
import data
import model
import translate


def load_checkpoint():
    checkpoint = torch.load(args.save)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def save_checkpoint():
    check_point = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(check_point, args.save)


###############################################################################
# Training code
###############################################################################
def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0

    batch_iter = corpus.batch_iterator(corpus.train, args.batch_size, cuda=args.cuda, shuffle=True, mode='train', seed_feeding=model.seed_feeding)
    batch_num = -(-len(corpus.train) // args.batch_size)
    start_time = time.time()
    for i, elems in enumerate(batch_iter):
        optimizer.zero_grad()

        hidden = model.init_hidden(len(elems[0]), model.dhid)  # nlayer, batch, d_hid; nlayer, batch, d_hid; 
        if args.char:
            if args.use_formation:
                char_emb = model.get_char_embedding(elems[1], elems[7])  # batch, d_char_emb; 
            else:
                char_emb = model.get_char_embedding(elems[1])  # batch, d_char_emb; 
        else:
            char_emb = None

        (word,  # batch
        chars,  # batch, d_emb * 2
        vec,  # batch, d_emb
        src,  # maxlen, batch
        trg,  # maxlen + 1
        eg,  # maxlen_eg, batch
        eg_mask, # maxlen_eg, batch
        fms, # batch
        mor1,  # maxlen_mor1, batch
        mor1_len, # batch, mor1 
        mor1_mask, # maxlen_mor1, batch
        mor2,  # maxlen_mor2, batch
        mor2_len, # batch, mor2
        mor2_mask, # maxlen_mor2, batch
        sm_vecs # batch, d_emb
        ) = elems
        eg_emb = model.get_nn_embedding(eg)  # maxlen_eg, batch, 2 * d_hid; 
        Wh_enc = model.attention.map_enc_states(eg_emb)  # batch, maxlen_eg, d_hid; 
        output, hidden = model(src, hidden, vec, eg_emb, eg_mask, Wh_enc, fms, mor1, mor1_len, mor1_mask, mor2, mor2_len, mor2_mask, sm_vecs, args.cuda, seed_feeding=model.seed_feeding, char_emb=char_emb, teacher_ratio=args.teacher_ratio)  # maxlen + 1, batch, vocab

        loss = criterion(output.view(output.size(0) * output.size(1), -1), trg)  # (maxlen + 1) * batch, vocab; (maxlen + 1) * batch
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.data

        if (i+1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss.item() / args.log_interval
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.07f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, i+1, batch_num, lr, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)), 
                flush=True
            )
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Morpheme')
    parser.add_argument('--data', type=str, default='./data/noraset',
                        help='location of the data corpus')
    parser.add_argument('--data_usage', type=int, default=100,
                        help='how many train data to be used (0 - 100 [%])')
    parser.add_argument('--vocab_size', type=int, default=-1,
                        help='vocabulary size (-1 = all vocabl)')
    parser.add_argument('--vec', type=str, default='./data/GoogleNews-vectors-negative300.txt',
                        help='location of the word2vec data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--init_dist', type=str, default='uniform',
                        help='distribution to be used to initialize (uniform, xavier)')
    parser.add_argument('--init_range', type=float, default=0.05,
                        help='initialize parameters using uniform distribution between -uniform and uniform.')
    parser.add_argument('--init_embedding', action='store_true',
                        help='initialize word embeddings with read vectors from word2vec')
    parser.add_argument('--seed_feeding', action='store_true',
                        help='feed seed embedding at the first step of decoding')
    parser.add_argument('--use_eg', action='store_true',
                        help='whether to use example information')
    parser.add_argument('--simple_eg', action='store_true',
                        help='whether to use rule-based simple example, for analysis')
    parser.add_argument('--use_wordvec', action='store_true',
                        help='whether to use wordvec information')
    parser.add_argument('--use_morpheme', action='store_true',
                        help='whether to use morpheme information')
    parser.add_argument('--use_formation', action='store_true',
                        help='whether to use formation information')
    parser.add_argument('--use_sememe', action='store_true',
                        help='whether to use sememe information')
    parser.add_argument('--teacher_ratio', type=float, default=1.0,
                        help='teacher forcing ratio')
    parser.add_argument('--char', action='store_true',
                        help='character embedding')
    parser.add_argument('--fix_embedding', action='store_true',
                        help='fix initialized word embeddings')
    parser.add_argument('--dhid', type=int, default=300,
                        help='dimension of hidden states')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help='factor by which learning rate is decayed (lr = lr * factor)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=500,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--emb_dropout', type=float, default=0.2,
                        help='dropout applied to embedding layer (0 = no dropout)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str,  default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--opt', type=str, default='adam',
                        help='optimizer (adam, sgd)')
    parser.add_argument('--sentence_bleu', type=str, default='./sentence-bleu',
                        help='Compiled binary file of sentece-bleu.cpp')
    parser.add_argument('--valid_all', action='store_true',
                        help='Run validation with all data (only for debugging)')
    args = parser.parse_args()

    print(args)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    ###############################################################################
    # Load data
    ###############################################################################

    corpus = data.MorphemeCorpus(args)

    if args.data_usage < 100.0:
        corpus.sample_train_data(args.data_usage)
    translator = translate.Translator(corpus, sentence_bleu=args.sentence_bleu, valid_all=args.valid_all)
    eval_batch_size = 10

    ###############################################################################
    # Build the model
    ###############################################################################
    vocab_size = len(corpus.id2word)
    model = model.MORPHEME_EXT(args, corpus)

    if args.init_embedding == True:
        model.init_embedding(corpus, fix_embedding=args.fix_embedding)

    if args.cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss(ignore_index=corpus.word2id['<pad>'])  # 默认 ignore -100 这个 index，对应 seed 位置生成的词; 改成 <pad> 作为 ign_idx 了

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # Loop over epochs.
    lr = args.lr
    best_val_loss = 99999999
    best_bleu = -1
    no_improvement = 0

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train()
            [val_loss] = translator.eval_loss(model, mode='valid', max_batch_size=args.batch_size, cuda=args.cuda)
            hyp = translator.greedy(model, mode="valid", max_batch_size=args.batch_size, cuda=args.cuda, max_len=corpus.max_len)
            val_bleu_corpus = translator.bleu(hyp, mode="valid", nltk='corpus')
            val_bleu_sentence = translator.bleu(hyp, mode="valid", nltk='sentence')
            
            # debug: evaluate training set
            hyp_train = translator.greedy(model, mode="train", max_batch_size=args.batch_size, cuda=args.cuda, max_len=corpus.max_len)
            val_bleu_corpus_train = translator.bleu(hyp_train, mode="train", nltk='corpus')
            val_bleu_sentence_train = translator.bleu(hyp_train, mode="train", nltk='sentence')
            print('train_bleu(corpus/sentence): ({:5.2f}/{:5.2f})'.format(val_bleu_corpus_train * 100, val_bleu_sentence_train * 100))

            if val_loss < best_val_loss:
                save_checkpoint()
                best_val_loss = val_loss
                best_bleu = val_bleu_sentence  # we are interested in the best bleu after ppl stop decreasing
                no_improvement = 0
            elif val_bleu_sentence > best_bleu:
                save_checkpoint()
                best_bleu = val_bleu_sentence
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == 6:
                    load_checkpoint()
                    lr *= args.lr_decay
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= args.lr_decay
                if no_improvement == 12:
                    break

            print('-' * 112)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | ppl {:8.2f} | BLEU(C/S) {:5.2f} /{:5.2f} | not improved: {:d}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_bleu_corpus * 100, val_bleu_sentence * 100, no_improvement), 
                    flush=True
            )
            print('-' * 112)
    except KeyboardInterrupt:
        print('-' * 112)
        print('Exiting from training early')

    # Load the best saved model.
    load_checkpoint()

    # Run on test data.
    hyp = translator.greedy(model, mode="test", max_batch_size=args.batch_size, cuda=args.cuda, max_len=corpus.max_len)
    print('-' * 112)
    print('Decoded:')
    for (word, desc) in hyp:
        print(word, end='\t')
        new_def = []
        for w in desc:
            if w not in corpus.id2word:
                new_def.append('[' + w + ']')
            else:
                new_def.append(w)
        print(' '.join(new_def), flush=True)

    test_loss = translator.eval_loss(model, mode='test', max_batch_size=args.batch_size, cuda=args.cuda)
    test_bleu_corpus = translator.bleu(hyp, mode="test", nltk='corpus')
    test_bleu_sentence = translator.bleu(hyp, mode="test", nltk='sentence')
    print('=' * 112)
    print('| End of training | test BLEU (corpus.nltk / sent.nltk): {:5.2f}/{:5.2f}'.format(test_bleu_corpus * 100, test_bleu_sentence * 100))
    print('| End of training | best_valid BLEU (sent.nltk): {:5.2f}'.format(best_bleu * 100))
    print('=' * 112)
