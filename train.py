import argparse
import h5py
import numpy as np
import os

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

import utils
# import lstm, birnn_lm
import cnn_lm

def train(args):
    # base_path = '//fsu/Shares/NNRewriting/google_1b_data/bpe/freq3-vocab25000'
    base_path = 'C:/Users/kyshaffe/Documents/neural_lm/data/1B-lm-files'
    train_file = os.path.join(base_path, 'train.txt')
    valid_file = os.path.join(base_path, 'valid.txt')
    # vocab_file = 'C:/Users/kyshaffe/Documents/NeuralRewriting/user/kyshaffe/data/bpe_dic.txt'
    vocab_file = 'C:/Users/kyshaffe/Documents/neural_lm/data/1B-lm-files/vocab_200k.txt'

    # Grab vocabulary
    assert os.path.exists(vocab_file)
    vocab = utils.get_bpe_vocab(vocab_file)
    print('vocab length:', len(vocab))
    print('highest word id:', max(vocab.values()))
    print('lowest word id:', min(vocab.values()))
    model_name = ''

    # nn_object = lstm.BpeLM(args, vocab=vocab,train_file=train_file,
    #                         valid_file=valid_file, model_name=model_name)
    nn_object = cnn_lm.ConvLM(args, vocab=vocab, train_file=train_file,
                                valid_file=valid_file, model_name='')
    # print('model instantiated')
    # if args.prev_model_path is not None:
    #     nn_object.load_previous_model(args.prev_model_path)
    #     print('training from previous model:', args.prev_model_path)
    # nn_object.train_generator()
    nn_object.model.save('test.h5')
    print('Model Saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=30301028)
    parser.add_argument('--num_val_examples', type=int, required=False, default=61010)
    parser.add_argument('--print_freq', type=int, required=False, default=4194304)
    parser.add_argument('--seq_len', type=int, required=False, default=40)
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=1)
    parser.add_argument('--embedding_dropout_rate', type=float, required=False, default=0.2)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--window_size', type=int, required=False, default=3)
    parser.add_argument('--output_dim', type=int, required=False, default=512)
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--prev_model_path', type=str, required=False, default=None)
    args = parser.parse_args()

    train(args)
    # test(args)
