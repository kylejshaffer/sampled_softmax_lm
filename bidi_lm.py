import argparse
import math
import numpy as np
import os

import sys
import tensorflow as tf
import utils

import keras
from keras.layers import Layer
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Activation, Input, Dropout, Embedding, GRU, LSTM, Dense, Lambda
from keras.layers.merge import Concatenate
from keras.layers.wrappers import Bidirectional, TimeDistributed


class BidirectionalLM(object):
    def __init__(self, args, vocab, train_file, valid_file, model_name):
        self.vocab = vocab
        self.train_file = train_file
        self.valid_file = valid_file
        self.model_name = model_name
        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)
        self.num_train_examples = args.num_train_examples
        self.num_val_examples = args.num_val_examples
        self.eval_thresh = args.print_freq
        self.batch_size = args.batch_size
        self.valid_batch_size = 256
        self.epochs = args.epochs
        self.seq_len = args.seq_len
        self.vocab_size = len(self.vocab)
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_hidden_layers = args.num_hidden_layers
        self.num_sampled = args.num_sampled
        self.embedding_dropout = args.embedding_dropout_rate
        self.learning_rate = args.learning_rate
        self.opt_string = args.optimizer
        # self.visualize_gradients = args.visualize_gradients
        self._choose_optimizer()
        self.build_graph()
        # self.compile()
        # self._tboard_setup()
        # self.saver = tf.train.Saver()

    def _init_layers(self):
        # Sequence placeholders
        # self.input_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.input_seq = Input(shape=(None,), name='x_input')
        self.output_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_output')

        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                        embeddings_initializer='glorot_uniform', name='embedding')
        self.embedding_dropout_layer = Dropout(self.embedding_dropout, name='embedding_dropout')
        self.fwd_layers = [LSTM(units=self.hidden_dim, return_sequences=True, go_backwards=False) for _ in range(self.num_hidden_layers)]
        self.bwd_layers = [LSTM(units=self.hidden_dim, return_sequences=True, go_backwards=True) for _ in range(self.num_hidden_layers)]
        self.proj_layer = Dense(units=512, activation='relu', name='down_projection')
        self.logits_layer = Dense(units=self.vocab_size, activation='linear', name='logits')

    def sparse_loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    def tf_s2s_loss(self, logits, targets, weights, average_across_timesteps=True, average_across_batch=True):
        return tf.contrib.seq2seq.sequence_loss(logits, targets, weights, average_across_timesteps, average_across_batch)

    def build_graph(self):
        self._init_layers()
        embedded = self.embedding_layer(in_layer)
        embedded = self.embedding_dropout_layer(embedded)
        for layer_idx in list(range(len(self.fwd_layers)))[:-1]:
            fw_layer = self.fwd_layers[layer_idx]
            bw_layer = self.bwd_layers[layer_idx]
            if layer_idx == 0:
                fw_encoded = fw_layer(embedded)
                bw_encoded = bw_layer(embedded)
            else:
                fw_encoded = fw_layer(fw_bw_context)
                bw_encoded = bw_layer(fw_bw_context)
            # fw_context = Lambda(lambda x: x[:, :-2, :])(fw_encoded)
            # bw_context = Lambda(lambda x: x[:, 2:, :])(bw_encoded)
            fw_bw_context = Concatenate(axis=-1)([fw_encoded, bw_encoded])

        final_fw_layer = self.fwd_layers[-1]
        final_bw_layer = self.bwd_layers[-1]
        final_fw = final_fw_layer(fw_bw_context)
        final_bw = final_bw_layer(fw_bw_context)
        final_fw_context = Lambda(lambda x: x[:, :-2, :])(final_fw)
        final_bw_context = Lambda(lambda x: x[:, 2:, :])(final_bw)
        final_encoded_context = Concatenate(axis=-1)([final_fw_context, final_bw_context])

        encoded_projection = self.proj_layer(final_encoded_context)
        logits = self.logits_layer(encoded_projection)

        self.W, self.b = self.logits_layer.weights[0], self.logits_layer.weights[-1]
        self.model = Model(inputs=self.input_seq, outputs=logits)
        self.model.compile(loss=self.sparse_loss, optimizer=self.opt_string, target_tensors=[self.output_seq])
        self.model.summary()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adadelta', 'adam', 'sgd', 'momentum', 'rmsprop'}, 'Please select valid optimizer!'

        learning_rate = self.learning_rate

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.opt_string == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer()
        elif self.opt_string == 'adam':
            self.optimizer = keras.optimizers.Adam()
        elif self.opt_string == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def compile(self):
        self.sess.run(tf.global_variables_initializer())

    def train(self):
        np.random.seed(7)

        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.batch_size

        train_data = utils.LanguageModelData(data_file=self.train_file, vocab=self.vocab,
                                             max_seq_len=self.seq_len, batch_size=self.batch_size)
        valid_data = utils.LanguageModelData(data_file=self.valid_file, vocab=self.vocab,
                                             max_seq_len=self.seq_len, batch_size=self.valid_batch_size)

        train_datagen = train_data.generate_batches(mask=True)
        valid_datagen = valid_data.generate_batches(mask=True)

        ckpt_fname = 'bidi_lm_{epoch:02d}-{val_loss:.2f}.h5'
        ckpt = ModelCheckpoint(ckpt_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters,
                                epochs=self.epochs, validation_data=valid_datagen, validation_steps=n_valid_iters,
                                callbacks=[ckpt])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=10000000)
    parser.add_argument('--num_val_examples', type=int, required=False, default=60000)
    parser.add_argument('--eval_thresh', type=int, required=False, default=1836000)
    parser.add_argument('--batch_size', type=int, required=False, default=256)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--seq_len', type=int, required=False, default=30)
    parser.add_argument('--vocab_size', type=int, required=False, default=200002)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--hidden_dim', type=int, required=False, default=512)
    parser.add_argument('--kernel_width', type=int, required=False, default=3)
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=3)
    parser.add_argument('--optimizer', type=str, required=False, default='adagrad')
    parser.add_argument('--num_sampled', type=int, required=False, default=5000)
    parser.add_argument('--embedding_dropout', type=float, required=False, default=0.2)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--visualize_gradients', type=bool, required=False, default=False)

    args = parser.parse_args()

    lm = BidirectionalLM(args)
    print('Model instantiated')
    # print('Training...\n\n')
