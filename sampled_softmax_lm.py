import argparse
import math
import numpy as np
import os

import sys
import tensorflow as tf
import utils

from utils import *
from keras.callbacks import ModelCheckpoint
from gated_conv import GCNN

import keras
from keras.layers import Layer
from keras import backend as K
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Activation, Input, Dropout, Embedding, GRU, LSTM, Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed


class SSLanguageModel(object):
    def __init__(self, args):
        self.config = tf.ConfigProto(allow_soft_placement=True)
        # self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        K.set_session(self.sess)
        self.num_train_examples = args.num_train_examples
        self.num_val_examples = args.num_val_examples
        self.eval_thresh = args.eval_thresh
        self.batch_size = args.batch_size
        self.val_batch_size = 256
        self.epochs = args.epochs
        self.seq_len = args.seq_len
        self.vocab_size = args.vocab_size
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.kernel_width = args.kernel_width
        self.num_hidden_layers = args.num_hidden_layers
        self.num_sampled = args.num_sampled
        self.embedding_dropout = args.embedding_dropout
        self.learning_rate = args.learning_rate
        self.opt_string = args.optimizer
        self.visualize_gradients = args.visualize_gradients

        # Sequence placeholders
        self.input_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_input')
        self.output_seq = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_output')

        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim,
                                        embeddings_initializer='glorot_uniform', name='embedding')
        self.embedding_dropout_layer = Dropout(self.embedding_dropout, name='embedding_dropout')

        self._choose_optimizer()
        self.build_graph()
        self.compile()
        # self._tboard_setup()
        self.saver = tf.train.Saver()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adadelta', 'adam', 'sgd', 'momentum', 'rmsprop'}, 'Please select valid optimizer!'

        learning_rate = self.learning_rate

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.opt_string == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer()
        elif self.opt_string == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif self.opt_string == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def _tboard_setup(self):
        self.loss_summary = tf.summary.scalar('loss', self.train_loss)
        self.summary_writer = tf.summary.FileWriter('logs/', graph=tf.get_default_graph())
        # Setup for visualizing gradients
        if self.visualize_gradients:
            for grad, var in self.grads:
                tf.summary.histogram(var.name + '/gradient', grad)

        self.merged_summary_op = tf.summary.merge_all()

    def build_graph(self):
        embedded = self.embedding_layer(self.input_seq)
        embedded = self.embedding_dropout_layer(embedded)
        
        # Add convolutional layers
        for i in range(self.num_hidden_layers - 1):
            embedded = GCNN(self.hidden_dim, window_size=self.kernel_width, name='gcnn_{}'.format(i+1))(embedded)
        self.encoded = GCNN(self.hidden_dim, window_size=self.kernel_width, name='gcnn_{}'.format(self.num_hidden_layers))(embedded)
        self.logits_layer = Dense(units=self.vocab_size, name='logits')
        self.logits_out = self.logits_layer(self.encoded)
        self.W, self.b = self.logits_layer.weights[0], self.logits_layer.weights[-1]

        # Reshaped input for sampled-softmax
        inputs_reshaped = tf.reshape(self.encoded, [-1, int(self.encoded.get_shape()[2])])
        weights_reshaped = tf.transpose(self.W)
        labels_reshaped = tf.reshape(self.output_seq, [-1, 1])

        # Set up train loss
        self.train_step_loss = tf.nn.sampled_softmax_loss(weights=weights_reshaped, biases=self.b, inputs=inputs_reshaped,
                                               labels=labels_reshaped, num_sampled=self.num_sampled,
                                               num_classes=self.vocab_size)
        # Set up eval loss
        self.valid_step_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_seq,
                                                                              logits=self.logits_out)
        
        self.train_loss = tf.reduce_mean(self.train_step_loss)
        self.valid_loss = tf.reduce_mean(self.valid_step_loss)

        # Set up trainable optimizer
        if not self.visualize_gradients:
            self.train_op = self.optimizer.minimize(self.train_loss)
        else:
            print('\nSetting up for gradient visualization using TensorBoard\n')
            grads = tf.gradients(self.train_loss, tf.trainable_variables())
            self.grads = list(zip(grads, tf.trainable_variables()))
            self.train_op = self.optimizer.apply_gradients(grads_and_vars=self.grads)

    def compile(self):
        self.sess.run(tf.global_variables_initializer())

    def _train_on_batch(self, x_batch, y_batch):
        _, loss_ = self.sess.run([self.train_op, self.train_loss],
                                  feed_dict={self.input_seq: x_batch, self.output_seq: y_batch})
        
        return loss_

    def _eval_on_batch(self, x, y, normalize=False):
        valid_loss_ = self.sess.run(self.valid_loss, feed_dict={self.input_seq: x,
                                                              self.output_seq: y})
        return valid_loss_

    def train(self):
        base_path = 'C:/Users/kyshaffe/Documents/neural_lm/data/1B-lm-files'
        train_file = os.path.join(base_path, 'train.txt')
        valid_file = os.path.join(base_path, 'valid.txt')
        vocab_file = 'C:/Users/kyshaffe/Documents/neural_lm/data/1B-lm-files/vocab_200k.txt'

        # Grab vocabulary
        assert os.path.exists(vocab_file)
        vocab = utils.get_bpe_vocab(vocab_file)
        self.vocab = vocab
        print('vocab length:', len(vocab))
        print('highest word id:', max(vocab.values()))
        print('lowest word id:', min(vocab.values()))

        np.random.seed(7)
        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.batch_size

        train_data = LanguageModelData(data_file=train_file, vocab=self.vocab, max_seq_len=self.seq_len, batch_size=self.batch_size)
        valid_data = LanguageModelData(data_file=train_file, vocab=self.vocab, max_seq_len=self.seq_len, batch_size=self.val_batch_size)
        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.val_batch_size

        for e in range(self.epochs):
            train_datagen = train_data.generate_batches()
            valid_datagen = valid_data.generate_batches()
            
            all_train_loss = 0
            batch_cnt = 0
            samples_cnt = 0
            for train_iter in range(n_train_iters):
                x_batch, y_batch = next(train_datagen)
                if (samples_cnt % self.eval_thresh == 0) and (batch_cnt > 0):
                    self.evaluate(valid_generator=valid_datagen)
                    self.save()
                
                loss_ = self._train_on_batch(x_batch, y_batch)

                batch_cnt += 1
                all_train_loss += loss_
                samples_cnt += len(x_batch)
                update_loss = all_train_loss / batch_cnt
                sys.stdout.write('\r num_samples_trained: {} \t|\t loss : {:8.3f} \t|\t prpl : {:8.3f}'.format((samples_cnt),
                                update_loss, np.exp(update_loss)))

                # Update TBoard visualizations
                # self.summary_writer.add_summary(summary, batch_cnt)

            # Epoch summary metrics
            print('\n\nEPOCH {} METRICS'.format(e+1))
            print('=' * 60)
            self.evaluate(valid_generator=valid_datagen, num_eval_examples=self.num_val_examples)
            print('\n\n')
            self.save(ckpt_name='model_epoch{}.ckpt'.format(e+36))

    def evaluate(self, valid_generator, num_eval_examples=10000):
        n_batch_iters = num_eval_examples // self.batch_size
        total_val_loss = 0
        val_batch_cntr = 0
        print('\n\nvalidating')
        for i in range(n_batch_iters):
            print('=', end='', flush=True)
            val_batch_cntr += 1
            x_val_batch, y_val_batch = next(valid_generator)
            valid_loss_ = self._eval_on_batch(x=x_val_batch, y=y_val_batch)
            total_val_loss += valid_loss_

        report_loss = total_val_loss / val_batch_cntr
        print('\nValidation metrics - loss: {:8.3f} | prpl: {:8.3f}\n'.format(report_loss, np.exp(report_loss)))

    def save(self, save_path='./', ckpt_name='model.ckpt'):
        self.saver.save(self.sess, save_path + ckpt_name)
        print('Model saved to file:', save_path)

    def load(self, save_path='./', ckpt_name='model.ckpt'):
        self.saver.restore(self.sess, save_path + ckpt_name)
        print('Model restored...')


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
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=4)
    parser.add_argument('--optimizer', type=str, required=False, default='adagrad')
    parser.add_argument('--num_sampled', type=int, required=False, default=5000)
    parser.add_argument('--embedding_dropout', type=float, required=False, default=0.2)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01)
    parser.add_argument('--visualize_gradients', type=bool, required=False, default=False)

    args = parser.parse_args()

    lm = SSLanguageModel(args)
    print('Model instantiated')
    # print('Training...\n\n')

    # lm.train()
    lm.load(ckpt_name='model_epoch36.ckpt')
    print('Model loaded')

    print('\nTraining...\n\n')
    lm.train()
