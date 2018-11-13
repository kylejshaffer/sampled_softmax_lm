import argparse
import math
import numpy as np
import sys
import tensorflow as tf
import utils

from keras import backend as K
from keras.models import load_model
from keras.models import Model, Sequential
from keras.layers import Activation, Input, Dropout, Embedding, GRU, LSTM, Dense
from keras.layers.wrappers import Bidirectional, TimeDistributed


class BpeLM:
    def __init__(self, args, vocab, train_file, valid_file, model_name=''):
        self.train_file = train_file
        self.valid_file = valid_file
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.model_name = model_name
        self.num_train_examples = args.num_train_examples
        self.num_val_examples = args.num_val_examples
        self.print_freq = args.print_freq
        self.seq_len = args.seq_len
        self.cell_type = args.cell_type
        self.num_hidden_layers = args.num_hidden_layers
        self.embedding_dropout_rate = args.embedding_dropout_rate
        self.embedding_dim = args.embedding_dim
        self.recurrent_dim = args.recurrent_dim
        self.opt_string = args.optimizer
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        # self.session = tf.Session()
        # Explicitly register TF session with Keras
        # K.set_session(self.session)
        self._choose_optimizer()
        self._choose_recurrent_cell()
        self.init_layers()
        self.build_graph()
        self._log_params()

    def _choose_optimizer(self):
        assert self.opt_string in {'adagrad', 'adam', 'sgd', 'momentum'}, 'Please select valid optimizer!'

        learning_rate = self.learning_rate

        # Set optimizer
        if self.opt_string == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif self.opt_string == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
        elif self.opt_string == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif self.opt_string == 'momentum':
            self.optmizer = tf.train.MomentumOptimizer(learning_rate)
        else:
            'Invalid optimizer selected - exiting'
            sys.exit(1)

    def _choose_recurrent_cell(self):
        assert self.cell_type.lower() in {'gru', 'gru_gpu', 'lstm', 'lstm_gpu'}

        if self.cell_type.lower() == 'gru':
            self.recurrent_cell_fn = GRU
        elif self.cell_type.lower() == 'lstm':
            self.recurrent_cell_fn = LSTM

    def _log_params(self):
        print()
        print('Training Params:')
        print('number of training examples:', self.num_train_examples)
        print('number of validation examples:', self.num_val_examples)
        print('training with: {} / {}'.format(self.opt_string, self.optimizer))
        print('batch size:', self.batch_size)
        print('number of epochs:', self.epochs)
        print('learning rate:', self.learning_rate)
        print('number hidden layers:', self.num_hidden_layers)
        print('model filename:', self.model_name)
        print()

    def sparse_loss(self, y_true, y_pred, from_logits=True):
        return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    def init_layers(self):
        # self.x_ph = tf.placeholder(tf.int32, shape=(None, None), name='x_ph')
        self.y_ph = tf.placeholder(tf.int32, shape=(None, None), name='y_ph')
        self.embedding_dropout = Dropout(self.embedding_dropout_rate, name='embedding_dropout')
        self.embedding_layer = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, mask_zero=True, name='embedding_layer')
        if self.num_hidden_layers > 1:
            recurrent_layers = []
            for i in range(self.num_hidden_layers):
                recurrent_layers.append(self.recurrent_cell_fn(units=self.recurrent_dim, return_sequences=True, name='recurrent_layer{}'.format(i)))
        else:
            self.recurrent_layer1 = self.recurrent_cell_fn(units=self.recurrent_dim, return_sequences=True, recurrent_dropout=0.2, name='recurrent_layer1')
            self.recurrent_layer2 = self.recurrent_cell_fn(units=self.recurrent_dim, return_sequences=True, recurrent_dropout=0.2, name='recurrent_layer2')
        self.logits_layer = TimeDistributed(Dense(self.vocab_size, activation='linear', name='y_logprobs'))
        self.softmax_layer = Activation('softmax', name='y_probs')

    def build_graph(self):
        in_layer = Input(shape=(None,), name='x_ph')
        embedded = self.embedding_layer(in_layer)
        embedded = self.embedding_dropout(embedded)
        encoded = self.recurrent_layer1(embedded)
        encoded = self.recurrent_layer2(encoded)
        y_logits = self.logits_layer(encoded)
        y_probs = self.softmax_layer(y_logits)
        model = Model(inputs=in_layer, outputs=y_logits)
        model.compile(loss=self.sparse_loss, optimizer=self.opt_string, target_tensors=[self.y_ph])
        model.summary()
        self.model = model

    def train_keras(self):
        tf.set_random_seed(1)

        # self.session.run(tf.global_variables_initializer())
        sys.stdout.write('\n Global variables initialized \n')

        n_batch_iters = self.num_train_examples // self.batch_size
        batch_eval_thresh = self.print_freq // self.batch_size

        for epoch in range(self.epochs):
            # Hack to continue training from previous model
            # epoch = epoch + 1
            train_datagen = utils.generate_batches(filename=self.train_file, seqlen=self.seq_len,
                                                   vocab=self.vocab, batch_size=self.batch_size)
            print('starting training epoch {}'.format(epoch + 1))
            
            train_loss = 0
            count = 0
            batch_count = 0

            for batch_idx in range(n_batch_iters):
                x_batch, y_batch = next(train_datagen)
                train_loss_ = self.model.train_on_batch(x_batch, y_batch)
                count += len(x_batch)
                batch_count += 1
                
                train_loss += train_loss_
                update_loss = train_loss / batch_count
                update_prpl = math.exp(update_loss)
                sys.stdout.write(
                        '\rloss = {0:8.3f} - perplexity = {1:8.3f} - train iters = {2} / {3} '.format(update_loss, update_prpl,
                                                                                                      batch_count,
                                                                                                      n_batch_iters))
                if (batch_count > 0) and (batch_count % batch_eval_thresh == 0):
                    print('\n>> Update: Average train loss : {} - Average train prpl : {}\n'.format(update_loss,
                                                                                                    update_prpl))
                    valid_loss = self.validate_keras(max_count=self.print_freq // 16)
                    print('\n>>Update:  Average validation loss : {} - Average valdiation prpl : {}\n'.format(
                                valid_loss, math.exp(valid_loss)))

            # Epoch summaries
            print('current count:', count)
            train_loss /= batch_count
            valid_total_loss = self.validate_keras(max_count=self.num_val_examples)
            print()
            print('Epoch {} Summary:'.format(epoch + 1))
            print('=' * 80)
            print("(epoch {0:5}) train loss = {1:8.3f} ({2:8.3f}); valid loss = {3:8.3f} ({4:8.3f})".format(
                epoch, train_loss, math.exp(train_loss),
                valid_total_loss, math.exp(valid_total_loss)
            ))
            print('=' * 80)
            print()

            # Save epoch model here
            self.model.save('keras_lstm_lm_epoch{0}_val-prpl{1:8.3f}_v2.h5'.format(epoch+21, math.exp(valid_total_loss)))

    def validate_keras(self, max_count=-1):
        print('validating')
        valid_total_loss = 0
        count = 0
        batch_count = 0
        n_val_batch_iters = self.num_val_examples // self.batch_size

        valid_datagen = utils.generate_batches(filename=self.valid_file, seqlen=self.seq_len,
                                               vocab=self.vocab, batch_size=self.batch_size)

        for _ in range(n_val_batch_iters):
            x_val_batch, y_val_batch = next(valid_datagen)
            print('=', end='', flush=True)
            val_loss = self.model.test_on_batch(x_val_batch, y_val_batch)
            valid_total_loss = valid_total_loss + val_loss
            count += len(x_val_batch)
            batch_count += 1

            if  max_count > 0 and count > max_count:
                break

        print('.')
        valid_total_loss /= batch_count
        return valid_total_loss

    def load_previous_model(self, model_path):
        model = load_model(model_path, custom_objects={'sparse_loss': self.sparse_loss})
        last_lr = K.get_value(model.optimizer.lr)
        model.compile(loss=self.sparse_loss, optimizer='adam', target_tensors=[self.y_ph])
        if K.get_value(model.optimizer.lr) != last_lr:
            K.set_value(model.optimizer.lr, last_lr)
            print(K.get_value(model.optimizer.lr))
        print('New Model Summary')
        model.summary()
        self.model = model

        # model_skeleton = Sequential()
        # model_skeleton.add(Embedding(input_dim=model.layers[1].input_dim, output_dim=model.layers[1].output_dim,
        #                     mask_zero=model.layers[1].mask_zero))
        # model_skeleton.add(Dropout(model.layers[2].rate))
        # model_skeleton.add(LSTM(units=model.layers[3].units, return_sequences=model.layers[3].return_sequences,
        #                         recurrent_dropout=model.layers[3].recurrent_dropout))
        # model_skeleton.add(LSTM(units=model.layers[4].units, return_sequences=model.layers[4].return_sequences,
        #                         recurrent_dropout=model.layers[4].recurrent_dropout))
        # model_skeleton.add(TimeDistributed(Dense(units=self.vocab_size, activation='linear', name='y_logprobs')))

        # for layer_idx, layer in enumerate(model.layers[1:]):
        #     model_skeleton.layers[layer_idx].set_weights(layer.get_weights())

        # model_skeleton.compile(loss=self.sparse_loss, optimizer='adam', target_tensors=[self.y_ph])
        # self.model = model_skeleton
        