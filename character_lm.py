from lstm import *
from utils import *

from keras.layers import BatchNormalization, Concatenate, Conv2D, GRU, MaxPooling2D, Reshape
from keras.callbacks import ModelCheckpoint

class CharLanguageModel(BpeLM):
    def __init__(self, args, vocab, char_vocab, train_file, valid_file, model_name=''):
        self.max_word_length = 24
        self.char_vocab = char_vocab
        self.batch_norm = args.batch_norm
        self.feature_maps = args.feature_maps
        self.kernels = args.kernels
        self.optimizer = args.optimizer
        super(CharLanguageModel, self).__init__(args, vocab, train_file, valid_file, model_name='')
        self.__model_param_checks()
    
    def __model_param_checks(self):
        assert len(self.feature_maps) == len(self.kernels), "Number of feature maps must include number of kernel values!"
        return

    def cnn_block(self, max_seq_length, max_word_length, feature_maps, kernels, x_input):
        concat_input = []
        for idx, (feature_map, kernel) in enumerate(zip(feature_maps, kernels)):
            reduced_length = max_word_length - kernel + 1
            conv = Conv2D(filters=feature_map, kernel_size=(1, kernel), activation='tanh', data_format='channels_last', name='conv_{}'.format(idx))(x_input)
            pooling = MaxPooling2D(pool_size=(1, reduced_length), data_format='channels_last', name='pooling_{}'.format(idx))(conv)
            concat_input.append(pooling)

        x = Concatenate()(concat_input)
        x = Reshape((max_seq_length + 1, sum(feature_maps)))(x)
        return x

    def build_graph(self):
        y_ph = tf.placeholder(tf.int32, shape=(None, None), name='y_ph')
        chars_input = Input(batch_shape=(None, None, self.max_word_length), name='char_input')
        char_embedding = Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dim, name='char_embeddings')(chars_input)
        cnn = self.cnn_block(max_seq_length=self.seq_len, max_word_length=self.max_word_length, feature_maps=self.feature_maps, kernels=self.kernels, x_input=char_embedding)
        if self.batch_norm:
            x = BatchNormalization()(cnn)
        else:
            x = cnn
        for recurrent_layer in range(self.num_hidden_layers):
            x = GRU(units=self.recurrent_dim, activation='tanh', recurrent_activation='sigmoid', return_sequences=True, name='gru_{}'.format(recurrent_layer))(x)
        output = TimeDistributed(Dense(units=self.vocab_size, activation='linear'), name='output_logits')(x)
        model = Model(inputs=chars_input, outputs=output)
        model.compile(loss=self.sparse_loss, optimizer=self.optimizer, target_tensors=[y_ph])
        model.summary()
        self.model = model

    def train_generator(self):
        np.random.seed(7)
        n_train_iters = self.num_train_examples // self.batch_size
        n_valid_iters = self.num_val_examples // self.batch_size

        train_data = CharacterLMData(data_file=self.train_file, vocab=self.vocab, char_vocab=self.char_vocab, max_seq_len=self.seq_len, batch_size=self.batch_size)
        valid_data = CharacterLMData(data_file=self.valid_file, vocab=self.vocab, char_vocab=self.char_vocab, max_seq_len=self.seq_len, batch_size=self.batch_size)

        train_datagen = train_data.generate_batches()
        valid_datagen = valid_data.generate_batches()

        ckpt_fname = 'char_lm_{epoch:02d}-{val_loss:.2f}.h5'
        ckpt = ModelCheckpoint(ckpt_fname, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        self.model.fit_generator(generator=train_datagen, steps_per_epoch=n_train_iters,
                                epochs=self.epochs, validation_data=valid_datagen, validation_steps=n_valid_iters,
                                callbacks=[ckpt])

# Test if model compiles
if __name__ == '__main__':
    import string

    class Args:
        batch_norm = True
        feature_maps = [200, 100, 100]
        kernels = [3, 2, 2]
        optimizer = 'adam'
        num_train_examples = 20000
        num_val_examples = 20000
        print_freq = 10000
        seq_len = 35
        cell_type = 'lstm'
        num_hidden_layers = 2
        embedding_dim= 128
        embedding_dropout_rate = 0.2
        recurrent_dim = 128
        optimizer = 'adam'
        batch_size = 128
        epochs = 10
        learning_rate = 0.01

    args = Args()
    fake_vocab_words = ['dog', 'cat', 'storm', 'cloud', 'atom', 'grass']
    vocab = dict(zip(fake_vocab_words, list(range(len(fake_vocab_words)))))
    fake_chars = string.printable
    char_vocab = dict(zip(fake_chars, list(range(len(fake_chars)))))
    train_file = 'train.txt'
    valid_file = 'valid.txt'
    model_obj = CharLanguageModel(args=args, vocab=vocab, char_vocab=char_vocab, train_file=train_file, valid_file=valid_file)
