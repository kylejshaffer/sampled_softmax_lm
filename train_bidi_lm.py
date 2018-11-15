import argparse
import os
import utils
from bidi_lm import BidirectionalLM

def train(args):
    base_path = '../tweet_lm_data'
    train_file = os.path.join(base_path, 'train.txt')
    valid_file = os.path.join(base_path, 'valid.txt')
    vocab_file = os.path.join(base_path, 'tweet_vocab_thresh3.txt')

    # Grab vocabulary
    assert os.path.exists(vocab_file)
    vocab = utils.get_vocab(vocab_file)
    print('vocab length:', len(vocab))
    print('highest word id:', max(vocab.values()))
    print('lowest word id:', min(vocab.values()))
    model_name = ''

    nn_object = BidirectionalLM(args, vocab=vocab, train_file=train_file,
                                valid_file=valid_file, model_name='')
    print('model instantiated')
    nn_object.train()
    
    nn_object.model.save('bidi_lm_final.h5')
    print('Model Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=8163134)
    parser.add_argument('--num_val_examples', type=int, required=False, default=60000)
    parser.add_argument('--num_sampled', type=int, required=False, default=20000)
    parser.add_argument('--print_freq', type=int, required=False, default=816313)
    parser.add_argument('--seq_len', type=int, required=False, default=40)
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=1)
    parser.add_argument('--embedding_dropout_rate', type=float, required=False, default=0.2)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--hidden_dim', type=int, required=False, default=512)
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--prev_model_path', type=str, required=False, default=None)
    args = parser.parse_args()

    train(args)
