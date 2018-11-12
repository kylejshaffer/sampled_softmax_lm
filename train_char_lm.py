import argparse
import os

import utils
import character_lm

def train(args):
    base_path = 'C:/Users/kyshaffe/Documents/neural_lm/data/bpe/freq3-vocab25000'
    train_file = os.path.join(base_path, 'train.txt')
    valid_file = os.path.join(base_path, 'valid.txt')
    word_vocab_file = 'C:/Users/kyshaffe/Documents/NeuralRewriting/user/kyshaffe/data/bpe_dic.txt'
    char_vocab_file = os.path.join(base_path, 'char_vocab.txt')

    word_vocab = utils.get_bpe_vocab(word_vocab_file)
    char_vocab = utils.get_char_vocab(char_vocab_file)

    model_obj = character_lm.CharLanguageModel(args=args, vocab=word_vocab,
                                               char_vocab=char_vocab, train_file=train_file,
                                               valid_file=valid_file)
    model_obj.train_generator()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, required=False, default=30301028)
    parser.add_argument('--num_val_examples', type=int, required=False, default=61010)
    parser.add_argument('--print_freq', type=int, required=False, default=4194304)
    parser.add_argument('--seq_len', type=int, required=False, default=40)
    parser.add_argument('--cell_type', type=str, required=False, default='lstm')
    parser.add_argument('--num_hidden_layers', type=int, required=False, default=1)
    parser.add_argument('--embedding_dropout_rate', type=float, required=False, default=0.2)
    parser.add_argument('--embedding_dim', type=int, required=False, default=128)
    parser.add_argument('--recurrent_dim', type=int, required=False, default=256)
    parser.add_argument('--optimizer', type=str, required=False, default='adam')
    parser.add_argument('--batch_size', type=int, required=False, default=128)
    parser.add_argument('--epochs', type=int, required=False, default=10)
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001)
    parser.add_argument('--batch_norm', type=bool, required=False, default=False)
    parser.add_argument('--feature_maps', type=int, required=False, nargs="*", default=[512, 512, 256])
    parser.add_argument('--kernels', type=int, required=False, nargs="*", default=[4, 3, 2])
    parser.add_argument('--prev_model_path', type=str, required=False, default=None)
    args = parser.parse_args()

    print("kernels", args.kernels)
    print("feature maps:", args.feature_maps)

    train(args)
