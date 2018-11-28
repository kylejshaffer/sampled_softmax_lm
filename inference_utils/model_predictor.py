import argparse
import os
import sys
sys.path.append('C:/Users/kyshaffe/Desktop/sampled_softmax_lm')
import numpy as np
import tensorflow as tf
import utils

from nltk.tokenize import word_tokenize

def load_graph(graph_path, name):
    with tf.gfile.GFile(graph_path, 'rb') as infile:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(infile.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=name)

    return graph

class Vocab(object):
    def __init__(self, vocab_file):
        self.map = utils.get_vocab(vocab_file)
        self.inv_map = {v: k for k, v in self.map.items()}
        self.bos = self.map['<s>']
        self.eos = self.map['</s>']
        self.unk = self.map['<UNK>']
        self.pad = self.map['<PAD>']

    def get_id(self, token):
        if token in self.map:
            return self.map[token]
        return self.unk

class ModelPredictor(object):
    def __init__(self, args):
        self.vocab = Vocab(args.vocab_file)
        self.graph = load_graph(args.graph_file, 'tweet_lm')
        self.x_ph = self.graph.get_tensor_by_name('tweet_lm/x_input:0')
        self.predictions_fw = self.graph.get_tensor_by_name('tweet_lm/logits/MatMul:0')
        self.predictions_bw = self.graph.get_tensor_by_name('tweet_lm/logits_1/MatMul:0')
        self.sess = tf.Session(graph=self.graph)

    def tokenize(self, input_sent, use_nltk=False):
        if use_nltk:
            toks = word_tokenize(input_sent)
        else:
            toks = input_sent.strip().split()
        tok_ids = [self.vocab.get_id(w) for w in toks]
        tok_ids.insert(0, self.vocab.bos)
        tok_ids.append(self.vocab.eos)
        return tok_ids

    def score_sent(self, sent, use_nltk=False, normalize_with_length=True):
        sent_ids = self.tokenize(sent, use_nltk=use_nltk)
        prediction_sent_ids = np.reshape(a=np.array(sent_ids),newshape=(1, len(sent_ids)))

        logprobs_fw = self.sess.run(self.predictions_fw, feed_dict={self.x_ph: prediction_sent_ids})
        logprobs_bw = np.flip(self.sess.run(self.predictions_bw, feed_dict={self.x_ph: prediction_sent_ids}), axis=0)
        print(logprobs_fw.shape)

        output_fw, output_bw = [], []
        for idx, word_id in enumerate(sent_ids):
            fw_score = logprobs_fw[idx, word_id]
            bw_score = logprobs_bw[idx, word_id]
            output_fw.append(fw_score)
            output_bw.append(bw_score)

        final_scores = [((fw_score + bw_score) / 2) for fw_score, bw_score in zip(output_fw, output_bw[::-1])]
        
        if normalize_with_length:
            sent_score = sum(final_scores) / len(final_scores)
        else:
            sent_score = sum(final_scores)
        return sent_score

    def rank_slot_entries(self, sent, word_idx, use_nltk=False, n_best=100):
        sent_ids = self.tokenize(sent, use_nltk=use_nltk)
        prediction_sent_ids = np.reshape(a=np.array(sent_ids), newshape=(1, len(sent_ids)))

        logprobs_fw = self.sess.run(self.predictions_fw, feed_dict={self.x_ph: prediction_sent_ids})
        logprobs_bw = np.flip(self.sess.run(self.predictions_bw, feed_dict={self.x_ph: prediction_sent_ids}), axis=0)

        # Add 1 to index to account for <s> tag at beginning
        word_logprobs_fw = logprobs_fw[(word_idx + 1), :]
        word_logprobs_bw = logprobs_bw[(word_idx + 1), :]

        word_slot_scores = [((fw_score + bw_score) / 2) for fw_score, bw_score in zip(word_logprobs_fw, word_logprobs_bw)]
        scored_words = sorted(list(zip(range(len(word_slot_scores)), word_slot_scores)), key=lambda x: x[-1], reverse=True)[:n_best]
        ranked_word_ids = [scored_word[0] for scored_word in scored_words]

        return ranked_word_ids

    def time_sent_predictions(self, sent, n_loops=100):
        import time

        sent_ids = self.tokenize(sent)
        sent_ids = np.reshape(a=np.array(sent_ids), newshape=(1, len(sent_ids)))

        times = []
        print('Timing...')
        for _ in range(n_loops):
            print('=', end='', flush=True)
            start = time.time()
            logprobs = self.sess.run(self.predictions_fw,
                                     feed_dict={self.x_ph: sent_ids})
            end = time.time()
            times.append(end - start)

        print()
        print('Min. inf. time:', min(times))
        print('Max. inf. time:', max(times))
        print('Avg. inf. time:', sum(times[1:]) / len(times[1:]))

    def time_batch_predictions(self, n_loops=100):
        import time

        batch_size = 200
        line_count = 0
        inf_batch = []
        with tf.gfile.GFile('data/big_sample.txt', mode='r') as infile:
            for line in infile:
                toks = self.tokenize(line.strip())
                inf_batch.append(toks)
                line_count += 1
                if line_count >= batch_size:
                    break

        max_batch_len = max([len(seq) for seq in inf_batch])
        for idx, seq in enumerate(inf_batch):
            if len(seq) < max_batch_len:
                padding_diff = max_batch_len - len(seq)
                pad = [self.vocab.pad] * padding_diff
                new_seq = seq + pad
                inf_batch[idx] = new_seq

        inf_batch = np.asarray(inf_batch)
        print(inf_batch.shape)

        times = []
        print('Timing batches...')
        for _ in range(n_loops):
            print('=', flush=True, end='')
            start = time.time()
            logprobs = self.sess.run(self.predictions_fw, feed_dict={self.x_ph: inf_batch})
            end = time.time()
            times.append(end - start)

        print('Min. inf. time:', min(times))
        print('Max. inf. time:', max(times))
        print('Avg. inf. time:', sum(times) / len(times))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_file')
    parser.add_argument('--vocab_file')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model_predictor = ModelPredictor(args)
    print('Model instantiated!')

    # Test inference
    sent1 = 'Trump told White House counsel that he wanted to order the Justice Department to prosecute his political adversaries.'
    output = model_predictor.score_sent(sent1, use_nltk=True)
    print('Log-prob of dummy sentence 1:')
    print(output)
    # print(len(output))
    # print(sum(output) / len(output))
    # print(sum(output))
    print()

    sent2 = "best site for facebook likes , twitter and youtube likes . register here : it's free . :-)"
    output2 = model_predictor.score_sent(sent2, use_nltk=False)
    print('Log-prob of dummy sentence 2:')
    print(output2)
    # print(len(output2))
    # print(sum(output2) / len(output2))
    # print(sum(output2))
    print()

    sent3 = "sweet 16 Emoji_75 gonna be awesome ♥"
    output3 = model_predictor.score_sent(sent3, use_nltk=False)
    print('Log-prob of dummy sentence 3:')
    print(output3)
    # print(len(output3))
    # print(sum(output3) / len(output3))
    # print(sum(output3))
    print()

    sent4 = "sweet 16 Emoji_75 gonna be dopeness ♥"
    output4 = model_predictor.score_sent(sent4, use_nltk=False)
    print('Log-prob of dummy sentence 4:')
    print(output4)
    print()

    sent5 = "i'm so sad my sister is going to school tomorrow . i feel like crying :("

    # Get ranked output for a slot in sentence
    word_idx = 5
    slot_word_ids = model_predictor.rank_slot_entries(sent=sent3, word_idx=word_idx, n_best=50)
    print('Best Word IDs:')
    print(slot_word_ids)
    print(sent3.split()[:word_idx])
    print()
    print('Candidate slot words:')
    slot_words = [model_predictor.vocab.inv_map[w_id] for w_id in slot_word_ids]
    print(slot_words)
