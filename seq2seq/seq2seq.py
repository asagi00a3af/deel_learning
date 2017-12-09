import argparse
from tqdm import tqdm
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

UNK = 0
EOS = 1

def sequence_embed(embed, xs):
    # embedにまとめて入れるために区切りを保存する
    x_len = [len(x) for x in xs]
    x_section = np.cumsum(x_len[:-1])
    # まとめて入力するためにconcat
    ex = embed(F.concat(xs, axis=0))
    # データを再分割
    exs = F.split_axis(ex, x_section, 0)
    return exs

class Seq2seq(chainer.Chain):
    """
    seq2seqモデル
    """
    def __init__(self, n_layers, n_vocab, n_units):
        super(Seq2seq, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.decoder = L.NStepLSTM(n_layers, n_units, n_units, 0.1)
            self.W = L.Linear(n_units, n_vocab)

        self.n_layers = n_layers
        self.n_units = n_units

    def __call__(self, xs, ys):
        # xsを逆順に
        xs = [x[::-1] for x in xs]
        # ysにeosを挟む
        eos = self.xp.array([EOS], 'i')
        ys_in = [F.concat([eos, y], axis=0) for y in ys]
        ys_out = [F.concat([y, eos], axis=0) for y in ys]

        #
        exs = sequence_embed(self.embed, xs)
        eys = sequence_embed(self.embed, ys_in)

        batch = len(xs)

        hx, cx, _ = self.encoder(None, None, exs)
        _, _, os = self.decoder(hx, cx, eys)

        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(self.W(concat_os), concat_ys_out, reduce='no')) / batch
        chainer.report({'loss': loss.data}, self)
        return loss

def load_vocab(vocab_path):
    """
    語彙idを返す関数
    '***' は<UNK>と統一で良さげ?
    """
    with open(vocab_path, 'r') as f:
        word_ids = {line.strip() : i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = UNK
    word_ids['<EOS>'] = EOS
    return word_ids

def load_data(vocab, seq_in, seq_out):
    """
    データセットを返す関数
    """
    x_data = []
    y_data = []
    with open(seq_in, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            x_data.append(np.array([vocab.get(w, UNK) for w in words], 'i'))
    with open(seq_out, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            y_data.append(np.array([vocab.get(w, UNK) for w in words], 'i'))

    if len(x_data) != len(y_data):
        raise ValueError('len(x_data) != len(y_data)')

    data = [(x, y) for x, y in zip(x_data, y_data)]

    return data

def main():
    """
    main関数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', '-v', type=str, default='dataset/vocab.txt')
    parser.add_argument('--seq_in', '-i', type=str, default='dataset/input_sequence.txt')
    parser.add_argument('--seq_out', '-o', type=str, default='dataset/output_sequence.txt')
    args = parser.parse_args()

    vocab_ids = load_vocab(args.vocab)
    train_data = load_data(vocab_ids, args.seq_in, args.seq_out)



if __name__ == '__main__':
    main()
