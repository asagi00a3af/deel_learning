import argparse
from tqdm import tqdm
import numpy as np
import subprocess
import time
import MeCab
import mojimoji
import chainer
from chainer import cuda
import chainer.functions as F
import chainer.links as L
# from chainer import training
# from chainer.training import extensions

UNK = 0
EOS = 1

mecab_tagger_option = '-Owakati -d '
mecab_tagger_option += subprocess.check_output(['mecab-config', '--dicdir']).decode().strip()
mecab_tagger_option += '/mecab-ipadic-neologd'

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

    def response(self, xs, max_length=100):
        batch = len(xs)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            xs = [x[::-1] for x in xs]
            exs = sequence_embed(self.embed, xs)
            h, c, _ = self.encoder(None, None, exs)
            ys = self.xp.full(batch, EOS, 'i')
            res = []
            for i in range(max_length):
                eys = self.embed(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                res.append(ys)

        res = cuda.to_cpu(self.xp.stack(res).T)

        outs = []
        for y in res:
            inds = np.argwhere(y == EOS)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            outs.append(y)
        return outs

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

def words2ids(txt):
    tagger = MeCab.Tagger(mecab_tagger_option)

    txt = mojimoji.zen_to_han(txt, kana=False)
    txt = tagger.parse(txt)
    txt = txt.strip().split(' ')
    return txt

class Talker():
    def __init__(self, vocab_path='vocab_skype_nucc.txt', gpu=-1, model_path='seq2seq_conversation.npz'):
        self.word_ids = load_vocab(vocab_path)
        self.ids_word = {i:w for w, i in self.word_ids.items()}

        self.model = Seq2seq(n_layers=3, n_units=256, n_vocab=len(self.word_ids))
        if gpu >= 0:
            chainer.cuda.get_device(gpu).use()
            self.model.to_gpu(gpu)
        chainer.serializers.load_npz(model_path, self.model)

        self.tagger = MeCab.Tagger(mecab_tagger_option)


    def _words2ids(self, txt):
        txt = mojimoji.zen_to_han(txt, kana=False)
        txt = self.tagger.parse(txt)
        txt = txt.strip().split(' ')
        txt = np.array([self.word_ids.get(w, UNK) for w in txt], 'i')
        data = []
        data.append(txt)
        return data

    def response(self, txt):
        data = self._words2ids(txt)
        res = self.model.response(data)[0]
        res = ''.join([self.ids_word[i] for i in res])
        return res

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', '-v', type=str, default='dataset/vocab_skype_nucc.txt')
    parser.add_argument('--model', '-m', type=str, default='dataset/seq2seq_conversation_v2_e200_l3_u256_model.npz')
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--layer', '-l', type=int, default=3)
    parser.add_argument('--unit', '-u', type=int, default=256)
    args = parser.parse_args()

    # モデル準備
    print("#Start test")
    talker = Talker(vocab_path=args.vocab, model_path=args.model)

    try:
        while True:
            txt = input('in: ')
            res = talker.response(txt)
            print("out: ", res)
    except KeyboardInterrupt:
        print("\n#End test")

if __name__ == '__main__':
    main()
