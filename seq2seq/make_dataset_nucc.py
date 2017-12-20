from os import path
from glob import glob
import mojimoji
from tqdm import tqdm
import MeCab
import re
import unicodedata
from xml.sax.saxutils import unescape
import subprocess

mecab_tagger_option = '-Owakati -d '
mecab_tagger_option += subprocess.check_output(['mecab-config', '--dicdir']).decode().strip()
mecab_tagger_option += '/mecab-ipadic-neologd'

emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

def normalzie(text):
    # unicode正規化
    text = unicodedata.normalize('NFKC', text)
    # emoji 除去
    text = emoji_pattern.sub('', text)
    # 改行タブ文字除去
    text = text.replace('\n', ' ').replace('\t', ' ')
    # htmlエスケープ文字の修正
    return unescape(text)

if __name__ == '__main__':
    # Mecabの準備
    tagger = MeCab.Tagger(mecab_tagger_option)
    # ファイル読み込み
    fname_list = sorted(glob('dataset/nucc/data*.txt'))
    sequence_pairs = []
    # 各ファイルに対して
    for fname in tqdm(fname_list):
        with open(fname, 'r') as f:
            last_line = None

            for line in f:
                # 全角文字を半角文字に変換,かなは全角のまま
                line = mojimoji.zen_to_han(line, kana=False)

                if line[0] == '@':
                    # メタデータ行は無視
                    continue
                elif line[0] == 'F' or line[0] == 'M':
                    # セリフ開始行Fは女,Mは男
                    if last_line is None:
                        last_line = line
                        continue
                    else:
                        seq_input = normalzie(last_line[5:])
                        seq_output = normalzie(line[5:])
                        last_line = line
                        sequence_pairs.append((seq_input, seq_output))
                else:
                    last_line = None

    print("Num of conv", len(sequence_pairs))
    # 語彙リスト
    vocab = []

    f_in = open('dataset/input_sequence_nucc.txt' , 'w')
    f_out = open('dataset/output_sequence_nucc.txt' , 'w')
    for seq_in, seq_out in tqdm(sequence_pairs):
        # 入力側
        seq_in = tagger.parse(seq_in)
        f_in.write(seq_in)
        seq_in = seq_in.split(' ')
        # 語彙 追加
        for word in seq_in:
            if not word in vocab:
                vocab.append(word)
        # 出力側
        seq_out = tagger.parse(seq_out)
        f_out.write(seq_out)
        seq_out =  seq_out.split(' ')
        # 語彙 追加
        for word in seq_out:
            if not word in vocab:
                vocab.append(word)

    f_in.close()
    f_out.close()

    print(len(vocab))
    with open('dataset/vocab_nucc.txt', 'w') as f:
        for w in vocab:
            f.write(w + '\n')
