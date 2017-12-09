import csv
import mojimoji
from tqdm import tqdm
import MeCab

if __name__ == '__main__':
    tagger = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd')

    sequence_pairs = []
    seq_in = None
    seq_out = None
    speaker_in = None
    speaker_out = None
    sequence_pairs = []
    with open('./dataset/skype_conv_log.csv', 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if speaker_in is None:
                speaker_in, seq_in = row
                continue
            # 今の発話者が質問者と異なる場合
            if row[0] != speaker_in:
                speaker_out, seq_out = row

                # 全角を半角に変換
                si = mojimoji.zen_to_han(seq_in, kana=False)
                so = mojimoji.zen_to_han(seq_out, kana=False)
                # sequenceペアに追加
                sequence_pairs.append((si, so))

                speaker_in, seq_in = speaker_out, seq_out
            else:
                speaker_in, seq_in = row

    vocab = []
    f_in = open('dataset/input_sequence_skype_log.txt', 'w')
    f_out = open('dataset/output_sequence_skype_log.txt', 'w')
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
    with open('dataset/skype_vocab.txt', 'w') as f:
        for w in vocab:
            f.write(w + '\n')
