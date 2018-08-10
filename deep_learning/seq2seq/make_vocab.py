"""
複合的な入出力sequenceファイルからvocabファイルを生成する
"""
from os import path
import subprocess
import MeCab
from tqdm import tqdm
def main():
    mecab_tagger_option = '-Owakati -d '
    mecab_tagger_option += subprocess.check_output(['mecab-config', '--dicdir']).decode().strip()
    mecab_tagger_option += '/mecab-ipadic-neologd'

    tagger = MeCab.Tagger(mecab_tagger_option)

    vocab_dict = {}
    # 入力文字列に対して
    with open('./dataset/skype_nucc/input_sequence.txt', 'r') as f:
        for line in tqdm(f):
            text = line.strip().split(' ')
            for word in text:
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1

    # 出力文字列に対して
    with open('./dataset/skype_nucc/output_sequence.txt', 'r') as f:
        for line in tqdm(f):
            text = line.strip().split(' ')
            for word in text:
                if word in vocab_dict:
                    vocab_dict[word] += 1
                else:
                    vocab_dict[word] = 1

    with open('./dataset/skype_nucc/vocab.txt', 'w') as f1, \
         open('./dataset/skype_nucc/vocab_hist.txt', 'w') as f2, \
         open('./dataset/skype_nucc/hist.txt', 'w') as f3:
        for k, v in tqdm(sorted(vocab_dict.items(), key=lambda x:x[1],reverse=True)):
            f2.writelines([k + '\t', str(v)+ '\n'])
            f1.write(k + '\n')
            f3.write(str(v) + '\n')
if __name__  == '__main__':
    main()
