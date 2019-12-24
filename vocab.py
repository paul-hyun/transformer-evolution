import sys, os, argparse, datetime, time, re, collections
import pandas as pd
import csv
import sentencepiece as spm


""" pretrain corpus 생성 """
def build_corpus(infile, outfile):
    csv.field_size_limit(sys.maxsize)
    SEPARATOR = u"\u241D"
    df = pd.read_csv(infile, sep=SEPARATOR, engine="python")

    with open(outfile, "w") as f:
        for index, row in df.iterrows():
            f.write(row["text"].lower())
            f.write("\n\n\n\n")
            print(f"build corpus ... {index + 1} / {len(df)}", end="\r")
    print()
    return outfile


""" vocab 생성 """
def build_vocab(args):
    spm.SentencePieceTrainer.train(
        f"--input={args.corpus} --model_prefix={args.prefix} --vocab_size={args.vocab_size + 7}" + 
        " --model_type=bpe" +
        " --max_sentence_length=999999" +
        " --pad_id=0 --pad_piece=[PAD]" +
        " --unk_id=1 --unk_piece=[UNK]" +
        " --bos_id=2 --bos_piece=[BOS]" +
        " --eos_id=3 --eos_piece=[EOS]" +
        " --user_defined_symbols=[SEP],[CLS],[MASK]")


""" vocab 로드 """
def load_vocab(file):
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="kowiki", type=str, required=False,
                        help="vocab prefix 입니다.")
    parser.add_argument("--vocab_size", default=8000, type=int, required=False,
                        help="생성할 vocab 수 입니다.")
    args = parser.parse_args()

    args.corpus = "data/kowiki.txt"
    if not os.path.isfile(args.corpus):
        build_corpus("data/kowiki.csv", args.corpus)
    build_vocab(args)

