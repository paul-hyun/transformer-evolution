import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections
from tqdm import tqdm, trange
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import load_vocab


""" pretrain 데이터 생성 """
def create_pretrain_instances(doc, n_seq):
    # for [BOS], [EOS]
    max_seq = n_seq - 2
    tgt_seq = max_seq
    
    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i]) # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):
                tokens = []
                for chunk in current_chunk: tokens.extend(chunk)
                tokens = tokens[:tgt_seq]
                if 1 < len(tokens):
                    instance = {
                        "tokens": ["[BOS]"] + tokens + ["[EOS]"],
                    }
                    instances.append(instance)
            current_chunk = []
            current_length = 0
    return instances


""" pretrain 데이터 생성 """
def make_pretrain_data(args, vocab):
    line_cnt = 0
    with open(args.input, "r") as in_f:
        for line in in_f:
            line_cnt += 1

    docs = []
    with open(args.input, "r") as f:
        doc = []
        for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {args.input}", unit=" lines")):
            line = line.strip()
            if line == "":
                if 0 < len(doc):
                    docs.append(doc)
                    doc = []
            else:
                pieces = vocab.encode_as_pieces(line)
                if 0 < len(pieces):
                    doc.append(pieces)
        if doc:
            docs.append(doc)

    with open(args.output, "w") as out_f:
        for i, doc in enumerate(tqdm(docs, desc=f"Making {args.output}", unit=" lines")):
            instances = create_pretrain_instances(doc, args.n_seq)
            for instance in instances:
                out_f.write(json.dumps(instance))
                out_f.write("\n")
              

""" pretrain 데이터셋 """
class PretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                instance = json.loads(line)
                self.sentences.append([vocab.piece_to_id(p) for p in instance["tokens"]])
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, item):
        return (torch.tensor(self.sentences[item]), torch.tensor(item))


""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    dec_inputs, item = list(zip(*inputs))

    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        dec_inputs,
        torch.stack(item, dim=0),
    ]
    return batch


""" pretrain 데이터 로더 """
def build_pretrain_loader(vocab, args, shuffle=True):
    dataset = PretrainDataSet(vocab, args.input)
    if 1 < args.n_gpu and shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, collate_fn=pretrin_collate_fn)
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=shuffle, collate_fn=pretrin_collate_fn)
    return loader, sampler


""" 영화 분류 데이터셋 """
class MovieDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentences = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                self.sentences.append([vocab.piece_to_id("[BOS]")] + [vocab.piece_to_id(p) for p in data["doc"]] + [vocab.piece_to_id("[EOS]")])
    
    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        return len(self.labels)
    
    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]))


""" movie data collate_fn """
def movie_collate_fn(inputs):
    labels, dec_inputs = list(zip(*inputs))

    dec_inputs = torch.nn.utils.rnn.pad_sequence(dec_inputs, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        dec_inputs,
    ]
    return batch


""" 데이터 로더 """
def build_data_loader(vocab, infile, args, shuffle=True):
    dataset = MovieDataSet(vocab, infile)
    if 1 < args.n_gpu and shuffle:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, collate_fn=movie_collate_fn)
    else:
        sampler = None
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch, sampler=sampler, shuffle=shuffle, collate_fn=movie_collate_fn)
    return loader, sampler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/kowiki.txt", type=str, required=False,
                        help="input text file")
    parser.add_argument("--output", default="../data/kowiki_gpt.json", type=str, required=False,
                        help="output json file")
    parser.add_argument("--n_seq", default=256, type=int, required=False,
                        help="sequence length")
    args = parser.parse_args()

    if not os.path.isfile(args.output):
        vocab = load_vocab("../kowiki.model")
        make_pretrain_data(args, vocab)
    else:
        print(f"{args.output} exists")

