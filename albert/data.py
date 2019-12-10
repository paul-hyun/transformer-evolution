import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections
from tqdm import tqdm, trange
import json
import random
from random import randrange, randint, shuffle, choice
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from vocab import load_vocab


SPAN_LEN = 3
SPAN_VALUE = np.array([i+1 for i in range(SPAN_LEN)])
SPAN_RATIO = np.array([1/i for i in SPAN_VALUE])
SPAN_RATIO = SPAN_RATIO / np.sum(SPAN_RATIO)


""" SPAN 길이 """
def get_span_length():
    return random.choices(SPAN_VALUE, SPAN_RATIO)[0]


""" 마스크 생성 """
def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    cand_idx = {}
    index = 0
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[index].append(i)
        else:
            index += 1
            cand_idx[index] = [i]
    keys = list(cand_idx.keys())
    shuffle(keys)

    mask_lms = []
    covered_idx = set()
    for index in keys:
        if len(mask_lms) >= mask_cnt:
            break
        span_len = get_span_length()
        if len(cand_idx) <= index + span_len:
            continue
        index_set = []
        for i in range(span_len):
            index_set.extend(cand_idx[index + i])
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        is_idx_covered = False
        for index in index_set:
            if index in covered_idx:
                is_idx_covered = True
                break
        if is_idx_covered:
            continue
        
        for index in index_set:
            covered_idx.add(index)
            masked_token = None
            if random.random() < 0.8: # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                if random.random() < 0.5: # 10% keep original
                    masked_token = tokens[index]
                else: # 10% random word
                    masked_token = choice(vocab_list)
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


""" 쵀대 길이 초과하는 토큰 자르기 """
def trim_tokens(tokens_a, tokens_b, max_seq):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break

        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()


""" pretrain 데이터 생성 """
def create_pretrain_instances(datas, doc_idx, doc, n_seq, mask_prob, vocab_list):
    # for CLS], [SEP], [SEP]
    max_seq = n_seq - 3
    tgt_seq = max_seq
    
    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(doc)):
        current_chunk.append(doc[i]) # line
        current_length += len(doc[i])
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 1 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])
                
                trim_tokens(tokens_a, tokens_b, max_seq)
                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_next = 0
                    # switch sentence order
                    tokens_tmp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = tokens_tmp
                else:
                    is_next = 1

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next,
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


""" pretrain 데이터 생성 """
def make_pretrain_data(args):
    vocab = load_vocab(args.vocab)
    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    line_cnt = 0
    with open(args.input, "r") as in_f:
        for line in in_f:
            line_cnt += 1
    
    datas = []
    with open(args.input, "r") as f:
        for i, line in enumerate(tqdm(f, total=line_cnt, desc="Loading Dataset", unit=" lines")):
            data = json.loads(line)
            if 0 < len(data["doc"]):
                datas.append(data)

    with open(args.output, "w") as out_f:
        for i, data in enumerate(tqdm(datas, desc="Make Pretrain Dataset", unit=" lines")):
            instances = create_pretrain_instances(datas, i, data["doc"], args.n_seq, args.mask_prob, vocab_list)
            for instance in instances:
                out_f.write(json.dumps(instance))
                out_f.write("\n")


""" pretrain 데이터셋 """
class PretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc="Make Pretrain Dataset", unit=" lines")):
                instance = json.loads(line)
                self.labels_cls.append(instance["is_next"])
                sentences = [vocab.piece_to_id(p) for p in instance["tokens"]]
                self.sentences.append(sentences)
                self.segments.append(instance["segment"])
                mask_idx = np.array(instance["mask_idx"], dtype=np.int)
                mask_label = np.array([vocab.piece_to_id(p) for p in instance["mask_label"]], dtype=np.int)
                label_lm = np.full(len(sentences), dtype=np.int, fill_value=-1)
                label_lm[mask_idx] = mask_label
                self.labels_lm.append(label_lm)
    
    def __len__(self):
        assert len(self.labels_cls) == len(self.labels_lm)
        assert len(self.labels_cls) == len(self.sentences)
        assert len(self.labels_cls) == len(self.segments)
        return len(self.labels_cls)
    
    def __getitem__(self, item):
        return (torch.tensor(self.labels_cls[item]),
                torch.tensor(self.labels_lm[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    labels_cls, labels_lm, inputs, segments = list(zip(*inputs))

    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels_cls, dim=0),
        labels_lm,
        inputs,
        segments
    ]
    return batch


""" pretraun 데이터 로더 """
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
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc="Loading Dataset", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                sentence = [vocab.piece_to_id("[CLS]")] + [vocab.piece_to_id(p) for p in data["doc"]] + [vocab.piece_to_id("[SEP]")]
                self.sentences.append(sentence)
                self.segments.append([0] * len(sentence))
    
    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        assert len(self.labels) == len(self.segments)
        return len(self.labels)
    
    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


""" movie data collate_fn """
def movie_collate_fn(inputs):
    labels, inputs, segments = list(zip(*inputs))

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        inputs,
        segments,
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
    parser.add_argument("--input", default="../data/kowiki.json", type=str, required=False,
                        help="input json file")
    parser.add_argument("--output", default="../data/kowiki_albert.json", type=str, required=False,
                        help="output json file")
    parser.add_argument("--n_seq", default=512, type=int, required=False,
                        help="sequence length")
    parser.add_argument("--vocab", default="../kowiki.model", type=str, required=False,
                        help="vocab file")
    parser.add_argument("--mask_prob", default=0.15, type=float, required=False,
                        help="probility of mask")
    args = parser.parse_args()

    if not os.path.isfile(args.output):
        make_pretrain_data(args)
    else:
        print(f"{args.output} exists")

