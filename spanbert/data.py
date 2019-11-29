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


SPAN_LEN = 10
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
            mask_lms.append({"index": index, "span_idx1": index_set[0] - 1, "span_idx2": index_set[-1] + 1, "label": tokens[index]})
            tokens[index] = masked_token
        # span boundary
        covered_idx.add(index_set[0] - 1)
        covered_idx.add(index_set[-1] + 1)
    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    span_idx1 = [p["span_idx1"] for p in mask_lms]
    span_idx2 = [p["span_idx2"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, span_idx1, span_idx2, mask_label


""" 쵀대 길이 초과하는 토큰 자르기 """
def trim_tokens(tokens, max_seq):
    while True:
        if len(tokens) <= max_seq:
            break

        if random.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()


""" pretrain 데이터 생성 """
def create_pretrain_instances(datas, doc_idx, doc, n_seq, mask_prob, vocab_list):
    # for CLS], [SEP]
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
                for j in range(len(current_chunk)):
                    tokens.extend(current_chunk[j])
                
                trim_tokens(tokens, max_seq)
                assert 0 < len(tokens)
                assert len(tokens) <= max_seq

                tokens = ["[CLS]"] + tokens + ["[SEP]"]
                segment = [0] * len(tokens)

                tokens, mask_idx, span_idx1, span_idx2, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 2) * mask_prob), vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "mask_idx": mask_idx,
                    "span_idx1": span_idx1,
                    "span_idx2": span_idx2,
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
        self.masks_idx = []
        self.labels_lm = []
        self.spans_idx1 = []
        self.spans_idx2 = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc="Make Pretrain Dataset", unit=" lines")):
                instance = json.loads(line)
                self.masks_idx.append(instance["mask_idx"])
                self.labels_lm.append([vocab.piece_to_id(p) for p in instance["mask_label"]])
                self.spans_idx1.append(instance["span_idx1"])
                self.spans_idx2.append(instance["span_idx2"])
                self.sentences.append([vocab.piece_to_id(p) for p in instance["tokens"]])
                self.segments.append(instance["segment"])
    
    def __len__(self):
        assert len(self.masks_idx) == len(self.labels_lm)
        assert len(self.masks_idx) == len(self.spans_idx1)
        assert len(self.masks_idx) == len(self.spans_idx2)
        assert len(self.masks_idx) == len(self.sentences)
        assert len(self.masks_idx) == len(self.segments)
        return len(self.masks_idx)
    
    def __getitem__(self, item):
        return (torch.LongTensor(self.masks_idx[item]),
                torch.LongTensor(self.labels_lm[item]),
                torch.LongTensor(self.spans_idx1[item]),
                torch.LongTensor(self.spans_idx2[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


""" pretrain data collate_fn """
def pretrin_collate_fn(inputs):
    masks_idx, labels_lm, spans_idx1, spans_idx2, inputs, segments = list(zip(*inputs))

    masks_idx = torch.nn.utils.rnn.pad_sequence(masks_idx, batch_first=True, padding_value=0)
    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    spans_idx1 = torch.nn.utils.rnn.pad_sequence(spans_idx1, batch_first=True, padding_value=0)
    spans_idx2 = torch.nn.utils.rnn.pad_sequence(spans_idx2, batch_first=True, padding_value=0)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        masks_idx,
        labels_lm,
        spans_idx1,
        spans_idx2,
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
    parser.add_argument("--output", default="../data/kowiki_span.json", type=str, required=False,
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

