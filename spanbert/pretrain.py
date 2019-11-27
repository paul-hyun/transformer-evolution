import sys
sys.path.append("..")
import os, argparse, datetime, time, re, collections, random
from tqdm import tqdm, trange
import numpy as np
import wandb

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from vocab import load_vocab
import config as cfg
import model as span_bert
import data
import optimization as optim


""" random seed """
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" init_process_group """ 
def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


""" destroy_process_group """
def destroy_process_group():
    dist.destroy_process_group()


""" 모델 epoch 학습 """
def train_epoch(config, rank, epoch, model, criterion_lm, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            masks_idx, labels_lm, spans_idx1, spans_idx2, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments, masks_idx, spans_idx1, spans_idx2)
            logits_lm, logits_sbo = outputs[0], outputs[1]

            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss_sbo = criterion_lm(logits_sbo.view(-1, logits_sbo.size(2)), labels_lm.view(-1))
            loss = loss_lm + loss_sbo

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


""" 모델 학습 """
def train_model(rank, world_size, args):
    if 1 < args.n_gpu:
        init_process_group(rank, world_size)
    master = (world_size == 0 or rank % world_size == 0)

    vocab = load_vocab(args.vocab)

    config = cfg.Config.load(args.config)
    config.n_enc_vocab, config.n_dec_vocab = len(vocab), len(vocab)
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

    best_epoch, best_loss = 0, 0
    model = span_bert.SpanBERTPretrain(config)
    if os.path.isfile(args.save):
        model.span_bert.load(args.save)
        print(f"rank: {rank} load pretrain from: {args.save}")
    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model.to(config.device)

    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')

    train_loader, train_sampler = data.build_pretrain_loader(vocab, args, shuffle=True)

    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=t_total)

    offset = best_epoch
    for step in trange(args.epoch, desc="Epoch"):
        if train_sampler:
            train_sampler.set_epoch(step)
        epoch = step + offset

        loss = train_epoch(config, rank, epoch, model, criterion_lm, optimizer, scheduler, train_loader)

        if master:
            best_epoch, best_loss = epoch, loss
            if isinstance(model, DistributedDataParallel):
                model.module.span_bert.save(best_epoch, best_loss, args.save)
            else:
                model.span_bert.save(best_epoch, best_loss, args.save)
            print(f">>>> rank: {rank} save model to {args.save}, epoch={best_epoch}, loss={best_loss:.3f}")

    if 1 < args.n_gpu:
        destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_half.json", type=str, required=False,
                        help="config file")
    parser.add_argument("--vocab", default="../kowiki.model", type=str, required=False,
                        help="vocab file")
    parser.add_argument("--input", default="../data/kowiki_span.json", type=str, required=False,
                        help="input pretrain data file")
    parser.add_argument("--save", default="save_pretrain.pth", type=str, required=False,
                        help="save file")
    parser.add_argument("--epoch", default=5, type=int, required=False,
                        help="epoch")
    parser.add_argument("--batch", default=32, type=int, required=False,
                        help="batch")
    parser.add_argument("--gpu", default=None, type=int, required=False,
                        help="GPU id to use.")
    parser.add_argument('--seed', type=int, default=42, required=False,
                        help="random seed for initialization")
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0
    set_seed(args)

    if 1 < args.n_gpu:
        mp.spawn(train_model,
             args=(args.n_gpu, args),
             nprocs=args.n_gpu,
             join=True)
    else:
        train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)



