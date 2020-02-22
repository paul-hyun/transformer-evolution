import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    """
    attention pad mask
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask


def get_attn_decoder_mask(seq):
    """
    attention decoder mask
    """
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1)  # upper triangular part of a matrix(2-D)
    return subsequent_mask


class ScaledDotProductAttention(nn.Module):
    """
    scale dot product attention
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
        self.num_buckets = 32
        self.relative_attention_bias = torch.nn.Embedding(self.num_buckets, self.config.n_head)

    def forward(self, Q, K, V, attn_mask, bidirectional=True):
        qlen, klen = Q.size(-2), K.size(-2)
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        # (1, n_head, n_q_seq, n_k_seq)
        position_bias = self.compute_bias(qlen, klen, bidirectional=bidirectional)
        scores += position_bias
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob

    def compute_bias(self, qlen, klen, bidirectional=True):
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        # (qlen, klen)
        relative_position = memory_position - context_position
        # (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            num_buckets=self.num_buckets,
            bidirectional=bidirectional
        )
        # (qlen, klen)
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        # (qlen, klen, n_head)
        values = self.relative_attention_bias(rp_bucket)
        # (1, n_head, qlen, klen)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def _relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret


class MultiHeadAttention(nn.Module):
    """
    multi head attention
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, Q, K, V, attn_mask, bidirectional=False):
        batch_size = Q.size(0)
        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask, bidirectional=bidirectional)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob


class PoswiseFeedForwardNet(nn.Module):
    """
    feed forward
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.active(self.conv1(inputs.transpose(1, 2)))
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_hidn)
        return output


class EncoderLayer(nn.Module):
    """
    encoder layer
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


class Encoder(nn.Module):
    """
    encoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, enc_embd, enc_self_mask):
        # (bs, n_enc_seq, d_hidn)
        enc_outputs = enc_embd

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            enc_outputs, attn_prob = layer(enc_outputs, enc_self_mask)
            attn_probs.append(attn_prob)
        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return enc_outputs, attn_probs


class DecoderLayer(nn.Module):
    """
    decoder layer
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)

    def forward(self, dec_inputs, enc_outputs, self_mask, ende_mask):
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq)
        self_att_outputs, self_attn_prob = self.self_attn(dec_inputs, dec_inputs, dec_inputs, self_mask, bidirectional=False)
        self_att_outputs = self.layer_norm1(dec_inputs + self_att_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_enc_att_outputs, dec_enc_attn_prob = self.dec_enc_attn(self_att_outputs, enc_outputs, enc_outputs, ende_mask)
        dec_enc_att_outputs = self.layer_norm2(self_att_outputs + dec_enc_att_outputs)
        # (bs, n_dec_seq, d_hidn)
        ffn_outputs = self.pos_ffn(dec_enc_att_outputs)
        ffn_outputs = self.layer_norm3(dec_enc_att_outputs + ffn_outputs)
        # (bs, n_dec_seq, d_hidn), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return ffn_outputs, self_attn_prob, dec_enc_attn_prob


class Decoder(nn.Module):
    """
    decoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

    def forward(self, dec_embd, enc_outputs, self_mask, ende_mask):
        # (bs, n_dec_seq, d_hidn)
        dec_outputs = dec_embd

        self_attn_probs, dec_enc_attn_probs = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, d_hidn), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, self_attn_prob, dec_enc_attn_prob = layer(dec_outputs, enc_outputs, self_mask, ende_mask)
            self_attn_probs.append(self_attn_prob)
            dec_enc_attn_probs.append(dec_enc_attn_prob)
        # (bs, n_dec_seq, d_hidn), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return dec_outputs, self_attn_probs, dec_enc_attn_probs


class T5(nn.Module):
    """
    t5
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.n_vocab, self.config.d_hidn)
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_vocab, bias=False)
        self.projection_lm.weight = self.embedding.weight

    def forward(self, enc_inputs, dec_inputs):
        enc_embd = self.embedding(enc_inputs)
        dec_embd = self.embedding(dec_inputs)

        enc_self_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.config.i_pad)
        dec_self_mask = self.get_attn_dec_mask(dec_inputs)
        dec_ende_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attn_probs = self.encoder(enc_embd, enc_self_mask)
        # (bs, n_dec_seq, d_hidn), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attn_probs, dec_enc_attn_probs = self.decoder(dec_embd, enc_outputs, dec_self_mask, dec_ende_mask)
        # (bs, n_dec_seq, n_vocab)
        dec_outputs = self.projection_lm(dec_outputs)
        # (bs, n_dec_seq, n_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return dec_outputs, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs

    def get_attn_dec_mask(self, dec_inputs):
        # (bs, n_dec_seq, n_dec_seq)
        dec_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_ahead_mask = get_attn_decoder_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_mask = torch.gt((dec_pad_mask + dec_ahead_mask), 0)
        # (bs, n_dec_seq, n_dec_seq)
        return dec_self_mask

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["loss"]


class T5Pretrain(nn.Module):
    """
    T5 pretrain
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.t5 = T5(self.config)

    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_dec_seq, n_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.t5(enc_inputs, dec_inputs)
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs


class MovieClassification(nn.Module):
    """
    naver movie classfication
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.t5 = T5(self.config)

    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_dec_seq, n_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs = self.t5(enc_inputs, dec_inputs)
        return logits, enc_self_attn_probs, dec_self_attn_probs, dec_enc_attn_probs
