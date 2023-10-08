import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class TransformerEncoders(nn.Module):
    
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, withCDP=None):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, withCDP)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None, src_pos=None):
        memory = self.encoder(src, pos=src_pos, src_key_padding_mask=src_key_padding_mask)
        return memory

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, withCDP=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:

            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.global_token_mixer = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.local_token_mixer = Local_Token_Mixer(dim=d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.global_token_mixer(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.local_token_mixer(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class CIA(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = nn.Sigmoid() 
        
    def forward(self, x_pre, x_pos):
        [N, B, C] = x_pre.shape
        fea_pred = F.normalize(x_pre, dim=2)
        fea_later = F.normalize(x_pos, dim=2)
        dis = torch.bmm(fea_pred.permute(1, 0, 2), fea_later.permute(1, 2, 0))
        dis = torch.diagonal(dis, dim1=1, dim2=2).unsqueeze(-1)
        weight = self.sigmoid(dis)
        weight = 1 - weight
        out = x_pos * weight.unsqueeze(1).reshape(N, B, 1)
        return out
 

class Local_Token_Mixer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.CTI = Cross_correlation_Token_Interaction(dim=dim)
        
    def forward(self, x):
        x = self.CTI(x)
        return x


class Cross_correlation_Token_Interaction(nn.Module):

    def __init__(self, dim):
        super().__init__()
        
        self.dim = dim
        self.softmax = nn.Softmax(dim=-1) 
        self.linear = nn.Linear(dim, 2, bias=False)
        self.depthwise = Depthwise_Conv(dim)
    
    def interaction(self, x):
        [B, C, N] = x.shape
        q = k = x
        matmul = torch.bmm(q.permute(0, 2, 1), k) # transpose check
        q_abs = torch.sqrt(torch.sum(q.pow(2) + 1e-6, dim=1, keepdim=True))
        k_abs = torch.sqrt(torch.sum(k.pow(2) + 1e-6, dim=1, keepdim=True))
        abs_matmul = torch.bmm(q_abs.permute(0, 2, 1), k_abs)
        io_abs = matmul / abs_matmul

        corr_seq = torch.zeros(x.shape).cuda()
        for i in range(B):

            abs = io_abs[i].fill_diagonal_(0)
            _map=torch.argmax(abs, dim=1)

            corr_seq[i, :, :] = x[i, :, _map]
        
        fus = x + corr_seq
        fus = fus.permute(0, 2, 1)
        weight = self.linear(fus)
        weight = self.softmax(weight)
        weight = weight.permute(0, 2, 1)
        output = x * weight[:, 0:1, :] + corr_seq * weight[:, 1:2, :]

        return output

    def token_partition(self, x, local_size):

        [N, B, C] = x.shape
        x = x.view(N // local_size, local_size, B, C)
        local = x.permute(1, 0, 2, 3).reshape(local_size, -1, C).permute(1, 2, 0)
        return local

    def token_reverse(self, x, local_size, token_size):
        B = int(x.shape[0] / (token_size / local_size))
        x = x.view(B, token_size // local_size, -1, local_size)
        output = x.reshape(B, -1, token_size)
        return output     
            
    def forward(self, x):
        local_seq = self.token_partition(x, local_size=1024)
        intered_seq = self.interaction(local_seq)
        s_intered_seq = self.token_reverse(intered_seq, local_size=1024, token_size=4096)
        output = self.depthwise(s_intered_seq)
        return output


class Depthwise_Conv(nn.Module):

    def __init__(self, dim, bias=False, kernel_size=7, padding=3):
        super().__init__()
        med_channels = int(dim * 2)
        self.dwconv = nn.Conv2d(
                    dim, dim, kernel_size=kernel_size,
                    padding=padding, groups=dim, bias=bias) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = nn.GELU()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        [B, C, N] = x.shape
        x = x.view(B, C, 64, 64)
        x = self.dwconv(x)
        x = x.permute(2, 3, 0, 1).reshape(N, B, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = self.pwconv2(x)
        return x



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
