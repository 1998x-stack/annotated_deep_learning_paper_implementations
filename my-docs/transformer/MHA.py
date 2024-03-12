import math
from typing import Optional, List

import torch
from torch import nn

from labml import tracker

'''
# Prepare for multi-head attention
This module does a linear transformation and splits the vector into given number of heads for multi-head attention. 
This is used to transform key, query, and value vectors.
'''

class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, heads: int, d_k: int, bias:bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # number of heads
        self.heads = heads
        # number of dimensions in each head
        self.d_k = d_k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        input has shape:
            [seq_len, batch_size, d_model]
            or
            [batch_size, d_model]
        apply the linear transformation to the last dimension and split that into the heads
        '''
        head_shape = x.shape[-1]
        # linear transform
        x = self.linear(x)
        # split into dimensions into heads
        x = x.view(*head_shape, self.heads, self.d_k)
        # output the shape [seq_len, batch_size, heads, d_k] or [batch_size, heads, d_k]
        return x
    
# Multi-head attention Module
# This computes scaled multi-head attention for given query, key and value
'''
$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

Use the dot-product of query and key as the indicator of how matching query and key are.
Before taking the softmax the dot-product is scaled by $\sqrt{d_k}$ to prevent the dot-product from getting too large, 
because softmax is non-sensitive to large numbers and get small gradients.

notice:
softmax is calculated along the axis of the seq or time
'''
class MultiHeadAttention(nn.Module):
    '''
    - heads: number of heads
    - d_model: number of dimensions in the input vectors(query, key, value)
    '''
    
    def __init__(
            self, 
            heads: int,
            d_model: int,
            dropout_prob: float = 0.1,
            bias: bool = False
        ):
        super().__init__()
        # number of dimensions in each head
        self.d_k = d_model // heads
        self.heads = heads
        # Transform Query, Key and Value vectors for multi-headed attention
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        
        #  softmax for attention along the time axis
        self.softmax = nn.Softmax(dim=-2)
        
        # output linear layer
        self.output = nn.Linear(heads * self.d_k, d_model, bias=bias)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # scaling the factor before taking the softmax
        self.scale = 1 / (self.d_k ** 0.5)
        
        # we store the attentions so that it can used for logging
        self.attn = None
    
    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        # be overridden for other variations like relative attention
        return torch.einsum('ibhd,jbhd->ijbh', query, key)
    
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        # mask has shape [seq_len_q, seq_len_k, batch_size].
        # if the query dimension is equal to 1 it will be broadcasted to the query shape
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        # same mask applied to all heads
        mask = mask.unsqueeze(-1)
        # resulting mask has shape [seq_len_q, seq_len_k, batch_size, heads]
        return mask
    
    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        query, key and value are the tensors that store collection of query, key and value vectors, 
        They have shape [seq_len, batch_size, d_model] or [batch_size, d_model]
        mask has shape [seq_len_q, seq_len_k, batch_size]
        and mask[i, j, b] indicates whether for batch b, query at position i can attend to key at position j
        '''
        
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
            
        # 