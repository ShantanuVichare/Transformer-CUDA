import torch
import torch.nn.functional as F
import torch.nn as nn

import self_attention


class SelfAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        mask = args[8]
        consts = args[9]
        outputs = self_attention.forward(vkq, weights, mask, consts)
        ctx.weights = weights
        ctx.consts = consts
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_out):
        outputs = self_attention.backward(
            grad_out.contiguous(), *ctx.saved_tensors, ctx.vkq, ctx.weights, ctx.consts
        )
        return tuple(outputs)


class SelfAttention(nn.Module):
    def __str__(self) -> str:
        return "SelfAttention-Extended"

    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.W_values = nn.Parameter(torch.empty(self.head_dim, self.head_dim))
        self.W_keys = nn.Parameter(torch.empty(self.head_dim, self.head_dim))
        self.W_query = nn.Parameter(torch.empty(self.head_dim, self.head_dim))
        self.W_out = nn.Parameter(torch.empty(embed_size, heads * self.head_dim))
        self.B_out = nn.Parameter(torch.empty(embed_size))

    def reset_parameters(self, init_weights=None):
        """
        Initialize weights and biases based on init_weights
        If not passed, initialize using Xavier Uniform
        Returns the initialized weights and biases for weight duplication elsewhere
        """
        if init_weights is None:
            nn.init.xavier_uniform_(self.W_values)
            nn.init.xavier_uniform_(self.W_keys)
            nn.init.xavier_uniform_(self.W_query)
            nn.init.xavier_uniform_(self.W_out)
            nn.init.zeros_(self.B_out)
        else:
            (
                w_values_init,
                w_keys_init,
                w_query_init,
                w_out_init,
                b_out_init,
            ) = init_weights
            with torch.no_grad():
                self.W_values.data.copy_(w_values_init)
                self.W_keys.data.copy_(w_keys_init)
                self.W_query.data.copy_(w_query_init)
                self.W_out.data.copy_(w_out_init)
                self.B_out.data.copy_(b_out_init)
        return [self.W_values, self.W_keys, self.W_query, self.W_out, self.B_out]

    def forward(self, values, keys, query, mask):
        # Check shapes
        assert (
            values.shape[0] == keys.shape[0] and values.shape[0] == query.shape[0]
        ), "Batch size should match for values, keys, and query"
        assert (
            values.shape[1] == keys.shape[1] and values.shape[1] == query.shape[1]
        ), "Sequence length should match for values, keys, and query"
        N, seq_len, _ = values.shape

        return SelfAttentionFunction.apply(
            values,
            keys,
            query,
            self.W_values,
            self.W_keys,
            self.W_query,
            self.W_out,
            self.B_out,
            mask,
            [N, seq_len, self.heads, self.head_dim, self.embed_size],
        )
