import torch
import torch.nn.functional as F
import torch.nn as nn

class SelfAttention(nn.Module): 
  def __str__(self) -> str:
    return "SelfAttention-Python"
  
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
    # self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    # self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    # self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.W_out = nn.Parameter(torch.empty(embed_size, heads*self.head_dim))
    self.B_out = nn.Parameter(torch.empty(embed_size))
    # self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
    # self.reset_parameters() # To be called manually
  
  def reset_parameters(self, init_weights = None):
    '''
    Initialize weights and biases based on init_weights
    If not passed, initialize using Xavier Uniform
    Returns the initialized weights and biases for weight duplication elsewhere
    '''
    
    if init_weights is None:
      nn.init.xavier_uniform_(self.W_values)
      nn.init.xavier_uniform_(self.W_keys)
      nn.init.xavier_uniform_(self.W_query)
      nn.init.xavier_uniform_(self.W_out)
      nn.init.zeros_(self.B_out)
    else:
      w_values_init, w_keys_init, w_query_init, w_out_init, b_out_init = init_weights
      with torch.no_grad():
        self.W_values.data.copy_(w_values_init)
        self.W_keys.data.copy_(w_keys_init)
        self.W_query.data.copy_(w_query_init)
        self.W_out.data.copy_(w_out_init)
        self.B_out.data.copy_(b_out_init)
    return [self.W_values, self.W_keys, self.W_query, self.W_out, self.B_out]

  def forward(self, values, keys, query, mask):
    # Do check for sizes
    assert (
      values.shape[0] == keys.shape[0] and values.shape[0] == query.shape[0]
    ), "Batch size should match for values, keys, and query"
    assert (
      values.shape[1] == keys.shape[1] and values.shape[1] == query.shape[1]
    ), "Sequence length should match for values, keys, and query"
    N, seq_len, _ = values.shape

    # Split the embedding into self.heads different pieces
    values = values.reshape(N, seq_len, self.heads, self.head_dim).transpose(1,2).reshape(N*self.heads, seq_len, self.head_dim)
    keys = keys.reshape(N, seq_len, self.heads, self.head_dim).transpose(1,2).reshape(N*self.heads, seq_len, self.head_dim)
    query = query.reshape(N, seq_len, self.heads, self.head_dim).transpose(1,2).reshape(N*self.heads, seq_len, self.head_dim)

    # Following will be of shape (N*self.heads, seq_len, head_dim)
    values = torch.matmul(values, self.W_values.T)
    keys = torch.matmul(keys, self.W_keys.T)
    query = torch.matmul(query, self.W_query.T)

    # energy will be of shape (N*self.heads, seq_len, seq_len)
    energy = torch.bmm(query, keys.transpose(-2,-1)) / (self.embed_size ** (1 / 2))

    if mask is not None:
      energy = energy.masked_fill(mask == 0, float("-1e20"))

    # attention will be of shape (N*self.heads, seq_len, seq_len)
    # out will be of shape (N*self.heads, seq_len, head_dim)
    attention = F.softmax(energy, dim=-1)
    out = torch.bmm(attention, values)

    out = out.reshape(N, self.heads, seq_len, self.head_dim).transpose(1,2).reshape(N, seq_len, self.heads*self.head_dim)
    out = F.linear(out, self.W_out, self.B_out)
    return out

