"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) Karpathy's nanoGPT implementation:
https://github.com/karpathy/nanoGPT/blob/master/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.configuration_utils import PretrainedConfig

from cache import KVCache


class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the GPT-2 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`GPT2Model`] or [`TFGPT2Model`].
        n_positions (`int`, *optional*, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        n_embd (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        n_inner (`int`, *optional*):
            Dimensionality of the inner feed-forward layers. `None` will set it to 4 times n_embd
        resid_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        embd_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the embeddings.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-05):
            The epsilon to use in the layer normalization layers.
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
            Scale attention weights by dividing by sqrt(hidden_size)..
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        bos_token_id (`int`, *optional*, defaults to 50256):
            Id of the beginning of sentence token in the vocabulary.
        eos_token_id (`int`, *optional*, defaults to 50256):
            Id of the end of sentence token in the vocabulary.

    Example:

    ```python
    >>> from ml_zoo.transformers.gpt2 import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


class GPT2Model(nn.Module):
    """
    A minimal GPT-2 style decoder-only transformer language model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        vocab_size: int,
        max_position_embeddings: int,
        num_layers: int,
        use_cache: bool = True,
    ) -> None:
        super().__init__()
        self.max_position_embeddings = max_position_embeddings

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(max_position_embeddings, embed_dim)
        self.blocks = nn.Sequential(
            *[GPT2Block(embed_dim, num_heads, use_cache) for _ in range(num_layers)]
        )

        self.layernorm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(sequence_length, device=idx.device)
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.layernorm(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            _, _, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last allowed token
            idx_cond = idx[:, -self.max_position_embeddings :]
            logits, _ = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class GPT2Block(nn.Module):
    def __init__(self, embed_dim, num_heads, use_cache):
        super().__init__()

        self.attn_layer = GPT2MultiHeadAttention(
            embed_dim, num_heads, use_cache=use_cache
        )
        self.mlp_layer = GPT2MLP(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn_layer(self.layernorm1(x))
        x = x + self.mlp_layer(self.layernorm2(x))

        return x


class GPT2MLP(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )  # see https://arxiv.org/pdf/1706.03762#page=5

    def forward(self, x):
        return self.net(x)


class GPT2MultiHeadAttention(nn.Module):
    """
    Multi-head scaled dot-product for causal attention.

    This module implements the standard multi-head attention mechanism introduced in
    the Transformer architecture (Vaswani et al., 2017). The idea behind multi-head
    attention is to allow the model to jointly attend to information from different
    representation subspaces at different positions. Instead of performing a single
    attention operation on the full embedding dimension, the input is projected into
    several smaller “heads,” each of which performs attention independently. The
    outputs of all heads are then concatenated and linearly projected back to the
    model dimension.
    Multi-head attention increases the representational power of the model by
    splitting the embedding dimension `embed_dim` into `num_heads` smaller subspaces
    (`head_dim = embed_dim // num_heads`)

    References
    ----------
    - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
      Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need.
      https://arxiv.org/abs/1706.03762

    """

    def __init__(
        self, embed_dim: int, num_heads: int, attn_pdrop: float = 0.1, use_cache=False
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.use_cache = use_cache
        self.kv_cache = None

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wo = nn.Linear(embed_dim, embed_dim, bias=False)

        self.dropout = nn.Dropout(attn_pdrop)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        q = q.permute(0, 2, 1, 3)  # (B, nH, T, H)
        k = k.permute(0, 2, 1, 3)  # (B, nH, T, H)
        v = v.permute(0, 2, 1, 3)  # (B, nH, T, H)

        if self.use_cache:
            if self.kv_cache is None:
                # Prefil: initialize kv cache
                self.kv_cache = KVCache(batch_size, self.num_heads, 2048, self.head_dim)
                k, v = self.kv_cache.update(k, v)  # 2 x (B, nH, T, H)
            else:
                # Decoding: add last token
                k, v = self.kv_cache.update(k[:, :, -1:, :], v[:, :, -1:, :])

        kq = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)  # (B, nH, T, T)

        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=kq.device, dtype=torch.bool), diagonal=1
        )  # (T, T)
        kq = kq.masked_fill(mask, float("-inf"))
        att = F.softmax(kq, dim=-1)

        att = self.dropout(att)

        o = att @ v

        o = o.permute(0, 2, 1, 3).contiguous()  # (B, T, nH, H)
        o = o.view(batch_size, seq_len, embed_dim)  # concat heads
        o = self.Wo(o)
        o = self.dropout(o)
        return o
