import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, dropout=0.0):
        super().__init__()
        #Key, Query, Value are linear layers which take as input the embedded vectors of words
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape #batch size, sequence length, embedding dimension
        k = self.key(x)   # (B,T,C)
        q = self.query(x)  # (B,T,C)
        # compute attention scores ("affinities")
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5 # @-> dot product
        wei = F.softmax(wei, dim=-1)  # (B, T, T) -> how much attention gives token i to token j
        #with softmax we have a weigthed relation of each word with every other word
        
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out


class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class SimpleSelfAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=35):
        super().__init__()

        self.n_head = 1 #number of heads 
        self.max_length = max_length

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True)
        self.position_embedding_table = nn.Embedding(self.max_length, dim) #creates embeddings that represents the position of the words
        #for example not before good

        head_size = dim // self.n_head #each head takes a part of the emb vector
        # n_head is the number of the heads
        self.sa = Head(head_size, dim)
        self.ffwd = FeedFoward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # TODO: Main-lab-Q3 - define output classification layer
        self.output = nn.Linear(dim, output_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding_table(x)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = x + self.sa(self.ln1(x)) #Normaliazation layer, attention , residual connection
        x = x + self.ffwd(self.ln2(x)) #push the x after attention to ffn

        # TODO: Main-lab-Q3 - avg pooling to get a sentence embedding
        x = x.mean(dim=1)  # (B,C)

        logits = self.output(x)  # (C,output)
        return logits














class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd) # create different heads (different weigths)
                                    for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) #take the pices of each head and reconvert it back to vectors size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x): #x  shape: (B, T, C)
        out = torch.cat([h(x) for h in self.heads], dim=-1) #each h(x) is a tensor of shape (Β, Τ, head_size)
        # [B, T, head_size] × num_heads  →  [B, T, head_size × num_heads] = [B, T, n_embd]
        out = self.dropout(self.proj(out))
        return out



class MultiHeadAttentionModel(nn.Module):

    def __init__(self, output_size, embeddings, max_length=35, n_head=3):
        super().__init__()

        # TODO: Main-Lab-Q4 - define the model
        # Hint: it will be similar to `SimpleSelfAttentionModel` but
        # `MultiHeadAttention` will be utilized for the self-attention module here
        self.n_head = n_head
        self.max_length = max_length

        embeddings = np.array(embeddings)
        num_embeddings, dim = embeddings.shape

        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True
        )

        self.position_embedding_table = nn.Embedding(self.max_length, dim)


        head_size = dim // n_head
        self.attn = MultiHeadAttention(n_head, head_size, dim)

        self.ffwd = FeedFoward(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, output_size)


    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)

        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = x.mean(dim = 1)  # shape: (B, C)

        logits = self.output(x)
        return logits







"""
Ο Transformer αποτελείται από πολλαπλά επαναλαμβανόμενα blocks,
ενώ το απλό attention-based μοντέλο σου έχει μόνο ένα block 
(attention + feedforward).
"""


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_head, head_size, n_embd):
        # n_embd: embedding dimension, n_head: the number of heads we'd like for each block
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) #(Β,Τ,C)
        return x






class TransformerEncoderModel(nn.Module):
    def __init__(self, output_size, embeddings, max_length=35, n_head=3, n_layer=3):
        super().__init__()

        # TODO: Main-Lab-Q5 - define the model
        # Hint: it will be similar to `MultiHeadAttentionModel` but now
        # there are blocks of MultiHeadAttention modules as defined below
        
        self.max_length = max_length
        self.n_head = n_head
        num_embeddings, dim = embeddings.shape
        self.token_embedding_table = nn.Embedding(num_embeddings, dim)
        self.token_embedding_table = self.token_embedding_table.from_pretrained(
            torch.Tensor(embeddings), freeze=True
        )

        self.position_embedding_table = nn.Embedding(self.max_length, dim)



        head_size = dim // self.n_head
        self.blocks = nn.Sequential(
            *[Block(n_head, head_size, dim) for _ in range(n_layer)]) #creates a lstack of n_layers blocks        self.ln_f = nn.LayerNorm(dim)  # final layer norm
        
        self.ln_f = nn.LayerNorm(dim)

        #(B, T, dim)  # δηλαδή: batch, tokens, embedding_dim

        self.output = nn.Linear(dim, output_size)

    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding_table(x)        # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T))  # (T, C)
        x = tok_emb + pos_emb                          # (B, T, C)


        x = self.blocks(x) # passes through transformer blocks
        x = self.ln_f(x)        
        x = x.mean(dim=1)        # (B, dim)

        logits = self.output(x)  # (B, output_size)
        return logits

