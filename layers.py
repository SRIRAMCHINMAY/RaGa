import torch
import torch.nn.functional as F
from torch import nn
from math import sqrt
from hparams import device

def abs_positional_encoding(max_p, d_model, n=3):
    # set of all positions to consider
    positions=torch.arange(max_p).float().to(device)

    #get angles to input to to sinusoid functions
    k=torch.arange(d_model).float().to(device)
    coeffs=1/torch.pow(1000,2*(k//2)/d_model)
    angles=positions.view(-1,1) @ coeffs.view(-1,1)
    # apply sin to the even indices of angles along the last axis
    angles[:,0::2]=torch.sin(angles[:,0::2])
    #applying cos to the odd indices
    angles[:,1::2]=torch.cos(angles[:,1::2])
    return angles.view(*[1 for _ in range(n-2)], max_p,d_model)

def skew(t):
    """
    Implements Huang et. al, 2018's skewing algorithm to correctly reorder the dot(Q, RelativePositionEmbeddings)
    matrix. This function generalizes to any shape and any number of dimensions. However, attention calculation
    requires shape (..., L, L).

    Algorithm:
        1. Pad T
        2. Reshape
        3. Slice

    Args:
        t (torch.Tensor): tensor to skew

    Returns:
        Srel: skewed t: nth column from the right is skewed into the nth diagonal under the main; same shape as t
    """
    #pad T
    padded=F.pad(t,[1,0])
    Srel=padded.reshape(-1,t.shape[-1]+1,t.shape[-2])

    Srel=Srel[:,1:]
    Srel=Srel.reshape(*t.shape)
    return Srel

def rel_scaled_dot_prod_attention(q,k,v,e=None,mask=None):
    """
    A modification given by Shaw et. al, 2018, improved by Huang et. al, 2018, to the Scaled Dot-Product Attention
    mechanism given in Vaswani et. al, 2017, which allows the Transformer model to attend to all relevant elements of
    the input sequences as well as the relative distances between them.

    RelAttention = softmax( mask( QKT + skew(QET) ) / sqrt(d_k) ) V

    Args:
        q: Queries tensor of shape (..., seq_len_q, d_model)
        k: Keys tensor of shape (..., seq_len_k, d_model)
        v: Values tensor of shape (..., seq_len_k, d_model)
        e (optional): Relative Position Embeddings tensor of shape (seq_len_k, d_model)
        mask (optional): mask for input batch with ones indicating the positions to mask

    Returns:
        output attention of shape (..., seq_len_q, d_model)
    """
    Qkt=torch.matmul(q,k.transport(-1,-2))
    if e is None:
        # assumes q.shape[:-2] == k.shape[:-2]
        Srel=torch.zeros(*q.shape[:-2],q.shape[-2],k.shape[-2],device=q.device)
    else:
        Srel=skew(torch.matmul(q,e.transpose(-1,-2))) # (..., seq_len_q, seq_len_k)
    dk=sqrt(k.shape[-1])
    scaled_attention_logits=(Qkt+Srel)/dk

    if mask is not None:
        scaled_attention_logits+=(mask*-1e9)
    return torch.matmul(F.softmax(scaled_attention_logits,dim=-1),v)

class MultiHeadAttention(nn.Module):
    def __innit__(self,d_model,num_heads,max_rel_dist,bias=True):
        """
        Args:
            d_model (int): Transformer hidden dimension size
            num_heads (int): number of heads along which to calculate attention
            max_rel_dist (int): maximum relative distance between positions to consider in creating
                                relative position embeddings; set to 0 to compute normal attention
            bias (bool, optional): if set to False, all Linear layers in the MHA block will not learn
                                   an additive bias. Default: True

        """
        super(MultiHeadAttention,self).__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        self.max_rel_dist=max_rel_dist
        self.batch_first=False

        if d_model % num_heads!=0:
            raise ValueError("d_model must be divisible by num_heads")
        self.depth=self.d_model//self.num_heads
        self.wq = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate Q from input
        self.wk = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate K from input
        self.wv = nn.Linear(self.d_model, self.d_model, bias=bias)  # parameter matrix to generate V from input

        self.E = nn.Embedding(self.max_rel_dist, self.d_model)      # relative position embeddings

        self.wo = nn.Linear(self.d_model, self.d_model, bias=True)  # final output layer