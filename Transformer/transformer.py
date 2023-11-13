import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed

# can we do a stack-transformer?

device = torch.device('cpu')

class LSTM(nn.Module):
    """Documentation for ClassName
    """
    def __init__(self, vocab_size, symbol_dim, lstm_dim):
        super(LSTM, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, symbol_dim)
        self.LSTM = nn.LSTM(lstm_dim,
                            lstm_dim,
                            bidirectional=False,
                            batch_first=True)

        self.classification_layer = nn.Linear(lstm_dim, vocab_size)

    def forward(self, input, predict_idxs):
        predict_idxs = torch.cat([predict_idxs[:,1:], torch.zeros(512).unsqueeze(-1).to(device)], -1)  
        input = self.embeddings(input)
        encoded, _ = self.LSTM(input)

        encoded = encoded.reshape(encoded.size(0)*encoded.size(1), encoded.size(-1))[predict_idxs.view(-1)>0]
        return self.classification_layer(encoded)


class TransformerModel(nn.Module):
    """Documentation for TranformerModel
    """
    def __init__(self, vocab_size, encoder_dim, pos_dim, num_heads, num_layers, max_len):
        super(TransformerModel, self).__init__()
        self.base_encoder_dim = encoder_dim
        self.embeddings = nn.Embedding(vocab_size, self.base_encoder_dim)
        self.positional_encoding = PositionalEncoding(pos_dim, max_len)
        self.encoder_dim = self.base_encoder_dim + pos_dim
        self.layers = nn.ModuleList([TransformerEncoder(self.encoder_dim, num_heads)
                                     for _ in range(num_layers)])

        self.classification_layer = nn.Linear(self.encoder_dim, vocab_size)

        self.num_heads = num_heads
        self.num_layers = num_layers
        
    def forward(self, input, predict_idxs):
        embedded = self.embeddings(input)
        positions = torch.arange(
            input.size(1)).repeat(
                input.size(0)).view(
                    input.size()).to(device)
        encoded = self.positional_encoding(positions, embedded)

        layer_container = torch.zeros(self.num_layers, self.num_heads, 20, 20).to(device)
        for j, att_layer in enumerate(self.layers):
            encoded, scores = att_layer(encoded)
            layer_container[j,:,:,:] = scores.squeeze(1)

        encoded = encoded.view(-1, encoded.size(-1))[predict_idxs.view(-1)>0]
        return self.classification_layer(encoded), layer_container#scores.squeeze(1)

class TransformerEncoder(nn.Module):
    """Documentation for TransformerEncoder
    """
    def __init__(self, dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.1)
        self.multiheaded_attention = nn.ModuleList([SelfAttention(dim)
                                                    for _ in range(num_heads)])


        self.ffn = nn.Sequential(self.dropout,
                                 nn.Linear(self.num_heads*dim, self.num_heads*dim),
                                 nn.GELU(),
                                 nn.Linear(self.num_heads*dim, dim))
        
    def forward(self, input):
        container = torch.zeros(self.num_heads,
                                input.size(0),
                                input.size(1),
                                input.size(2)).to(device)
        score_container = torch.zeros(self.num_heads,
                                      input.size(0),
                                      input.size(1),
                                      input.size(1)).to(device)
        for i, attention_head in enumerate(self.multiheaded_attention):
            output, scores = attention_head(input)
            container[i,:,:,:] = output
            score_container[i,:,:,:] = scores
            
        container = torch.cat([container[i,:] for i in range(self.num_heads)], -1)
        assert container.size(-1) == self.num_heads * input.size(-1)
        
        output = self.ffn(container)
        return output, score_container

class SelfAttention(nn.Module):
    """Documentation for SelfAttention
    """
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.Q_transform = nn.Linear(dim, dim)
        self.K_transform = nn.Linear(dim, dim)
        self.V_transform = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1) # 0.1 in "attn is all you need"
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.Q_transform(input),
                              self.K_transform(input).transpose(-2,-1))/math.sqrt(input.size(-1))
        s_scores = F.softmax(scores, -1)
        weighted_sum = torch.matmul(s_scores, self.V_transform(input))
        d_weighted_sum = self.dropout(weighted_sum)
        
        # residual connection and layer norm
        output = self.layer_norm(torch.add(d_weighted_sum, input))
        
        return output, s_scores
        
class PositionalEncoding(nn.Module):
    """Documentation for PositionalEncoding
    """
    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.pos_emb = nn.Embedding(max_len, dim)
        
    def forward(self, input, embedded):
        return torch.cat([embedded, self.pos_emb(input)],-1)
