import torch
import torch.nn as nn
import math
import torchvision
from transformers import BertModel, BertTokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.positional_encoding[:, :x.size(1)].detach()

model = BertModel.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
pretrained_embeddings = model.embeddings.word_embeddings

class TransformerClassifier(nn.Module):
    def __init__(self, seq_pad_length, num_classes):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        self.positional_encoding = PositionalEncoding(d_model=512)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

        #self.last_layer = nn.Sequential(
        #    nn.Linear()
        #)

        self.fc = nn.Linear(pretrained_embeddings.embedding_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        output = self.transformer(x)
        return output
    


if __name__ == '__main__':
    pass

