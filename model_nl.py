import torch
import torch.nn as nn
import math
import torchvision
from transformers import BertModel
from transformers import logging
logging.set_verbosity_error()
#from tokenizers import Tokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        self.positional_encoding = nn.Parameter(torch.zeros(max_len, d_model))
        self.positional_encoding[:, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        return x + self.positional_encoding[:x.size(0), :x.size(1)].detach()

model = BertModel.from_pretrained('bert-base-chinese')
#pretrained_embeddings = model.embeddings.word_embeddings

class TransformerClassifier(nn.Module):
    def __init__(self, seq_pad_length, num_classes, d_model=512):
        super().__init__()
        self.embedding_layer = model.embeddings
        #self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
        #self.positional_encoding = PositionalEncoding(d_model=d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=6)

        self.last_layer = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(in_features=seq_pad_length * d_model, out_features=num_classes), 
            nn.Dropout(p=0.1)
        )


    def forward(self, x_src, x_mask):
        with torch.no_grad():
            #x = self.embedding(x)
            x_src = self.embedding_layer(x_src)
            #x = self.positional_encoding(x)

        x_src = self.transformer.forward(x_src, src_key_padding_mask=x_mask)
        x_src = self.last_layer(x_src)

        return x_src
    


if __name__ == '__main__':
    from dataset import ChineseTitleDataset
    from torch.utils.data import Dataset, DataLoader

    dataset = ChineseTitleDataset(['data/chinese/test.xlsx'])
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    model = TransformerClassifier(30, 2, d_model=768).to('cpu')

    for i, sample in enumerate(dataloader):
        sample_seq = sample['seq'][0]
        sample_label = sample['label'].cpu().numpy()[0]
        sample_tokens = sample['tokens'].cpu().numpy()[0]
        sample_mask = sample['mask'].cpu().numpy()[0]
        
        #print(f'[{sample_label}]  seq: {sample_seq}, tokens: {sample_tokens}, mask: {sample_mask},   [{sample_tokens.shape}|{sample_mask.shape}]')

        output = model(sample['tokens'], sample['mask'])

        print(f'[{sample_label}]  seq: {sample_seq}, output: {output}')

        #break
        if i >= 20:
            break




