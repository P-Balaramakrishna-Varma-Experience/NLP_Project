import torch
import torch.nn as nn
from data import *

#hyperparmeter
#word_emb_size, pos_emb_size

class DepParser(nn.Module):
    def __init__(self, word_vocab, pos_vocab, word_emb_size, pos_emb_size):
        super().__init__()
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.word_emb_size = word_emb_size
        self.pos_emb_size = pos_emb_size

        # Embedding layer
        self.word_embd = nn.Embedding(len(word_vocab), word_emb_size, padding_idx=word_vocab["pad"])
        self.pos_embd = nn.Embedding(len(pos_vocab), pos_emb_size, padding_idx=pos_vocab["pad"])


    def forward(self, X):
        # Embedding layer
        words = X[:, :, 1]
        word_embeded = self.word_embd(words)

        pos = X[:, :, 2]
        pos_embeded = self.pos_embd(pos)
        x = torch.concatenate((word_embeded, pos_embeded), dim=2)
        
        assert(x.shape == (X.shape[0], X.shape[1], self.word_emb_size + self.pos_emb_size))



# vocab_pos_tags, vocab_deptyp, vocab_words = build_vocabularies("UD_English-EWT-master/en_ewt-ud-train.conllu")
# #print(vocab_deptyp["pad"], vocab_pos_tags["pad"], vocab_words["pad"])
# data_t = DepData("UD_English-EWT-master/en_ewt-ud-train.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
# data = DataLoader(data_t, batch_size=32, shuffle=True, collate_fn=custom_collate)

# device = torch.device("cuda")
# a = DepParser(vocab_words, vocab_pos_tags, 100, 20).to(device)
# for X in data:
#     X = X.to(device)
#     a(X)