import conllu
import torch
import torchtext
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence




def build_vocabularies(data_file):
    pos_tags = [ "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
    pos_tags = [[i] for i in pos_tags]
    assert(len(pos_tags) == 17)
    vocab_pos_tags = torchtext.vocab.build_vocab_from_iterator(pos_tags)
    vocab_pos_tags.append_token("pad")
    
    dep_typ = ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse" , "aux", "cop", "mark",	 "nmod", "appos", "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc",	 "fixed", "flat", "compound", "list", "parataxis", "orphan", "goeswith", "reparandum", "punct", "root", "dep"]
    dep_typ = [[i] for i in dep_typ]
    assert len(dep_typ) == 37
    vocab_deptyp = torchtext.vocab.build_vocab_from_iterator(dep_typ)
    vocab_deptyp.append_token("pad")

    words = [["<unk>"], ["<unk>"]]
    TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
    for TokenList in TokenLists:
        for token in TokenList:
            words.append([token["form"]])
    vocab_words = torchtext.vocab.build_vocab_from_iterator(words, min_freq=2)
    vocab_words.set_default_index(vocab_words["<unk>"])
    vocab_words.append_token("pad")
    return vocab_pos_tags, vocab_deptyp, vocab_words


class DepData(Dataset):
    def __init__(self, data_file, vocab_pos_tags, vocab_deptyp, vocab_words):
        super().__init__()
        self.vocab_pos_tags = vocab_pos_tags
        self.vocab_deptyp = vocab_deptyp
        self.vocab_words = vocab_words

        Data = []
        TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
        for TokenList in TokenLists:
            data = []
            data.append([0, self.vocab_words["pad"], self.vocab_pos_tags["pad"], 0, self.vocab_deptyp["pad"]])
            for token in TokenList:
                if type(token["id"]) == int:
                    data.append([token["id"], self.vocab_words[token["form"]], self.vocab_pos_tags[token["upos"]], token["head"], self.vocab_deptyp[token["deprel"].split(":")[0]]])
                else:
                    pass
            data = torch.LongTensor(data)
            Data.append(data)
        self.Data = Data

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, index):
        return self.Data[index]

 
def custom_collate(batch):
    return pad_sequence(batch, batch_first=True)




# vocab_pos_tags, vocab_deptyp, vocab_words = build_vocabularies("UD_English-EWT-master/en_ewt-ud-train.conllu")
# print(vocab_deptyp["pad"], vocab_pos_tags["pad"], vocab_words["pad"])
# data_t = DepData("UD_English-EWT-master/en_ewt-ud-train.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
# data_2 = DepData("UD_English-EWT-master/en_ewt-ud-test.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
# data_3 = DepData("UD_English-EWT-master/en_ewt-ud-dev.conllu",  vocab_pos_tags, vocab_deptyp, vocab_words)
# data_3_loader = DataLoader(data_3, batch_size=32, shuffle=True, collate_fn=custom_collate)
# for x in data_3_loader:
#     pass