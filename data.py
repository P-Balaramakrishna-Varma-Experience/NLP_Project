import conllu
import torch
import torchtext
import torch.utils.data.dataset as dataset


class DepData(dataset):
    def __init__(self, data_file):
        # build the three vobacs for words and pos embeddings
        pos_tags = [ "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
        pos_tags = [[i] for i in pos_tags]
        assert(len(pos_tags) == 17)
        vocab_pos_tags = torchtext.vocab.build_vocab_from_iterator(pos_tags)

        dep_typ = ["nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse" , "aux", "cop", "mark",	 "nmod", "appos", "nummod", "acl", "amod", "det", "clf", "case", "conj", "cc",	 "fixed", "flat", "compound", "list", "parataxis", "orphan", "goeswith", "reparandum", "punct", "root", "dep"]
        dep_typ = [[i] for i in dep_typ]
        assert len(dep_typ) == 37
        vocab_deptyp = torchtext.vocab.build_vocab_from_iterator(dep_typ)

        words = []
        TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
        for TokenList in TokenLists:
            for token in TokenList:
                words.append([token["form"]])
        vocab_words = torchtext.vocab.build_vocab_from_iterator(words)


        # building the data
        Data = []
        TokenLists = conllu.parse_incr(open(data_file, "r", encoding="utf-8"))
        for TokenList in TokenLists:
            no_of_tokens = len(TokenList)
            data = -1 * torch.ones((no_of_tokens, 5), dtype=torch.long)
            for i in range(no_of_tokens):
                token = TokenList[i]
                if type(token["id"]) == int:
                    data[i] = torch.LongTensor([token["id"], vocab_words[token["form"]], vocab_pos_tags[token["upos"]], token["head"], vocab_deptyp[token["deprel"].split(":")[0]]])
            Data.append(data)
        self.Data = Data

    def __len__(self):
        return len(self.Data)
    
    def __getitem__(self, index):
        return self.Data[index]