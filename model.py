import torch
import torch.nn as nn
from data import *
from tqdm import tqdm

#hyperparmeter
#word_emb_size, pos_emb_size, hidden_dimesntion, no_layers, reduced_size (<hidden_dim)

class variable_class_biaffine(nn.Module):
    def __init__(self, d) -> None:
        super().__init__()
        self.d = d
        self.W = torch.nn.Parameter(torch.zeros((d, d)))
        self.b = torch.nn.Parameter(torch.zeros((d,)))
        
    def forward(self, heads, tails):
        HW = torch.einsum("hij, jk -> hik", heads, self.W)
        #print(HW.shape, tails.shape)
        affine_term = torch.einsum("hij, hkj -> hik", HW, tails)
        bias_matrix = self.b.expand(heads.shape[1], self.d)
        bias_term = torch.einsum("hij, kj -> hik", heads, bias_matrix)
        #print(bias_term.shape, affine_term.shape)
        return affine_term + bias_term

class fixed_class_biaffine(nn.Module):
    def __init__(self, l, d) -> None:
        super().__init__()
        self.l = l
        self.d = d
        self.U = torch.nn.Parameter(torch.zeros((d, d)))
        self.W = torch.nn.Parameter(torch.zeros((l, 2 * d)))
        self.b = torch.nn.Parameter(torch.zeros((l,)))

    def forward(self, heads, tails, scores):
        best_heads_indices = scores.argmax(dim=1)
        assert(best_heads_indices.shape == (tails.shape[0], tails.shape[1]))
        best_heads = torch.ones(tails.shape).to(heads.device)
        for i in range(len(tails)):
            best_heads[i] = heads[i, best_heads_indices[i]]
        
        # head U tail term
        UT = torch.einsum("ij, hkj -> hik", self.U, tails)
        HUT = torch.einsum("hri, hir -> hr", heads, UT)
        to_expand = torch.unsqueeze(HUT, dim=1)
        head_u_tail_term = to_expand.expand(tails.shape[0], self.l, tails.shape[1])

        # tails head concat term
        tails_head_cat = torch.cat((tails, best_heads), dim=2)
        tails_head_term = torch.einsum("ij, hkj -> hik", self.W, tails_head_cat)

        # bias term 
        bias_matrix = self.b.expand(tails.shape[1], self.l).T
        bias_term = bias_matrix.expand(tails.shape[0], self.l, tails.shape[1])
        return tails_head_term + bias_term + head_u_tail_term


class DepParser(nn.Module):
    def __init__(self, word_vocab, pos_vocab, dep_vocab, word_emb_size, pos_emb_size, hidden_dim, bilstm_layers, reduced_size, mlp_hidden_dim):
        super().__init__()
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.dep_vocab = dep_vocab
        self.word_emb_size = word_emb_size
        self.pos_emb_size = pos_emb_size
        self.hidden_dim = hidden_dim
        self.bilstm_layers = bilstm_layers
        self.reduced_size = reduced_size
        self.mlp_hidden_dim = mlp_hidden_dim

        # Embedding layer
        self.word_embd = nn.Embedding(len(word_vocab), word_emb_size, padding_idx=word_vocab["<pad>"])
        self.pos_embd = nn.Embedding(len(pos_vocab), pos_emb_size, padding_idx=pos_vocab["<pad>"])

        # Context layer
        self.blstm = torch.nn.LSTM(word_emb_size + pos_emb_size, hidden_dim, bilstm_layers, batch_first=True, bidirectional=True)

        # Data filtering layer
        self.head_arc = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, reduced_size),
            nn.ReLU()
        )
        self.tail_arc = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, reduced_size),
            nn.ReLU()
        )
        self.head_lab = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, reduced_size),
            nn.ReLU()
        )
        self.tail_lab = nn.Sequential(
            nn.Linear(2 * hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, reduced_size),
            nn.ReLU()
        )

        # Biaffine layer (arc)
        self.arc_biaffine = variable_class_biaffine(reduced_size)

        # Biaffine layer (label)
        self.label_biaffine = fixed_class_biaffine(len(self.dep_vocab), reduced_size)


    def forward(self, X):
        # Embedding layer
        words = X[:, :, 1]
        word_embeded = self.word_embd(words)

        pos = X[:, :, 2]
        pos_embeded = self.pos_embd(pos)
        x = torch.concatenate((word_embeded, pos_embeded), dim=2)
        assert(x.shape == (X.shape[0], X.shape[1], self.word_emb_size + self.pos_emb_size))

        # context layer
        v, _ = self.blstm(x)
        assert(v.shape == (x.shape[0], x.shape[1], 2 * self.hidden_dim))

        # data filtering layer
        r_head_arc = self.head_arc(v)
        r_tail_arc = self.tail_arc(v)
        r_head_lab = self.head_lab(v)
        r_tail_lab = self.tail_lab(v)
        assert(r_head_arc.shape == (X.shape[0], X.shape[1], self.reduced_size))

        # biaffine layer (arc)
        arc_scores = self.arc_biaffine(r_head_arc, r_tail_arc)
        assert(arc_scores.shape == (X.shape[0], X.shape[1], X.shape[1]))
        
        # biaffine layer (label)
        label_scores = self.label_biaffine(r_head_lab, r_tail_lab, arc_scores)
        assert(label_scores.shape == (X.shape[0], len(self.dep_vocab), X.shape[1]))

        return arc_scores, label_scores



if __name__ == "__main__":
    device = torch.device("cuda")
    print(device)

    vocab_pos_tags, vocab_deptyp, vocab_words = build_vocabularies("UD_English-EWT-master/en_ewt-ud-train.conllu")
    data_t = DepData("UD_English-EWT-master/en_ewt-ud-train.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    data = DataLoader(data_t, batch_size=32, shuffle=True, collate_fn=custom_collate)

    a = DepParser(vocab_words, vocab_pos_tags, vocab_deptyp, 100, 20, 128, 2, 60, 50).to(device)

    for X in tqdm(data):
        X = X.to(device)
        pred_arc, pred_labels = a(X)

        loss_arcs = 0
        for sen in range(X.shape[0]):
            label_mask = X[sen, :, 1] != 0
            pred_mask = label_mask.expand(X.shape[1], label_mask.shape[0])
            loss_arcs += torch.nn.functional.cross_entropy(pred_arc[sen][pred_mask].reshape((X.shape[1], -1)).T, X[sen][label_mask][:,3]).sum()    
        loss_labels = torch.nn.functional.cross_entropy(pred_labels[:, :, :], X[:, :, 4], ignore_index=vocab_deptyp["<pad>"]).sum()
        loss = loss_arcs + loss_labels
        loss.backward()
