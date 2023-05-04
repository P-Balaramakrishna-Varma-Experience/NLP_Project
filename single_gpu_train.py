import torch
from tqdm import tqdm

from data import *
from model import *
import math


import matplotlib.pyplot as plt

def plot(valid_losss, train_losss, filename):
    plt.clf()
    plt.plot(valid_losss, label="valid loss")
    plt.plot(train_losss, label="train loss")
    plt.legend()
    plt.savefig(filename)


def train_epoch(model, dataloader, loss_func, optimizer, device):
    model.train()
    for X in dataloader:
        # Data collection
        X = X.to(device)
        
        # forward pass
        pred_arc, pred_labels = model(X)
        loss = loss_func(pred_arc, pred_labels, X)
        
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval(model, dataloader, loss_func, device):
    model.eval()
    loss = 0
    uas = 0
    las = 0
    total_samples = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred_arc, pred_labels = model(X)
            loss += loss_func(pred_arc, pred_labels, X).item()
            
            mask = X[:, :, 1] != 0
            uas_mask = pred_arc.argmax(dim=1)[mask] == X[:, :, 3][mask]
            las_mask = torch.logical_and(uas_mask, pred_labels.argmax(dim=1)[mask] == X[:, :, 4][mask])
            
            total_samples += mask.sum().item()
            uas += uas_mask.sum().item()
            las += las_mask.sum().item()

    return loss / total_samples, uas / total_samples, las / total_samples            


def loss_cal(pred_arc, pred_labels, X):
    loss_arcs = 0
    for sen in range(X.shape[0]):
        label_mask = X[sen, :, 1] != 0
        pred_mask = label_mask.expand(X.shape[1], label_mask.shape[0])
        loss_arcs += torch.nn.functional.cross_entropy(pred_arc[sen][pred_mask].reshape((X.shape[1], -1)).T, X[sen][label_mask][:,3]).sum()    
    loss_labels = torch.nn.functional.cross_entropy(pred_labels[:, :, :], X[:, :, 4], ignore_index=0).sum()
    loss = loss_arcs + loss_labels
    return loss


def single_gpu_train(hypar):
    device = torch.device("cuda")
    print(device)

    ## hypar
    batch_size = hypar["batch_size"]
    lr = hypar["lr"]
    weight_decay = hypar["weight_decay"]
    epochs = hypar["epochs"]

    word_embedding_dim = hypar["word_embedding_dim"]
    pos_embedding_dim = hypar["pos_embedding_dim"]
    lstm_hidden_dim = hypar["lstm_hidden_dim"]
    lstm_num_layers = hypar["lstm_num_layers"]
    arch_mlp_size = hypar["arch_mlp_size"]
    label_mlp_size = hypar["label_mlp_size"]
    dropout = hypar["dropout"]

    ## data processing
    vocab_pos_tags, vocab_deptyp, vocab_words = build_vocabularies("UD_English-EWT-master/en_ewt-ud-train.conllu")
    data_train = DepData("UD_English-EWT-master/en_ewt-ud-train.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    data_valid = DepData("UD_English-EWT-master/en_ewt-ud-dev.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    data_test = DepData("UD_English-EWT-master/en_ewt-ud-test.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    valid_dataloader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)

    ## model, optimizer, loss function
    model = DepParser(vocab_words, vocab_pos_tags, vocab_deptyp, word_embedding_dim, pos_embedding_dim, lstm_hidden_dim, lstm_num_layers, arch_mlp_size, label_mlp_size, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.9))
    loss_func = loss_cal

    ## training
    valid_losss = []
    train_losss = []
    for epoch in tqdm(range(epochs)):
        train_epoch(model, train_dataloader, loss_func, optimizer, device)
        valid_loss, _, _ = eval(model, valid_dataloader, loss_func, device)
        train_loss, _, _ = eval(model, train_dataloader, loss_func, device)
        valid_losss.append(valid_loss)
        train_losss.append(train_loss)
    
    stats = eval(model, test_dataloader, loss_func, device)


    return valid_losss, train_losss, stats



if __name__ == "__main__":
    hypar = {"word_embedding_dim": 100, "pos_embedding_dim": 20, "lstm_hidden_dim": 400, "lstm_num_layers": 3, "arch_mlp_size": 500, "label_mlp_size": 100, "batch_size": 32, "lr": 2e-3, "weight_decay": 1e-4 , "dropout": 0.1, "epochs": 10}
    
    # hypar["dropout"] = 0.1
    # valid_loss, train_loss, stats = single_gpu_train(hypar)
    # print("dropout = 0.1", stats)
    # plot(valid_loss, train_loss, "dropout_01.png")

    # hypar["dropout"] = 0.2
    # valid_loss, train_loss, stats = single_gpu_train(hypar)
    # print("dropout = 0.2", stats)
    # plot(valid_loss, train_loss, "dropout_02.png")

    # hypar["dropout"] = 0.3
    # valid_loss, train_loss, stats = single_gpu_train(hypar)
    # print("dropout = 0.3", stats)
    # plot(valid_loss, train_loss, "dropout_03.png")

    # hypar["dropout"] = 0.4
    # valid_loss, train_loss, stats = single_gpu_train(hypar)
    # print("dropout = 0.4", stats)
    # plot(valid_loss, train_loss, "dropout_04.png")

    hypar["dropout"] = 0.5
    valid_loss, train_loss, stats = single_gpu_train(hypar)
    print("dropout = 0.5", stats)
    plot(valid_loss, train_loss, "dropout_05.png")

    hypar["dropout"] = 0.6
    valid_loss, train_loss, stats = single_gpu_train(hypar)
    print("dropout = 0.6", stats)
    plot(valid_loss, train_loss, "dropout_06.png")

    hypar["dropout"] = 0.8
    valid_loss, train_loss, stats = single_gpu_train(hypar)
    print("dropout = 0.8", stats)
    plot(valid_loss, train_loss, "dropout_06.png")