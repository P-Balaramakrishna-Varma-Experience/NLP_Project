import torch
from tqdm import tqdm

from data import *
from model import *


import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


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


def ddp_setup(rank, world_size):
    # rank 0 process is master (coummication with in process)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size=world_size)




def multi_gpu_train(rank, world_size, hypar, queue, Event):
    # Configuaration
    ddp_setup(rank, world_size)
    device = torch.device("cuda", rank)

    ## hypar
    batch_size = hypar["batch_size"]
    lr = hypar["lr"]
    weight_decay = hypar["weight_decay"]
    epochs = hypar["epochs"]

    word_embedding_dim = hypar["word_embedding_dim"]
    pos_embedding_dim = hypar["pos_embedding_dim"]
    lstm_hidden_dim = hypar["lstm_hidden_dim"]
    lstm_num_layers = hypar["lstm_num_layers"]
    reduce_dim = hypar["reduce_dim"]
    mlp_hidden_dim = hypar["mlp_hidden_dim"]

    ## data processing
    vocab_pos_tags, vocab_deptyp, vocab_words = build_vocabularies("UD_English-EWT-master/en_ewt-ud-train.conllu")
    data_train = DepData("UD_English-EWT-master/en_ewt-ud-train.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    data_valid = DepData("UD_English-EWT-master/en_ewt-ud-dev.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    data_test = DepData("UD_English-EWT-master/en_ewt-ud-test.conllu", vocab_pos_tags, vocab_deptyp, vocab_words)
    
    train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(data_train))
    valid_dataloader = DataLoader(data_valid, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(data_valid))
    test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=False, collate_fn=custom_collate, sampler=DistributedSampler(data_test))

    ## model, optimizer, loss function
    model = DepParser(vocab_words, vocab_pos_tags, vocab_deptyp, word_embedding_dim, pos_embedding_dim, lstm_hidden_dim, lstm_num_layers, reduce_dim, mlp_hidden_dim).to(device)
    model = DDP(model, device_ids=[device])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
    
    loss_values = torch.zeros((epochs, 2))
    loss_values[:][0] = torch.tensor(train_losss)
    loss_values[:][1] = torch.tensor(valid_losss)
    queue.put(loss_values)
    Event.wait()

    print("hurray")
    destroy_process_group()
    return valid_losss, train_losss


def get_stats_multi_gpu(hyperpar):
    world_size = torch.cuda.device_count()
    print("Number of GPUs: ", world_size)
    Events = [mp.Event() for _ in range(world_size)]
    qeue = mp.SimpleQueue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=multi_gpu_train, args=(rank, world_size, hyperpar, qeue, Events[rank]))
        processes.append(p)
        p.start()
    
    Data = torch.zeros((hyperpar["epochs"], 2))
    for _ in range(world_size):
        Data += qeue.get()
    Data = Data / world_size

    for event in Events:
        event.set()
    for p in processes:
        p.join()

    return Data



if __name__ == "__main__":
    hypar = {"word_embedding_dim": 100, "pos_embedding_dim": 20, "lstm_hidden_dim": 128, "lstm_num_layers": 2, "reduce_dim": 60, "mlp_hidden_dim": 50, "batch_size": 32, "lr": 0.001, "weight_decay": 0.0001, "epochs": 2}
    Data = get_stats_multi_gpu(hypar)
    print(Data)