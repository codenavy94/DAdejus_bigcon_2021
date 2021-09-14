import torch
from opt import parse_opts
from utils.data_loader import *
import os

import torch.optim as optim
from utils.epoch_loader import *
import copy
import torch.nn as nn
from sklearn.model_selection import train_test_split


def create_dataset(X):

    sequences = X.astype(np.float32).to_numpy().tolist()

    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
    dataset = torch.stack(dataset)
    n_seq, seq_len, n_feafures = dataset.shape

    return dataset, seq_len, n_feafures
opt = parse_opts()

def set_model(dpath='../dataset/cifar-10-batches-py',train_size=0.8, batch_size=40):



    device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

    print(device, 'use')


    # result = load_data('./dataset/baseball_train_final_not_push.csv')
    result = load_data('./dataset/base_ball_seq.csv')
    print(result)

    X = result[['t'+str(i) for i in range(2,12)]]
    y = result['t1']

    # dataset = SequenceDataset(X=torch.Tensor(X), y=torch.Tensor(y))

    # X_train, X_val, y_train, y_val = train_test_split(X,y)

    # train, val = data.random_split(dataset,
    #                                [int(len(dataset) * train_size), len(dataset) - int(len(dataset) * train_size)])
    #
    # train_loader = data.DataLoader(train, batch_size=batch_size, shuffle=True)
    # print()
    # val_loader = data.DataLoader(val, batch_size=batch_size, shuffle=True)
    train_X, _, val_X, _ = train_test_split(X, y, test_size=0.1)

    train_dset, seq_len, n_feafures = create_dataset(train_X)
    val_dataset, _, _ = create_dataset(val_X)


    # train_loader = data.DataLoader(train_dset, )

    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    print()

    model = nn.LSTM(input_size=seq_len, hidden_size=n_feafures)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = dict(train_los=[], train_acc=[], val_los=[], val_acc=[], present_params=copy.deepcopy(model.state_dict()))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    model_params = dict(best_params=best_model_wts, best_loss=best_loss)

    print()



    # start_epoch = 1
    # opt.n_epochs = 1
    # for epoch in range(start_epoch, opt.n_epochs + 1):
    #     val_loss, val_acc = val_epoch(model, val_loader, criterion, device)

    return model, (train_loader, val_loader),  criterion, optimizer, history, model_params, device

def epoch_train(epoch, model, loaders, criterion, optimizer, history, model_params, device):

    train_loader, val_loader = loaders

    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
    history['train_los'].append(train_loss), history['train_acc'].append(train_acc)

    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
    history['val_los'].append(val_loss), history['val_acc'].append(val_acc)
    history['present_params'] = copy.deepcopy(model.state_dict())

    if val_loss < model_params['best_loss']:
        model_params['best_loss'] = val_loss
        model_params['best_params'] = copy.deepcopy(model.state_dict())

    return epoch+1, model, loaders, criterion, optimizer, history, model_params, device


def full_epoch_train(model, loaders, criterion, optimizer, history, model_params, device):

    start_epoch = 1
    opt.n_epochs = 1

    train_loader, val_loader = loaders

    for epoch in range(start_epoch, opt.n_epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
        history['train_los'].append(train_loss), history['train_acc'].append(train_acc)
        val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
        history['val_los'].append(val_loss), history['val_acc'].append(val_acc)

        if val_loss < model_params['best_loss']:
            model_params['best_loss'] = val_loss
            model_params['best_params'] = copy.deepcopy(model.state_dict())

    return history, model_params

def main():
    model, loaders, criterion, optimizer, history, model_params, device = set_model(
        dpath='../dataset/cifar-10-batches-py',
        train_size=0.8, batch_size=40)

    start_epoch = 1
    opt.n_epochs = 1
    epoch = 1
    epoch_train(epoch, model, loaders, criterion, optimizer, history, model_params, device)
    # full_epoch_train(model, loaders, criterion, optimizer, device)


if __name__=='__main__':
    # main()
    model, loaders, criterion, optimizer, history, model_params, device = set_model(
        dpath='../dataset/cifar-10-batches-py',
        train_size=0.8, batch_size=20)
    history, model_params = full_epoch_train(model, loaders, criterion, optimizer, history, model_params, device)

