import argparse
import numpy as np
import torch.nn as nn
import torch

from models import RNN, LSTM, GRU
from data import AustinWeather
from torch.utils.data import DataLoader
from supervisor import Supervisor

def main(args):
    """
    main function to start training and testing:
    Params:
        args.batch_size: batch_size for training data.
        args.epochs: number of epochs to iterate the entire training set over.
        args.lr: optimizer's learning rate.
        args.horizon: predicting horizon.
        args.file_span: number of files to use.
        args.hidden_size: size of hidden states.
        args.num_layers: number of RNN layers.
        args.output_size: network output size, default=1 for binary classification.
        args.cell_type: RNN cell type \in [RNN, LSTM, GRU].

    """
    print("Preparing training data...")
    dataset = AustinWeather("../../Processed Data/austin_weather.csv", args.seq_len, args.horizon, args.feat_idx)
    print("Preparing validation data...")
    val_set = AustinWeather("../../Processed Data/austin_weather.csv", args.seq_len, args.horizon, args.feat_idx, mode="val")
    print("Training set size:", dataset.__len__())
    print("Validation set size:", val_set.__len__())
    print("Input shape is", dataset.x.shape, "Class number", np.unique(dataset.y, return_counts=True))
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_set.__len__())
    if args.cell_type == "RNN":
        model = RNN(input_size=dataset.x.shape[2], hidden_size=2, num_layers=2, output_size=args.output_size)
    elif args.cell_type == "LSTM":
        model = LSTM(input_size=dataset.x.shape[2], hidden_size=2, num_layers=2, output_size=args.output_size)
    elif args.cell_type == "GRU":
        model = GRU(input_size=dataset.x.shape[2], hidden_size=2, num_layers=2, output_size=args.output_size)
    else:
        raise NameError("Specified cell type not recognized!")
    
    trainer = Supervisor(model, args.epochs, args.batch_size, torch.optim.RMSprop, args.lr, nn.BCELoss(), args.horizon)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    np.random.seed(123)
    torch.manual_seed(456)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=128, type=int, help="batch size of training data")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs for training")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--horizon", default=1, type=int, help="prediction horzion: int * 5 = min ahead")

    parser.add_argument("--file-span", default=15, type=int, help="number of month to use for both training and testing")
    parser.add_argument("--seq-len", default=7, type=int, help="sequence history to use")
    parser.add_argument("--hidden-size", default=100, type=int, help="num of hidden states")
    parser.add_argument("--num-layers", default=100, type=int, help="number of RNN layers")
    parser.add_argument("--output-size", default=1, type=int, help="output size (depends on the task)")

    parser.add_argument("--cell-type", default="RNN", type=str, help="RNN cell type \in [RNN, LSTM, GRU]")
    parser.add_argument("--feat_idx", default="1 4 7 9 12 18", type=str)

    args = parser.parse_args()

    main(args)