import argparse
import numpy as np
import torch.nn as nn
import torch

from models import RNN, LSTM, GRU
from data import WeatherDataset
from torch.utils.data import DataLoader
from supervisor import Supervisor

def main(args):
    dataset = WeatherDataset("../../Processed Data/cleaned_data/", args.file_span, args.seq_len, args.horizon)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=10)

    model = RNN(input_size=5, hidden_size=10, num_layers=2, output_size=args.output_size)
    trainer = Supervisor(model, args.epochs, args.batch_size, torch.optim.Adam, args.lr, nn.CrossEntropyLoss(), args.horizon)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--horizon", default=5, type=int)

    parser.add_argument("--file-span", default=10, type=int)
    parser.add_argument("--seq-len", default=12, type=int)
    parser.add_argument("--hidden-size", default=10, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--output-size", default=6, type=int)

    args = parser.parse_args()

    main(args)