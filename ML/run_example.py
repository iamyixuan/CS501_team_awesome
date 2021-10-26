import argparse
import numpy as np
import torch.nn as nn
import torch

from models import RNN
from data import WeatherDataset
from torch.utils.data import DataLoader
from supervisor import Supervisor

def main(args):
    dataset = WeatherDataset("../../Processed Data/", 3, 12, 5)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=10)

    model = RNN(1, 3, 2, 6)
    trainer = Supervisor(model, args.epochs, args.batch_size, torch.optim.Adam, args.lr, nn.CrossEntropyLoss(), args.horizon)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--horizon", default=5, type=int)

    args = parser.parse_args()

    main(args)