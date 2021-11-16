import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from metrics import *

class Supervisor:
    """
    Supervised learning training class
    Methods:
        train: train the model with specified number of epochs, batch size, optimizer, learning rate, loss function, and prediction horizon
        predict: make predictions using the trained model. 
    """
    def __init__(self, model, epochs, batch_size, optimizer, lr, loss_fn, horizon) -> None:
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.horizon = horizon
        self.loss_fn = loss_fn
        self.lr = lr
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', verbose=True, factor=.5, patience=30)
        
    def train(self, train_loader, val_loader):
        f = open("output.txt", "a")
        num_iter = len(train_loader)
        print("Start training...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = []
            running_f1 = []
            for x_train, y_train in train_loader:
                self.optimizer.zero_grad()
                out = self.model(x_train)
                loss = self.loss_fn(out, y_train)
                f1 = f1_score(y_train.detach().numpy(), np.where(out.detach().numpy() > 0.5, 1, 0))
                running_loss.append(loss.item())
                running_f1.append(f1)
                loss.backward()
                self.optimizer.step()

            self.model.eval()
            for x_val, y_val in val_loader:
                pred_val = self.model(x_val) # model output
                val_loss = self.loss_fn(pred_val, y_val)
                val_f1 = f1_score(y_val.detach().numpy(), np.where(pred_val.detach().numpy()> 0.5, 1, 0))

                
            train_loss = np.mean(running_loss)
            train_f1 = np.mean(running_f1)
            self.scheduler.step(val_loss)

            print("Epoch %s: training loss is %.3f, validiation loss is %.3f; training f1 %.3f val f1 %.3f" % (epoch, train_loss, val_loss.item(), train_f1, val_f1))
            print('%s,%.3f,%.3f' % (epoch, train_loss, val_loss.item()), file=f)
        np.savetxt("pred_vals.txt", pred_val.detach().numpy())
        np.savetxt("y_vals.txt", y_val.detach().numpy())
    
    def predict(self, test_loader):
        pass    
