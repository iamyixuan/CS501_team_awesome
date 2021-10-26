import numpy as np
import torch

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
        
    def train(self, train_loader, val_loader):
        num_iter = len(train_loader)
        print("Start training...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = []
            for train in train_loader:
                train_batch = train
                x_train = train_batch["cloud_1"] 
                y_train = train_batch["precipitation"].view(-1,)
                self.optimizer.zero_grad()
                out = self.model(x_train)
                loss = self.loss_fn(out, y_train)
                running_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            for val in val_loader:
                val_data = val
                x_val = val_data["cloud_1"]
                y_val = val_data["precipitation"].view(-1, )
                pred_val = self.model(x_val)
                val_loss = self.loss_fn(pred_val, y_val).item()
                break
                
            train_loss = np.mean(running_loss)
            print("Epoch %s: training loss is %.3f, validiation loss is %.3f" % (epoch, train_loss, val_loss))

    
    def predict(self, test_loader):
        pass    