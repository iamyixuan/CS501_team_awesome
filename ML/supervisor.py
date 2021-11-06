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
        f = open("output.txt", "a")
        num_iter = len(train_loader)
        print("Start training...")
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = []
            for train in train_loader:
                train_batch = train
                x_train = torch.cat((train_batch["cloud_1"], train_batch["cloud_2"], train_batch["cloud_3"],
                                    train_batch["temperature"], train_batch["dew_point"]), axis=2) 
                y_train = train_batch["precipitation"]
                self.optimizer.zero_grad()
                out = self.model(x_train)
                loss = self.loss_fn(out, y_train)
                running_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
            
            self.model.eval()
            for val in val_loader:
                val_data = val
                x_val = torch.cat((val_data["cloud_1"], val_data["cloud_2"], val_data["cloud_3"],
                                    val_data["temperature"], val_data["dew_point"]), axis=2) 
                y_val = val_data["precipitation"] # ground truth
                pred_val = self.model(x_val) # model output
                val_loss = self.loss_fn(pred_val, y_val).item()
                break
                
            train_loss = np.mean(running_loss)

            print("Epoch %s: training loss is %.3f, validiation loss is %.3f" % (epoch, train_loss, val_loss))
            print('%s,%.3f,%.3f' % (epoch, train_loss, val_loss), file=f)
        np.savetxt("pred_vals.txt", pred_val.detach().numpy())
        np.savetxt("y_vals.txt", y_val.detach().numpy())
    
    def predict(self, test_loader):
        pass    
