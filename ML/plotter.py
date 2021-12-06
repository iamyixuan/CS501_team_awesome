import numpy as np
import matplotlib.pyplot as plt
import sys

def train_val_plotter(file_name):
    path='C:/Users/pjfar/Documents/CS_501/Project/CS501_team_awesome_new/CS501_team_awesome/Output Files/Trainer Evaluation/'
    epoch,train_loss,val_loss=np.loadtxt(path+file_name,delimiter=',',unpack=True)
    plt.figure()
    plt.plot(epoch,train_loss,label='Training')
    plt.plot(epoch,val_loss,label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend(loc=0)
    plt.xlim(left=0,right=len(epoch))
    plt.grid()
    plt.show()

    return

if __name__=='__main__':
    train_val_plotter(sys.argv[1])
