import torch
import numpy as np
import sys
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def metrics(true_name,pred_name,train_y_name,train_pred_name,threshold):
    f = open("C:/Users/pjfar/Documents/CS_501/Project/CS501_team_awesome_new/CS501_team_awesome/Output Files/Metrics/metrics_out.txt","a")
    
    path="C:/Users/pjfar/Documents/CS_501/Project/CS501_team_awesome_new/CS501_team_awesome/Output Files/Model Outputs/"
    true_val = np.genfromtxt(path+true_name,dtype='f')
    pred_val = np.genfromtxt(path+pred_name,dtype='f')
    train_pred=np.genfromtxt(path+train_pred_name,dtype='f')
    train_y=np.genfromtxt(path+train_y_name,dtype='f')

    pred_bin = []
    for i in pred_val:
        if i >= threshold:
            pred_bin.append(1)
        else:
            pred_bin.append(0)

    train_pred_bin = []
    for i in train_pred:
        if i >= threshold:
            train_pred_bin.append(1)
        else:
            train_pred_bin.append(0)

    val_conMat=confusion_matrix(true_val, pred_bin)
    print(f'Confusion Matrix:\n {val_conMat}')
    val_tn= val_conMat[0][0]
    val_fn= val_conMat[1][0]
    val_fp= val_conMat[0][1]
    val_tp= val_conMat[1][1]
    
    train_conMat=confusion_matrix(train_y,train_pred_bin)
    print(f'Confusion Matrix:\n {train_conMat}')
    train_tn= train_conMat[0][0]
    train_fn= train_conMat[1][0]
    train_fp= train_conMat[0][1]
    train_tp= train_conMat[1][1]
    
    val_recall=recall_score(true_val, pred_bin)
    val_acc=balanced_accuracy_score(true_val, pred_bin)
    val_f1=f1_score(true_val, pred_bin, average='binary')
    val_prec=average_precision_score(true_val, pred_bin)

    train_recall=recall_score(train_y,train_pred_bin)
    train_acc=balanced_accuracy_score(train_y,train_pred_bin)
    train_f1=f1_score(train_y,train_pred_bin, average='binary')
    train_prec=average_precision_score(train_y,train_pred_bin)
    
    val_roc=roc_auc_score(true_val, pred_bin)
    train_roc=roc_auc_score(train_y, train_pred_bin)

    figax=plt.axes()
    RocCurveDisplay.from_predictions(true_val, pred_bin,label='Val_AUC='+str(round(val_roc,3)),ax=figax)
    RocCurveDisplay.from_predictions(train_y, train_pred_bin,label='Train_AUC='+str(round(train_roc,3)),ax=figax)
    plt.plot([0,1],[0,1],'--',label='Y=X-line')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.title('ROC Curve')
    plt.grid()
    plt.show()

    print('Validation Accuracy, Recall, F1, Precision, ROC, tn, fn, fp, and tp are:\n %.3f, %.3f, %.3f, %.3f, %.3f, %.f, %.f, %.f, and %.f respectively' % (val_acc,val_recall,val_f1,val_prec,val_roc,val_tn,val_fn,val_fp,val_tp))
    print('Training Accuracy, Recall, F1, Precision, ROC, tn, fn, fp, and tp are:\n %.3f, %.3f, %.3f, %.3f, %.3f, %.f, %.f, %.f, and %.f respectively' % (train_acc,train_recall,train_f1,train_prec,train_roc,train_tn,train_fn,train_fp,train_tp))
    print('%.3f,%.3f,%.3f,%.3f,%.3f,%.f,%.f,%.f,%.f' % (val_acc,val_recall,val_f1,val_prec,val_roc,val_tn,val_fn,val_fp,val_tp), file=f)
    print('%.3f,%.3f,%.3f,%.3f,%.3f,%.f,%.f,%.f,%.f' % (train_acc,train_recall,train_f1,train_prec,train_roc,train_tn,train_fn,train_fp,train_tp), file=f)

    return

if __name__ == "__main__":
    metrics(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],float(sys.argv[5]))
