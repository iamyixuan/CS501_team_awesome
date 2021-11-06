import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

f = open("metrics_out.txt","a")
pred_val=np.genfromtxt("results.txt", dtype=float64, encoding=None, delimiter=",")
true_val=np.genfromtxt("y_vals.txt", dtype=float64, encoding=None, delimiter=",")

acc=balanced_accuracy_score(true_val, pred_val)

f1=f1_score(true_val, pred_val, average=None)

prec=average_precision_score(true_val, pred_val)

roc=roc_auc_score(true_val, pred_val)
display = RocCurveDisplay.from_predictions(true_val, pred_val)
plt.plot(display)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('%.3f,%.3f,%.3f,%.3f' % (acc,f1,prec,roc))
print('%.3f,%.3f,%.3f,%.3f' % (acc,f1,prec,roc), file=f)

return
