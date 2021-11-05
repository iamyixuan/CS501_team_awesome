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
data = np.genfromtxt("results.txt", dtype=int64, encoding=None, delimiter=",")
pred_val=data(:,0)
true_val=data(:,1)

acc=balanced_accuracy_score(true_val, pred_val)

f1=f1_score(true_val, pred_val, average=None)

prec=average_precision_score(true_val, pred_val)

#ROC using non threshold decisions method
#roc=roc_auc_score(y, clf.decision_function(pred_val))
# ROC using probability estimates method
clf = LogisticRegression(solver="liblinear", random_state=0).fit(pred_val, true_val)
roc=roc_auc_score(true_val, clf.predict_proba(pred_val)[:, 1])
display = RocCurveDisplay.from_predictions(true_val, clf.predict_proba(pred_val)[:, 1])
plt.plot(display)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

print('%.3f,%.3f,%.3f,%.3f' % (acc,f1,prec,roc))
print('%.3f,%.3f,%.3f,%.3f' % (acc,f1,prec,roc), file=f)

return
