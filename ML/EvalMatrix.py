import sklearn.metrics
import matplotlib.pyplot as plt

def evaluationMatrix(predicted, actual, threshold):
    #Both inputs are arrays

    #turn predicated into a binary list
    pred_bin = []
    for i in predicted:
        if i >= threshold:
            pred_bin.append(1)
        else:
            pred_bin.append(0)

    #Confusion Matrix
    conMat = sklearn.metrics.confusion_matrix(actual, pred_bin)
    print('Confusion Matrix')
    print(conMat)

    print('Accuracy: ', sklearn.metrics.accuracy_score(actual,pred_bin))
    print('Precision: ', sklearn.metrics.precision_score(actual,pred_bin))
    print('recall: ', sklearn.metrics.recall_score(actual,pred_bin))
    print('F1 score: ', sklearn.metrics.f1_score(actual,pred_bin))


#
# def plot_roc_curve(fpr, tpr):
#     plt.plot(fpr, tpr, color='orange', label='ROC')
#     plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend()
#     plt.show()
#     #https://stackabuse.com/understanding-roc-curves-with-python/




#Run code
actual = [1,0,0,1,0,0,1,0,0,1]
predicted = [1,0,0,1,0,0,0,1,0,0]
# actual = [0]
# predicted = [0.1]

evaluationMatrix(predicted, actual, .5)

#DID ANYTHING CHANGE

