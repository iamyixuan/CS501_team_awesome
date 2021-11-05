import sklearn.metrics

def evaulation_matrix(predicted, actual):
    #Both inputs are arrays

    #turn predicated into a binary list
    pred_bin = []
    threshold = .5
    for i in predicted:
        if i >= threshold:
            pred_bin.append(1)
        else:
            pred_bin.append(0)

    #Confusion Matrix
    conMat = sklearn.metrics.confusion_matrix(actual, pred_bin)
    print(conMat)
    TP = conMat[0,0]
    FN = conMat[1,0]
    FP = conMat[0,1]
    TN = conMat[1,1]

    accuracy = ()



#Run code
actual = [1,1,1,1,1,0,0,0,0,0]
predicted = [.6,.7,.8,.1,.2,.8,.1,.1,.2,.5]
actual = [0]
predicted = [0.1]

evaulation_matrix(predicted, actual)

