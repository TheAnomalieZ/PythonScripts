import numpy as np
import math as mm
import csv as C
import time

from hmmlearn import hmm
from sklearn.externals import joblib

def readFile(train):
    inputfile = open(train)
    try:
        reader = C.reader(inputfile)
        values = []
        for row in reader:
            values.append(map(int, row))
    finally:
        inputfile.close()

    data_set = np.array(values)
    return data_set;

def trainModel(train_data, n):
    modellist = []
    for num in range(2, n):
        time1 = time.time()
        model = hmm.MultinomialHMM(n_components = num)
        model.fit(train_data)
        modellist.append(model)
        time2 = time.time()
        print "Loop " + str(num-2) + " takes " + str(time2 - time1)
    print "Model training Finished"
    i = 0
    for mod in modellist:
        joblib.dump(mod, "modelname" + str(i) + ".pkl")
        i = i+1
    print "Training Success"
    return modellist;

def findModel(test_data, models):
    modellist = models
    """for num in range(2, 8):
        model = hmm.MultinomialHMM(n_components = num)
        model.fit(train_data)
        modellist.append(model)"""

    scorelist = []

    score1 = modellist[0].score(test_data)
    scorelist.append(score1)
    finalModel = modellist[0]
    for num in range(2, 7):
        value = modellist[num-1].score(test_data)
        if(score1 < value):
            finalModel = modellist[num-1]
            score1 = value
        scorelist.append(value)

    for val in scorelist:
        print val
    i = 0
    for mod in modellist:
        joblib.dump(mod, "modelname" + str(i) + ".pkl")
        i = i+1
    print "Model Found"
    return finalModel;

train_data = readFile('train.csv')
test_data = readFile('test.csv')
modellist = trainModel(train_data, 10)
#modelne = findModel(test_data, modellist)
