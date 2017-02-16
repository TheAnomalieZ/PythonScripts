import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math as mm
import csv as C
import time
from linked_list import LinkedList
from sklearn.externals import joblib
import sys

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
np.random.seed(42)

#training part
f = open('train2.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

train_data = np.array(floats)
model = hmm.MultinomialHMM(n_components=n1)
model.fit(train_data)

def testing(filename, n):
    scorelist = []
    f = open(filename)
    try:
        reader = C.reader(f)
        floats = []
        for row in reader:
            floats.append(map(int, row))
    finally:
        f.close()

    test_data = np.array(floats)
    #print test_data
    testlist = LinkedList()
    start = False
    count = 0
    numb1 = 0
    for data in test_data:
        testlist.appendLast(data)
        count = count + 1
        if count == n:
            start = True
        if start:
            data_list = []
            data_list = testlist.printList()
            final_list = np.array(data_list)
            #print final_list
            value = model.score(np.array(data_list))
            scorelist.append(value)
            testlist.deleteHead()
            numb1 = numb1 + 1
    return scorelist


scorelist1 = testing('testnormal1.csv',n2)
scorelist2 = testing('anomaly2.csv',n2)
scorelist3 = testing('anomaly3.csv',n2)
scorelist4 = testing('anomaly4.csv',n2)
scorelist5 = testing('anomaly5.csv',n2)

numb = len(scorelist1)
#graph
hstatelist = []
for i in range(0, numb):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist1, label = 'Normal')
plt.plot(hstatelist,scorelist2, label = 'Anomaly1')
plt.plot(hstatelist,scorelist3, label = 'Anomaly2')
plt.plot(hstatelist,scorelist4, label = 'Anomaly3')
plt.plot(hstatelist,scorelist5, label = 'Anomaly4')

plt.xlabel('HMM Model')
plt.ylabel('Probability')
plt.title('Evaluation')
plt.legend()
plt.show()
