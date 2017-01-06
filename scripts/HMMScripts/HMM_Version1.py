import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math as mm
import csv as C
import time
from sklearn.externals import joblib

np.random.seed(42)
f = open('train.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

train_data = np.array(floats)
"""
model1 = hmm.MultinomialHMM(n_components = 40)
model1.fit(train_data)"""
modellist = []
print (time2-time1)
for num in range(2, 10):
    time1 = time.time()
    model = hmm.MultinomialHMM(n_components=num)
    model.fit(train_data)
    modellist.append(model)
    time2 = time.time()
    print "Loop 1 "+ "takes "+ str(time2 - time1)

f = open('test.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

test_data = np.array(floats)

scorelist = []
for num in range(2, 10):
    value = modellist[num-15].score(test_data)
    val = mm.exp(value* mm.log(10.))
    scorelist.append(value)
    print "done"

hstatelist = []
for i in range(15, 30):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist)
plt.xlabel('HMM Model')
plt.ylabel('Probability')
plt.title('Evaluation')
plt.show()
