import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math as mm
import csv as C

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

modellist = []

for num in range(15, 50):
    model = hmm.MultinomialHMM(n_components=num)
    model.fit(train_data)
    modellist.append(model)

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
for num in range(15, 50):
    value = modellist[num-15].score(test_data)
    #val = mm.exp(value* mm.log(10.))
    scorelist.append(value)


hstatelist = []
for i in range(15, 50):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist)
plt.xlabel('HMM Model')
plt.ylabel('Probability')
plt.title('Evaluation')
plt.show()
