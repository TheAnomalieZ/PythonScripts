import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math as mm
import csv as C
import time
from sklearn.externals import joblib
import sys

num = int(sys.argv[1])
print num
np.random.seed(42)
f = open('train6.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

train_data = np.array(floats)
model = hmm.MultinomialHMM(n_components=num)
model.fit(train_data)
val = model.monitor_.converged
print val
scorelist = []
#1
test_data1 = np.array([[0], [2], [0], [2], [2], [0], [0], [0], [2], [2]])
value = model.score(test_data1)
scorelist.append(value)
#2
test_data1 = np.array([[2], [0], [2], [2], [0], [0], [0], [2], [2], [1]])
value = model.score(test_data1)
scorelist.append(value)
#3
test_data1 = np.array([[0], [2], [2], [0], [0], [0], [2], [2], [1], [1]])
value = model.score(test_data1)
scorelist.append(value)
#4
test_data1 = np.array([[2], [2], [0], [0], [0], [2], [2], [1], [1], [0]])
value = model.score(test_data1)
scorelist.append(value)
#5
test_data1 = np.array([[2], [0], [0], [0], [2], [2], [1], [1], [0], [0]])
value = model.score(test_data1)
scorelist.append(value)
#6
test_data1 = np.array([[0], [0], [0], [2], [2], [1], [1], [0], [0], [2]])
value = model.score(test_data1)
scorelist.append(value)
#7
test_data1 = np.array([[0], [0], [2], [2], [1], [1], [0], [0], [2], [0]])
value = model.score(test_data1)
scorelist.append(value)
#8
test_data1 = np.array([[0], [2], [2], [1], [1], [0], [0], [2], [0], [2]])
value = model.score(test_data1)
scorelist.append(value)
#9
test_data1 = np.array([[2], [2], [1], [1], [0], [0], [2], [0], [2], [2]])
value = model.score(test_data1)
scorelist.append(value)
#10
test_data1 = np.array([[2], [1], [1], [0], [0], [2], [0], [2], [2], [1]])
value = model.score(test_data1)
scorelist.append(value)

anomalyscore = []
count = 1;
end = len(scorelist)
temp = scorelist[0]
while count < end:
    val = temp - scorelist[count]
    print val
    temp = scorelist[count]
    count = count + 1

'''scorelist2 = []
#1
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [2], [0]])
value = model.score(test_data1)
scorelist2.append(value)
#2
test_data1 = np.array([[1], [1], [1], [2], [2], [0], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#3
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [2]])
value = model.score(test_data1)
scorelist2.append(value)
#4
test_data1 = np.array([[0], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#5
test_data1 = np.array([[1], [1], [1], [2], [0], [1], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#6
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [2], [0]])
value = model.score(test_data1)
scorelist2.append(value)
#7
test_data1 = np.array([[1], [1], [1], [2], [0], [1], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#8
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [2], [0], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#9
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)
#10
test_data1 = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]])
value = model.score(test_data1)
scorelist2.append(value)

hstatelist = []
for i in range(1, 11):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist)
plt.plot(hstatelist,scorelist2)
plt.xlabel('HMM Model')
plt.ylabel('Probability')
plt.title('Evaluation')
plt.show()
'''
