import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import math as mm
import csv as C
import time
from linked_list import LinkedList
from sklearn.externals import joblib
import sys

n = int(sys.argv[1])
print n

np.random.seed(42)
'''model = hmm.MultinomialHMM(n_components=n)
f = open('train6.csv')
try:
    reader = C.reader(f)
    count = 0;
    numb = 0;
    floats = []
    for row in reader:
        floats.append(map(int, row))
        count = count + 1
        if count % 100 == 0:
            numb = numb + 1
            train_data = np.array(floats)
            model.fit(train_data)
            floats = []
finally:
    f.close()

print numb
print count
'''
f = open('train2.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

train_data = np.array(floats)
model = hmm.MultinomialHMM(n_components=n)
model.fit(train_data)
'''modellist = []
for num in range(2, n):
    time1 = time.time()
    model = hmm.MultinomialHMM(n_components=num)
    model.fit(train_data)
    val = model.monitor_.converged
    modellist.append(model)
    time2 = time.time()
    print val
    print "Loop  " + str(num-2) + " takes "+ str(time2 - time1)
'''
#Normal
scorelist = []
'''f = open('test4.csv')
try:
    reader = C.reader(f)
    count = 0;
    numb = 0;
    floats = []
    for row in reader:
        floats.append(map(int, row))
        count = count + 1
        if count % 15 == 0:
            numb = numb + 1
            test_data = np.array(floats)
            value = model.score(test_data)
            scorelist.append(value)
            floats = []

finally:
    f.close()
'''
f = open('testnormal2.csv')
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
numb = 0
for data in test_data:
    testlist.appendLast(data)
    count = count + 1
    if count == 10:
        start = True
    if start:
        data_list = []
        data_list = testlist.printList()
        final_list = np.array(data_list)
        print final_list
        value = model.score(np.array(data_list))
        scorelist.append(value)
        testlist.deleteHead()
        numb = numb + 1
print count
print testlist.size()
#Anomaly
scorelist1 = []
f = open('testanomaly1.csv')
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
numb = 0
for data in test_data:
    testlist.appendLast(data)
    count = count + 1
    if count == 10:
        start = True
    if start:
        data_list = []
        data_list = testlist.printList()
        final_list = np.array(data_list)
        print final_list
        value = model.score(np.array(data_list))
        scorelist1.append(value)
        testlist.deleteHead()
        numb = numb + 1
print count
print testlist.size()


'''scorelist1 = []
f = open('test5.csv')
try:
    reader = C.reader(f)
    count = 0;
    floats = []
    for row in reader:
        floats.append(map(int, row))
        count = count + 1
        if count % 15 == 0:
            test_data = np.array(floats)
            value = model.score(test_data)
            scorelist1.append(value)
            floats = []

finally:
    f.close()
'''
print numb
#
# test_data1 = np.array(floats)
#
# #test_data1 = np.array([[1], [0], [0], [2], [1], [2], [3], [1]])
# #test_data2 = np.array([[0], [0], [1], [0], [1], [2], [0], [1]])
#
#
#
# for num in range(2, n):
#     value = modellist[num-2].score(test_data)
#     #val = np.exp(value)
#     #print val
#     scorelist.append(value)
#     print "done"
#
# scorelist1 = []
# for num in range(2, n):
#     value = modellist[num-2].score(test_data1)
#     #val = np.exp(value)
#     #print val
#     scorelist1.append(value)
#     print "done"

hstatelist = []
for i in range(0, numb):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist, label = "Normal")
plt.plot(hstatelist,scorelist1, 'ro', label = "Anomaly")
plt.xlabel('HMM Model')
plt.ylabel('Probability')
plt.title('Evaluation')
plt.show()
