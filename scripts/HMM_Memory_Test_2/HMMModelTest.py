import numpy as np
# from hmmlearn import hmm
import matplotlib.pyplot as plt
# import math as mm
# import csv as C
# import time
# from linked_list import LinkedList
from sklearn.externals import joblib
import sys
from HMMModel import HMMModel

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
n3 = int(sys.argv[3])

np.random.seed(42)
hmm = HMMModel()
model, listdata = hmm.trainingHMM('no_big_states.csv', n1, n2)
# joblib.dump(model, "model1" + ".pkl")

for i in range(2, 7):
    values = []
    size = i*5
    scorelist2 = hmm.testingHMM('anomaly_validation_2',size, n3, model, listdata)
    labellist1 = hmm.labeling(220,270, scorelist2)
    labellist11 = hmm.relabeling(305, 345, labellist1)
    labellist12 = hmm.relabeling(388, 397, labellist11)
    roc_x, roc_y = hmm.drawROC(scorelist2, labellist12, size)
    values = hmm.areaROC1(scorelist2, labellist12)
    print values

# for i in range(1, 10):
#     values = []
#     size = i*5
#     scorelist2 = hmm.testingHMM('anomaly_final_states.csv',size, n3, model, listdata)
#     labellist1 = hmm.labeling(445,595, scorelist2)
#     labellist11 = hmm.relabeling(1095, 1365, labellist1)
#     labellist12 = hmm.relabeling(1880, 2000, labellist11)
#     roc_x, roc_y = hmm.drawROC(scorelist2, labellist12, size)
#     values = hmm.areaROC1(scorelist2, labellist12)
#     print values


# size = 10
# scorelist2 = hmm.testingHMM('anomaly_final_states1.csv', size, n3, model)
# #     #scorelist2 = hmm.testingHMM('ano3_states.csv',size, n3, model)
# labellist1 = hmm.labeling(460,600, scorelist2)
# #     #labellist1n = hmm.relabeling(1110,)
# #
# roc_x, roc_y = hmm.drawROC(scorelist2, labellist1, size)
# print roc_x
# print "================================================================================="
# print roc_y
#     values = hmm.areaROC1(scorelist2, labellist1)
#     print values

# for i in range(1, 9):
#     states = i + 1
#     model = hmm.trainingHMM('no_big_states.csv', states)
#     scorelist2 = hmm.testingHMM('anomaly_final_states1.csv', n2, n3, model)
#     #scorelist2 = hmm.testingHMM('ano3_states.csv',size, n3, model)
#     labellist1 = hmm.labeling(460,600, scorelist2)
#     #labellist1n = hmm.relabeling(1110,)
#
#     hmm.drawROC(scorelist2, labellist1, states)
#     values = hmm.areaROC1(scorelist2, labellist1)
#     print values
#
# labels = hmm.readData('labellist.csv')
# numb = len(labels)
# #graph
# hstatelist = []
# for i in range(0, numb):
#     hstatelist.append(i)
#
#
# plt.plot(hstatelist, labels)

plt.xlim(-1, 101)
plt.ylim(-1, 101)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Graph')
plt.legend()
plt.show()
