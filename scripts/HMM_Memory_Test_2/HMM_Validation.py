import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import sys
from HMMModel import HMMModel

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
n3 = int(sys.argv[3])

np.random.seed(42)
hmm = HMMModel()
model, listdata = hmm.trainingHMM('normal_final.csv', n1, n2)

scorelist0n = hmm.testingHMM('normal_states.csv',n2, n3, model, listdata)
scorelist0 = hmm.prepareData(scorelist0n, 15)
scorelist2n = hmm.testingHMM('anomaly_validation_2',n2, n3, model, listdata)
scorelist2 = hmm.prepareData(scorelist2n, 15)
print len(scorelist0), len(scorelist2)
numb = len(scorelist0)
hstatelist = []
for i in range(0, numb):
    hstatelist.append(i)

plt.plot(hstatelist,scorelist0, label = 'Normal')
plt.plot(hstatelist,scorelist2, label = 'Anomaly_Validation')

plt.xlabel('Sequence')
plt.ylabel('Probability')
#plt.title('HS = ' + str(n1) + ',    WS = ' + str(n2) + ',   Acc = ' + str(acc) + ',    DR = ' + str(dt) + ',    FPA = ' + str(fpr))
plt.legend()
plt.show()
