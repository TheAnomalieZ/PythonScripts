import matplotlib.pyplot as plt
from sklearn.externals import joblib
import sys
from HMMModel import HMMModel

n1 = int(sys.argv[1])
n2 = int(sys.argv[2])
n3 = int(sys.argv[3])

np.random.seed(42)
hmm = HMMModel()
model = hmm.trainingHMM('no_big.csv', n1)

scorelistNormal = hmm.testingHMM('normal_states1.csv',n2, 4529, model)
scorelist0 = hmm.testingHMM('anomaly3', n2, 1432, model)
scorelist1 = hmm.testingHMM('anomaly_final', n2, 4030, model)
scorelist2 = hmm.testingHMM('anomaly_validation_2',n2, 440, model)
# anomaly3 labels
labellist03 = hmm.labeling(125,265, scorelist0)
labellist02 = hmm.relabeling(280,390, labellist03)
labellist01 = hmm.relabeling(540,665, labellist02)
labellist0 = hmm.relabeling(745,1432, labellist01)
# anomaly final labels
labellist16 = hmm.labeling(440,590, scorelist1)
labellist15 = hmm.relabeling(1095,1355, labellist16)
labellist14 = hmm.relabeling(1885,2005, labellist15)
labellist13 = hmm.relabeling(2650,2860, labellist14)
labellist12 = hmm.relabeling(3370,3470, labellist13)
labellist1 = hmm.relabeling(3810,3850, labellist12)
#anomaly validation 2 labels
labellist2 = hmm.labeling(285,320, scorelist2)

threshold = hmm.generateThreshold(scorelistNormal, 30)

dR_list = []
percent_list = []

percent_list.append(hmm.anomalyPercent(labellist0))
percent_list.append(hmm.anomalyPercent(labellist1))
percent_list.append(hmm.anomalyPercent(labellist2))

dR_list.append(hmm.detectionRate(scorelist0, labellist0, threshold))
dR_list.append(hmm.detectionRate(scorelist1, labellist1, threshold))
dR_list.append(hmm.detectionRate(scorelist2, labellist2, threshold))

print percent_list
print dR_list
