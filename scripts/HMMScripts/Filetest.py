import numpy as np
import csv as C
#f = open('values.txt', 'r')
#x = f.read()
#poly_shape = []
#poly_shape.append(x)
#y = np.array(poly_shape)

#print(y)

# with open('output2.txt') as f:
#     polyShape = []
#     y = []
#     for word in f:
#         word = word.split() # to deal with blank
#         if word:            # lines (ie skip them)
#             word = [int(i) for i in word]
#             #np.append(y,word)
#             #y.reshape(-1,1)
#             polyShape.append(word)
# X = np.array(polyShape)
#X = np.array([[2], [3],[3],[3],[1],[3],[1],[3],[2],[2],[3],[1]])

#X.reshape(1,-1)
##print(X)

np.random.seed(42)
f = open('filetest.csv')
try:
    reader = C.reader(f)
    floats = []
    for row in reader:
        floats.append(map(int, row))
finally:
    f.close()

train_data = np.array(floats)
#print(train_data)

test_data1 = np.array([[0], [2], [0], [0], [0], [2], [2], [2]])
X1 = [[0], [2], [0], [0], [0], [2], [2], [2]]
X2 = [[1], [2], [0], [1], [0], [1], [2], [0]]
X = np.concatenate([X1, X2])
#length = [len[X1], len[X2]]
test_data1.reshape(-1,1)
print(X)
