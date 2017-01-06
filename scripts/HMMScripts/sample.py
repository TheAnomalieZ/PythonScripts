import numpy as np
from hmmlearn import hmm

np.random.seed(42)


X = np.array([[1], [0], [1], [1], [1], [1], [1], [1],[3],[1],[3],[2],[1],[3],[3],[3],[1],[1],[3],[3],[1],[3],[1],[3],[3],[1],[1],[3],[3],[1],[1],[3],[3],[1],[1],[1],[1],[1],[3],[3],[1],[3],[3],[1],[3],[3],[1],[3],[1],[3],[2],[1],[3],[1],[3],[3],[1],[1],[3],[3],[1],[1],[1],[3],[3],[1],[1],[3],[1],[3],[1],[3],[3],[3],[1],[3],[1],[3],[1],[3],[1],[3],[3],[1],[3],[1],[3],[3],[3],[1],[3],[1],[3],[3],[1],[3],[3],[3],[3],[1],[1],[3],[1],[1],[3],[1],[3],[3],[1],[3],[1],[3],[1],[3],[3],[1],[3],[1],[3],[3],[3],[1],[3],[1],[1],[3],[1],[3],[1],[3],[1],[1],[3],[1],[1],[1],[3],[3],[1],[1],[3],[1],[3],[1],[1],[3],[3],[1],[1],[3],[1],[1],[1],[3],[3],[3],[1],[1],[3],[1],[3],[3],[1],[3],[1],[3],[1],[3],[3],[1],[1],[3],[1],[1],[3],[1],[3],[1],[3],[1],[3],[1],[3],[3],[1],[3],[1],[3],[1],[3],[3],[3],[2],[1],[1],[3],[1],[3],[1],[3],[1],[1],[3],[1],[3],[3],[1],[3],[1],[1],[3],[3],[3],[1],[1],[3],[3],[3],[1]])
# tlen = np.array([200])
# start_probability = np.array([0.6, 0.4])
#
# transition_probability = np.array([
#   [0, 0],
#   [0, 0]
# ])
#
# emission_probability = np.array([
#   [0, 0, 0],
#   [0, 0, 0]
# ])
#
model = hmm.MultinomialHMM(n_components=2)
# model.startprob_ = start_probability
# model.transmat_  = transition_probability
# model.emissionprob_ = emission_probability

# model = hmm.MultinomialHMM(n_components=6,n_iter=1000).fit(X)
model.fit(X)

# Predict the optimal sequence of internal hidden state
# hidden_states = model.predict(X)
# X1 = np.array([[1], [2]])

# tlen = len(X1)
# Y = model.score(X1)

print("done")
print("Transition matrix")
print(model.transmat_)
print(model.startprob_)
print(model.emissionprob_)

# print(Y)
