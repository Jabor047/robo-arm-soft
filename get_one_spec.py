import utilities
import numpy as np
import matplotlib.pyplot as plt

X = utilities.graph_spectogram('train52311.wav')
# Y = np.load('Y.npy')

print(X.shape)
# plt.plot(Y)
plt.show()