import numpy as np
print('Loading 1st part')
X = np.load('X_triggerword_1.npy')
Y = np.load('Y_triggerword_1.npy')

print('Loading 2nd part')
X1 = np.load('X_triggerword_2.npy')
Y1 = np.load('Y_triggerword_2.npy')

print('Concatenating')
X = np.concatenate((X, X1), axis=0)
Y = np.concatenate((Y, Y1), axis=0)

print(X.shape)
print(Y.shape)

print('Saving')
np.save('X_triggerword.npy', X)
np.save('Y_triggerword.npy', Y)