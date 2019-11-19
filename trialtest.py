import numpy as np
from tensorflow.keras.models import load_model
# from sklearn.metrics import accuracy_score, confusion_matrix

model1 = load_model('best_commands_model.h5')
model2 = load_model('best_commands_model_GRU.h5')

X_test = np.load('X_test_commands.npy')
Y_test = np.load('Y_test_commands.npy')
X_test = X_test.reshape(-1, 8000, 1)

Y_pred1 = model1.predict(X_test)
Y_pred2 = model2.predict(X_test)

# cm1 = confusion_matrix(y_true=Y_test, y_pred=Y_pred1)
# cm2 = confusion_matrix(y_true=Y_test, y_pred=Y_pred2)

# print('Confusion Matrix')
# print('for model 1: {}'.format(cm1))
# print('for model 2: {}'.format(cm2))
# print('accuracy score')
# print('for model 1: {}'.format(accuracy_score(y_true=Y_test, y_pred=Y_pred1)))
# print('for model 2: {}'.format(accuracy_score(y_true=Y_test, y_pred=Y_pred2)))