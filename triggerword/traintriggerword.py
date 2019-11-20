import triggerwordmodel
# import triggerword
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

input_shape = (5511, 101)
model = triggerwordmodel.trigger_model(input_shape)

print('Loading data')
X = np.load('preprocessing/X_triggerword.npy')
Y = np.load('preprocessing/Y_triggerword.npy')

print('Swapping axis')
X_train = np.swapaxes(X, 1, 2)
Y_train = np.swapaxes(Y, 1, 2)

es = EarlyStopping(monitor='acc', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('best_trigger_model_TM_50lr0.1.h5', monitor='acc', mode='max', verbose=1, save_best_only=True)

print('Splitting and training')
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
history = model.fit(X_train, Y_train, callbacks=[es, mc], batch_size=32, epochs=50)

# accuracy graph
plt.plot(history.history['acc'])
plt.title('Model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.savefig('triggerword_accuracy_graph.png')

# loss graph
plt.plot(history.history['loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train'], loc='upper left')
plt.savefig('triggerword_loss_graph.png')