import commandsmodel
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model = commandsmodel.commandsmodel()

X_train = np.load('X_commands.npy')
Y_train = np.load('Y_commands.npy')

X_test = np.load('X_test_commands.npy')
Y_test = np.load('Y_test_commands.npy')
X_test = X_test.reshape(-1, 8000, 1)

es = EarlyStopping(monitor='val_acc', mode='max', patience=10, verbose=1)
mc = ModelCheckpoint('best_commands_model_GRU_2.h5', monitor='val_acc', save_best_only=True, verbose=1, mode='max')

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
history = model.fit(X_train, Y_train, callbacks=[es, mc], validation_data=[X_test, Y_test], batch_size=32, epochs=100)

# accuracy graph
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('commands_accuracy_graph.png')

# loss graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('commands_loss_graph.png')