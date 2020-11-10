import commandsmodelPlusFB
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler


model = commandsmodelPlusFB.commandsmodel()
AttModel = commandsmodelPlusFB.attRNN()

X = np.load('preprocessing/X_commands.npy')
YOneHot = np.load('preprocessing/YOnehot_commands.npy')
Y = np.load('preprocessing/Y_commands.npy')


# X_test = np.load('preprocessing/X_test_commands.npy')
# Y_test = np.load('preprocessing/Y_test_commands.npy')
# X_test = X_test.reshape(-1, 8000, 1)

def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.4
    epochs_drop = 15.0
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    if (lrate < 4e-5):
        lrate = 4e-5

    print('Changing learning rate to {}'.format(lrate))
    return lrate

lrate = LearningRateScheduler(step_decay)

es = EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)
mc = ModelCheckpoint('model_GRU.h5', monitor='val_accuracy', save_best_only=True, verbose=1, mode='max')

X_train, X_test, Y_train, Y_test = train_test_split(X, YOneHot, test_size=0.2, random_state=42, shuffle=True)
history = model.fit(X_train, Y_train, callbacks=[es, mc], validation_data=(X_test, Y_test),
                    batch_size=32, epochs=100)

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

attX_train, attX_test, attY_train, attY_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=True)
earlystopper = EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=10,
                             verbose=1, restore_best_weights=True)
checkpointer = ModelCheckpoint('model-attRNN.h5', monitor='val_sparse_categorical_accuracy',
                               verbose=1, save_best_only=True)
attHistory = AttModel.fit(attX_train, attY_train, validation_data=(attX_test, attY_test), epochs=60,
                          use_multiprocessing=False, workers=4, verbose=2,
                          callbacks=[earlystopper, checkpointer, lrate])

# accuracy graph
plt.plot(attHistory.history['acc'])
plt.plot(attHistory.history['val_acc'])
plt.title('Model_accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('tt_Acommands_accuracy_graph.png')

# loss graph
plt.plot(attHistory.history['loss'])
plt.plot(attHistory.history['val_loss'])
plt.title('model loss')
plt.xlabel('epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Att_commands_loss_graph.png')
