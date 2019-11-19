
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Activation, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam


def commandsmodel():
    '''Contains the deep learning model for the commands classification
    Returns:
    model -- tensorflow.keras model instance'''
    
    model = Sequential()
    
    # first layer (Conv1D)
    model.add(Conv1D(8, kernel_size=13, strides=1, padding='valid', input_shape=(8000, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    
    # Second layer(Second Conv1D layer)
    model.add(Conv1D(16, kernel_size=11, padding='valid', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    
    # third layer(third Conv1D layer)
    model.add(Conv1D(32, kernel_size=9, padding='valid', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    
    # fourth layer(fourth Conv1D layer)
    model.add(Conv1D(64, kernel_size=9, padding='valid', strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(Dropout(0.3))
    
    # fifth layer a Gru layer
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    
    # sixth layer (GRU)
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    
    # flatten layer
    model.add(Flatten())
    
    # Dense layer 1
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Dense layer 2
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    
    # output layer
    model.add(Dense(7, activation='softmax'))
    
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model