
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D
from tensorflow.keras.layers import Activation, GRU, BatchNormalization, TimeDistributed
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import ExponentialDecay


def trigger_model(input_shape):
    '''Çontains the deep learning model for the trigger word detection
    
    Arguements:
    input_shape -- shape of the models input data
    
    Returns:
    model -- tensorflow.keras model instance'''
    
    model = Sequential()
    
    # first layer(COnv1D)
    model.add(Conv1D(256, kernel_size=15, strides=4, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    
    # second layer (GRU)
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    
    # third layer (Second GRU)
    model.add(GRU(128, return_sequences=True))
    model.add(Dropout(0.8))
    model.add(BatchNormalization())
    model.add(Dropout(0.8))
    
    # fourth layer (Time-distributed dense layer)
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    
    # initial_lr = 0.1
    # lr_schedule = ExponentialDecay(initial_lr, decay_steps=10000, decay_rate=0.96, staircase=True)
    
    # when lr = 0.01 an accuracy of 95% was achieved
    opt = Adam(lr=0.1, beta_1=0.9, beta_2=0.999, decay=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model