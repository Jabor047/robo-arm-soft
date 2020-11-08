from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D, Conv2D, Permute
from tensorflow.keras.layers import Activation, GRU, BatchNormalization, LSTM, MaxPooling2D, Lambda
from tensorflow.keras.layers import Bidirectional, Input, Reshape, Dot, Softmax
from tensorflow.keras.optimizers import Adam
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D


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
    model.add(Dense(9, activation='softmax'))
    
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model

def attRNN():
    sr = 8000
    inputs = Input((8000, 1), name='input')

    x = Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, 8000),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = Permute((2, 1, 3))(x)

    x = Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = Bidirectional(LSTM(64, return_sequences=True)
                      )(x)  # [b_s, seq_len, vec_dim]

    x = Bidirectional(LSTM(64, return_sequences=True)
                      )(x)  # [b_s, seq_len, vec_dim]

    xFirst = Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = Dense(128)(xFirst)

    # dot product attention
    attScores = Dot(axes=[1, 2])([query, x])
    attScores = Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    x = Dense(64, activation='relu')(attVector)
    x = Dense(32)(x)

    output = Dense(9, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'],
                  metrics=['sparse_categorical_accuracy'])
    model.summary()

    return model
