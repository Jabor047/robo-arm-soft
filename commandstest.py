import os
import librosa
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils

target_list_test = ['up', 'down', 'left', 'right', 'off', 'silence', 'unknown']
currentpath_test = os.getcwd()
datapath_test = str(Path(currentpath_test).parent)
datapath_test = os.path.join(datapath_test, 'test')
dirs_test = [f for f in os.listdir(datapath_test) if os.path.isdir(os.path.join(datapath_test, f))]
dirs_test.sort()

all_wav_test = []
print('Loading the test commands')
for direct_test in dirs_test:
    if direct_test not in target_list_test:
        continue
    else:
        waves = [f for f in os.listdir(os.path.join(datapath_test, direct_test)) if f.endswith('wav')]
        for wav in waves:
            samples, sample_rate = librosa.load(os.path.join(os.path.join(datapath_test, direct_test), wav), sr=16000)
            samples = librosa.resample(samples, sample_rate, 8000)
            if len(samples) != 8000:
                continue
            else:
                all_wav_test.append([samples, direct_test])

wav_all_test = np.reshape(np.delete(all_wav_test, 1, 1), (len(all_wav_test)))
label_all_test = [i for i in np.delete(all_wav_test, 0, 1).tolist()]
wav_vals_test = np.array([x for x in wav_all_test])
label_vals_test = [x for x in label_all_test]
label_vals_test = np.array(label_vals_test)
label_vals_test = label_vals_test.reshape(-1, 1)

wav_vals_test = np.reshape(wav_vals_test, (-1, 8000))

labels = target_list_test
label_encoder = LabelEncoder()
y_test = label_encoder.fit_transform(label_vals_test.ravel())
classes = list(label_encoder.classes_)
y_test = utils.to_categorical(y_test)
print(classes)

print('saving')
np.save('X_test_commands.npy', wav_vals_test)
np.save('Y_test_commands.npy', y_test)
