import os
import librosa
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import utils
# import utilities

# utilities.isdatabalanced()

target_list = ['up', 'down', 'left', 'right', 'forward', 'backward', 'off']
currentpath = os.getcwd()
datapath = str((Path(currentpath).parent).parent)
dirs = [f for f in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, f))]
dirs.sort()
notsdir = ['background_trigger', 'code', 'train', 'negatives', 'test']
for notneeded in notsdir:
    dirs.remove(notneeded)


all_wav = []
unknown_wav = []
label_all = []

unknown_list = [d for d in dirs if d not in target_list and d != 'background_command']
background = [f for f in os.listdir(os.path.join(datapath, 'background_command')) if f.endswith('wav')]
background_noise = []
print('Loading background_command data')
for wav in background:
    samples, sample_rate = librosa.load(os.path.join(os.path.join(datapath, 'background_command'), wav), sr=16000)
    samples = librosa.resample(samples, sample_rate, 8000)
    background_noise.append(samples)

print('Loading the unknown commands and the target commands')
for direct in dirs:
    if direct == 'background_command':
        continue
    waves = [f for f in os.listdir(os.path.join(datapath, direct)) if f.endswith('wav')]
    for wav in waves:
        samples, sample_rate = librosa.load(os.path.join(os.path.join(datapath, direct), wav), sr=16000)
        samples = librosa.resample(samples, sample_rate, 8000)
        if len(samples) != 8000:
            continue
        if direct in unknown_list:
            unknown_wav.append(samples)
        else:
            all_wav.append([samples, direct])

wav_all = np.reshape(np.delete(all_wav, 1, 1), (len(all_wav)))
label_all = [i for i in np.delete(all_wav, 0, 1).tolist()]
wav_vals = np.array([x for x in wav_all])
label_vals = [x for x in label_all]
label_vals = np.array(label_vals)
label_vals = label_vals.reshape(-1, 1)
# # augment background_noise so as to have a more robust and balanced dataset


# def get_noise(num_noise=0):
#     selected_noise = background[num_noise]
#     start_idx = np.random.randint(0, len(selected_noise) - 1 - 8000)
#     return selected_noise[start_idx: start_idx + 8000]

# max_ratio = 0.1
# noise_wav = []
# augment = 1
# back_idx = np.random.choice()
# for i in range(augment):
#     noise = get_noise(i)
#     for i, s in enumerate(wav_all):
#         s = s + (max_ratio * noise)
#         noise_wav.append(s)
# label_vals = [x for x in label_all]

print('Selecting random number of unknowns')
np.random.shuffle(unknown_wav)
unknown = np.array(unknown_wav)
unknown = unknown[:4000]
unknown_label = np.array(['unknown' for _ in range(len(unknown))])
unknown_label = unknown_label.reshape(-1, 1)

print('Selecting a random number of backgrounds')
np.random.shuffle(background_noise)
silence = np.array(background_noise)
silence = silence[:4000]
silence_label = np.array(['silence' for _ in range(len(silence))])
silence_label = silence_label.reshape(-1, 1)

wav_vals = np.reshape(wav_vals, (-1, 8000))
unknown = np.reshape(unknown, (-1, 8000))
silence = np.reshape(silence, (-1, 8000))


wav_all = np.concatenate((wav_vals, unknown), axis=0)
wav_all = np.concatenate((wav_all, silence), axis=0)
wav_all = wav_all.reshape(-1, 8000, 1)

label_all = np.concatenate((label_vals, unknown_label), axis=0)
label_all = np.concatenate((label_all, silence_label), axis=0)

print(wav_all.shape)
print(label_all.shape)

labels = target_list
labels.append('silence')
labels.append('unknown')
label_encoder = LabelEncoder()
fit_y = label_encoder.fit(labels)
y = fit_y.transform(label_all.ravel())
classes = list(label_encoder.classes_)
y = utils.to_categorical(y)
print(classes)

print('saving')
np.save('preprocessing/X_commands.npy', wav_all)
np.save('preprocessing/Y_commands.npy', y)


# preprocessing for the test data
print('Now Doing the Test Data')

target_list_test = ['up', 'down', 'left', 'right', 'off']
currentpath_test = os.getcwd()
datapath_test = str((Path(currentpath_test).parent).parent)
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
wav_vals_test = wav_vals_test.reshape(-1, 8000, 1)

y_test = fit_y.transform(label_vals_test.ravel())
y_test = utils.to_categorical(y_test)
print(classes)

print('saving')
np.save('preprocessing/X_test_commands.npy', wav_vals_test)
np.save('preprocessing/Y_test_commands.npy', y_test)

classtxt = open('classlist.txt', 'w')
classtxt.write('The list is: {}'.format(classes))
classtxt.close()
