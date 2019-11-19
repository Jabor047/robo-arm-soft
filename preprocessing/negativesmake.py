import random
import os
import shutil
import sys

wavefiles = []

if sys.platform == 'linux':
    connector = '/'
elif sys.platform == 'win32':
    connector = '\\'

for foldername, subfolders, filenames in os.walk(r'E:\projects\deep_learning\dataset\data_speech_commands_v0.02'):
    for filename in filenames:
        if filename.endswith('wav'):
            wavefiles.append(foldername + connector + filename)

negatives = random.sample(wavefiles, 3845)

i = 0
for negative in negatives:
    shutil.copy(negative, 'E:\\projects\\deep_learning\\dataset\\5th\\negatives\\' + str(i) + '.wav')
    i += 1
print("{} files have been copied".format(len(negatives)))