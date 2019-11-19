
import os
import sys
# import shutil
import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks

chucklen = 1 * 1000
k = 0
savedir = '/mnt/e/projects/deep_learning/dataset/5th/background_command'

if sys.platform == 'linux':
    connector = '/'
elif sys.platform == 'win32':
    connector = '\\'
# j = 0
# for foldername, subfolder, filenames in os.walk(r'E:\projects\deep_learning\dataset\Noise'):
#     for filename in filenames:
#         if filename.endswith('wav'):
#             pass
#         else:
#             shutil.move(foldername + connector + filename, foldername + connector + '{}'.format(j) + '.mp3')
#             j += 1
            
for foldername, subfolder, filenames in os.walk('/mnt/e/projects/deep_learning/dataset/5th/background_trigger'):
    for filename in filenames:
        if filename.endswith('mp3'):
            myaudio = AudioSegment.from_mp3(foldername + connector + filename)
        elif filename.endswith('wav'):
            myaudio = AudioSegment.from_wav(foldername + connector + filename)
            
        mychunks = make_chunks(myaudio, chucklen)
        for i, chunk in enumerate(mychunks):
            chunkname = "chunk{}{}.wav".format(k, i)
            print("saving {}".format(chunkname))
            chunk.export(savedir + '{}{}'.format(connector, chunkname), format='wav')
        k += 1

for foldername, subfolder, filenames in os.walk(savedir):
    for filename in filenames:
        if filename.endswith('wav'):
            y, sr = librosa.load(foldername + connector + filename)
            if librosa.get_duration(y) < 1.0:
                os.remove(foldername + connector + filename)