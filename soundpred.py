import time
import librosa
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.ERROR)

model = load_model('best_commands_model_GRU_2.h5')
classes = ['down', 'left', 'off', 'right', 'silence', 'unknown', 'up']


def predict(audio):
    prob = model.predict(audio.reshape(-1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


def record():
    samplerate = 16000
    duration = 1
    filename = 'arm_command.wav'
    
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    print(recording)
    sd.wait()
    print('Saving your command')
    sf.write(filename, recording, samplerate)


def load_audio():
    samples, samplerate = librosa.load('arm_command.wav', sr=16000)
    samples = librosa.resample(samples, samplerate, 8000)
    return samples
    

def main():
    print('Hello there, please record your command immediately after you see start')
    time.sleep(4)
    print('Start')
    record()
    audio = load_audio()
    
    print('Your command is : {}'.format(predict(audio)))


if __name__ == "__main__":
    main()