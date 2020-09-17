import os
import time
import librosa
import logging
import serial
import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
from tensorflow.keras.models import load_model

logging.basicConfig(level=logging.ERROR)

model = load_model('commands/best_commands_model_GRU_2.h5')
classes = ['down', 'left', 'off', 'right', 'silence', 'unknown', 'up']
chimewav = os.getcwd() + "\\chime.wav"


def predict(audio):
    prob = model.predict(audio.reshape(-1, 8000, 1))
    index = np.argmax(prob[0])
    return classes[index]


def record():
    samplerate = 16000
    duration = 1
    filename = 'arm_command.wav'

    recording = sd.rec(int(samplerate * duration), samplerate=samplerate,
                       channels=1, blocking=True)
    print(recording)
    sd.wait()
    print('Saving your command')
    sf.write(filename, recording, samplerate)


def load_audio():
    samples, samplerate = librosa.load('arm_command.wav', sr=16000)
    samples = librosa.resample(samples, samplerate, 8000)
    return samples


def imp():
    time.sleep(4)
    chime = AudioSegment.from_wav(chimewav)
    play(chime)
    print('Start')
    record()
    audio = load_audio()

    prediction = predict(audio)
    command = prediction
    if prediction == "silence" or prediction == "unknown":
        command = imp()

    return command


# def prog(arduino):
#     command = imp()
#     if command == "left":
#         print('Your command is : {}'.format("Turning Left"))
#         prog()
#     elif command == "right":
#         print('Your command is : {}'.format("Turning Right"))
#         prog()
#     elif command == "up":
#         print('Your command is : {}'.format("Going Up"))
#         prog()
#     elif command == "down":
#         print('Your command is : {}'.format("Going Down"))
#         prog()
#     elif command == "off":
#         exit()


def main():
    print('Hello there, please record your command'
          'immediately after you see start')

    arduino = serial.Serial("COM1", 9600)
    command = imp()

    if command == "off":
        exit()

    command = str.encode(command)
    arduino.write(command)

    # prog(arduino)


if __name__ == "__main__":
    main()
