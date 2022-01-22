import os
import time
import librosa
import logging
import serial
import models
import numpy as np
import sounddevice as sd
import soundfile as sf
from time import sleep
from pydub import AudioSegment
from pydub.playback import play
from tensorflow.keras.models import load_model
from pyfirmata import Arduino, util
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D

logging.basicConfig(level=logging.ERROR)

currentpath = os.getcwd()
model = models.attRNN()
model.load_weights('commands/model-attRNN-weights.h5')
# model = load_model('commands/model-attRNN.h5', custom_objects={'Melspectrogram': Melspectrogram,
# 'Normalization2D': Normalization2D})
classes = ['backward', 'down', 'forward', 'left', 'off', 'right', 'silence', 'unknown', 'up']
chimewav = currentpath + "/chime.wav"


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
    # play(chime)
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

# pin.read to get servo angle before adding direction
def main():
    print('Hello there, please record your command'
          'immediately after you see start')
    # windows = 'COM8'
    # PiOS = '/dev/ttyUSB0'
    board = Arduino('/dev/ttyUSB0')
    RightLeftServoPin = board.get_pin('d:5:s')
    ForwardBackwardServoLPin = board.get_pin('d:6:s')
    UpDownServoPin = board.get_pin('d:7:s')
    # gripServoPin = board.get_pin('d:8:p')
    sleep(5)

    iterSer = util.Iterator(board)
    iterSer.start()

    angle = 90
    change = 45

    while True:
        command = imp()
        if command == "left":
            print('Your command is : {}'.format("Turning Left"))
            RightLeftServoPin.write(angle - change)
        elif command == "right":
            print('Your command is : {}'.format("Turning Right"))
            RightLeftServoPin.write(angle + change)
        elif command == "up":
            print('Your command is : {}'.format("Going Up"))
            UpDownServoPin.write(angle + change)
        elif command == "down":
            print('Your command is : {}'.format("Going Down"))
            UpDownServoPin.write(angle - change)
        elif command == "forward":
            print('Your command is : {}'.format("Going forward"))
            ForwardBackwardServoLPin.write(angle + change)
        elif command == "backward":
            print('Your command is : {}'.format("Going backward"))
            ForwardBackwardServoLPin.write(angle - change)
        elif command == "off":
            break

    board.exit()

if __name__ == "__main__":
    main()
