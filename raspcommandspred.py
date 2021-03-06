import os
import time
import librosa
import logging
import serial
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

model = load_model('commands/model_GRU.h5')
classes = ['backward', 'down', 'forward', 'left', 'off', 'right', 'silence', 'unknown', 'up']
chimewav = os.getcwd() + "/chime.wav"


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
    # chime = AudioSegment.from_wav(chimewav)
    # play(chime)
    print('Start')
    sleep(0.5)
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
          ' immediately after you see start')

    board = Arduino('/dev/ttyUSB0')
    RightLeftServoPin = board.get_pin('d:5:s')
    ForwardBackwardServoLPin = board.get_pin('d:6:s')
    UpDownServoPin = board.get_pin('d:7:s')
    # gripServoPin = board.get_pin('d:8:p')
    sleep(3)

    iterSer = util.Iterator(board)
    iterSer.start()

    angle = 90
    change = 45

    while True:
        command = imp()
        if command == "left":
            print('Your command is : {} \n Turning Left'.format("Left"))
            RightLeftServoPin.write(angle - change)
            board.pass_time(2)
        elif command == "right":
            print('Your command is : {} \n Turning Right'.format("Right"))
            RightLeftServoPin.write(angle + change)
            board.pass_time(2)
        elif command == "up":
            print('Your command is : {} \n Going Up'.format("Up"))
            UpDownServoPin.write(angle + change)
            board.pass_time(2)
        elif command == "down":
            print('Your command is : {} \n Going Down'.format("Down"))
            UpDownServoPin.write(angle - change)
            board.pass_time(2)
        elif command == "forward":
            print('Your command is : {} \n Going forward'.format("forward"))
            ForwardBackwardServoLPin.write(angle + change)
            board.pass_time(2)
        elif command == "backward":
            print('Your command is : {} \n Going backward'.format("backward"))
            ForwardBackwardServoLPin.write(angle - change)
            board.pass_time(2)
        elif command == "off":
            print('Your command is : {} \n Turning off '.format("off"))
            break

    board.exit()

if __name__ == "__main__":
    main()
