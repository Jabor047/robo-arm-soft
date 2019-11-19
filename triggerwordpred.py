import utilities
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tensorflow.keras.models import load_model


def record():
    samplerate = 16000
    duration = 10
    filename = 'trigger.wav'
    
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    print(recording)
    sd.wait()
    print('Saving your command')
    sf.write(filename, recording, samplerate)


def preprocess_audio():
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav('trigger.wav')[:10000]
    segment = padding.overlay(segment)
    
    segment = segment.set_frame_rate(44100)
    segment.export('trigger.wav', format='wav')


def detect_triggerword(audio, model):
    x = utilities.graph_spectogram(audio)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0, :, 0])
    plt.ylabel('probability')
    plt.show()
    return predictions


def chime_on(audio, chime_file, predictions, threshold):
    audio_clip = AudioSegment.from_wav(audio)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    consecutives_timesteps = 0
    for i in range(Ty):
        consecutives_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutives_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position=((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutives_timesteps = 0
    audio_clip.export('Chime_ouput.wav', format='wav')


def main():
    model = load_model('tr_model.h5')
    chime_file = 'chime.wav'
    print('Hello there, please record your command immediately after you see start')
    time.sleep(4)
    print('Start')
    record()
    preprocess_audio()
    audio = 'trigger.wav'
    preds = detect_triggerword(audio, model)
    chime_threshold = 0.5
    chime_on(audio, chime_file, preds, chime_threshold)
    

if __name__ == '__main__':
    main()