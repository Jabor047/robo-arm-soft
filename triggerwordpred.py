from triggerword import utilities
import time
import pyaudio
import wave
import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tensorflow.keras.models import load_model


def record():
    samplerate = 16000
    duration = 10.5
    filename = 'trigger.wav'
    
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, blocking=True)
    print(recording)
    sd.wait()
    print('Saving your command')
    sf.write(filename, recording, samplerate)
    

def take_input():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 3
    WAVE_OUTPUT_NAME = 'trigger.wav'
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)
    
    print('\n\n\n************ Started Lisening ___________________')
    frames = []
    for i in range(int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print('************* Done Listening')
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_NAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close
    return WAVE_OUTPUT_NAME


def preprocess_audio(audio):
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(audio)[:10000]
    segment = padding.overlay(segment)
    
    segment = segment.set_frame_rate(44100)
    segment.export(audio, format='wav')


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
    model = load_model('triggerword/best_trigger_model_TM_100.h5')
    chime_file = 'chime.wav'
    print('Hello there, please record your command immediately after you see start')
    time.sleep(4) 
    # print('Start')
    # record()
    audio = take_input()
    preprocess_audio(audio)
    preds = detect_triggerword(audio, model)
    chime_threshold = 0.5
    chime_on(audio, chime_file, preds, chime_threshold)
    

if __name__ == '__main__':
    main()