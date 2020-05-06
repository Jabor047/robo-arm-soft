import pyaudio
# import sys
import time
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Queue
from tensorflow.keras.models import load_model


def graph_spectogram(data):
    '''Draws a spectogram of the wavfile load through its arguement
    Arguements:
    wav_file -- a file ending with wav extension
    Returns:
    returns a spectogram picture'''
    nfft = 200  # length of each window segment
    fs = 44100  # sampling freequencies
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft,
                                            fs, noverlap=noverlap)
    return pxx


def detect_triggerword(audio, model):
    x = graph_spectogram(audio)
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions


def triggerword_heard(predictions, chunk_duration,
                      feed_duration, threshold=0.45):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the prediction data belonging to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered
                positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions.reshape(-1)
    # change predictions to a 0, 1 array
    predictions = predictions > threshold
    chunk_pred_samples = int(len(predictions) *
                             chunk_duration / feed_duration)
    chunk_pred = predictions[-chunk_pred_samples:]
    level = chunk_pred[0]
    for pred in chunk_pred:
        if pred > level:
            return True
        else:
            level = pred
    return False
    # for pred in chunk_pred:
    #     if pred > threshold:
    #         return True
    # return False


def get_audio_stream(callback):
    chunk_duration = 0.5  # step_size for one 10 sec window
    form = pyaudio.paInt16
    channels = 1
    rate = 44100
    chunk_samples = int(rate * chunk_duration)

    p = pyaudio.PyAudio()
    stream = p.open(format=form, channels=channels, rate=rate, input=True,
                    frames_per_buffer=chunk_samples, stream_callback=callback)
    return stream


python_path = "C:\\Users\\gkkar\\AppData\\Local\\Programs\\Python\\Python36\\python.exe"
commandfile_path = str(os.getcwd())+'\\commandspred.py'
t1 = time.time()
model = load_model('triggerword/best_trigger_model_TM_30.h5')

chunk_duration = 0.5  # step_size for one 10 sec window
rate = 44100
chunk_samples = int(rate * chunk_duration)
feed_duration = 10  # length of our window the one to be fed into the model
feed_samples = int(rate * feed_duration)

q = Queue()
run = True
silence_threshold = 100
timeout = time.time() + 30  # define the duration of the demo
data = np.zeros(feed_samples, dtype='int16')


def callback(in_data, frame_count, time_info, status):
    # changing these variables here will change them in the whole code
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False
        return(in_data, pyaudio.paComplete)
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        print('-')
        # skips the parts where the input is so low
        return (in_data, pyaudio.paContinue)
    else:
        print('.')

    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # process data asychronously by sending a queue
        q.put(data)
    return (data, pyaudio.paContinue)


print("getting the audio stream for 30 secs")
stream = get_audio_stream(callback)
stream.start_stream()

try:
    while run:
        if not q.empty():
            data = q.get()
            pred = detect_triggerword(data, model)
            new_trigger = triggerword_heard(pred, chunk_duration,
                                            feed_duration)
            if new_trigger:
                # print(1)
                subprocess.run([python_path, commandfile_path])
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False
    exit()

stream.stop_stream()
stream.close()
t2 = time.time()
print(f'time taken = {t2-t1} seconds')
exit()
