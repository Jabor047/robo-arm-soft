import time
import pyaudio
import numpy as np
data_c = None


def callback(in_data, frame_count, time_info, status):
    global data_c
    data_c = np.frombuffer(in_data, dtype='int16')
    print(np.abs(data_c).mean())
    return(in_data, pyaudio.paContinue)


chunk_duration = 0.5
form = pyaudio.paInt16
channels = 2
rate = 44100
chunk_samples = int(rate * chunk_duration)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True,
                frames_per_buffer=chunk_samples, input_device_index=0,
                stream_callback=callback)

stream.start_stream()
time.sleep(5.1)
stream.stop_stream()
stream.close()
