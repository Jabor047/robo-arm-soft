import sys
# import io
import os
# import glob
# import librosa
# import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
# import plotly.tools as tls
from scipy.io import wavfile
# from sklearn.decomposition import PCA
from pathlib import Path
from pydub import AudioSegment

currentpath = os.getcwd()
datapath = str((Path(currentpath).parent).parent)

if sys.platform == 'linux':
    connector = '/'
elif sys.platform == 'win32':
    connector = '\\'


def get_wav_info(wav_file):
    '''Reads the wav files and returns the sample rate and data'''
    rate, data = wavfile.read(wav_file)
    return rate, data


def graph_spectogram(wav_file):
    '''Draws a spectogram of the wavfile load through its arguement
    Arguements:
    wav_file -- a file ending with wav extension
    Returns:
    returns a spectogram picture'''
    rate, data = get_wav_info(wav_file)
    nfft = 200  # length of each window segment
    fs = 44100  # sampling freequencies
    noverlap = 120
    nchannels = data.ndim
    if nchannels == 1:
        pxx, freqs, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freqs, bins, im = plt.specgram(data[:, 0], nfft, fs,
                                            noverlap=noverlap)
    return pxx


def match_target_amplitude(sound, target_dBFS):
    '''Normalize the amplitude(volume) through out the dataset'''
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def isdatabalanced():
    '''Takes the number of files in each folder and draws a bar graph to see the difference in number
    files per command'''
    dirs = [f for f in os.listdir(datapath) if os.path.isdir(os.path.join(datapath, f))]
    dirs.sort()
    notsdir = ['_background_noise_', 'code', 'train']
    for notneeded in notsdir:
        dirs.remove(notneeded)
    numberofrecordings = []
    for direct in dirs:
        waves = [f for f in os.listdir(os.path.join(datapath, direct)) if f.endswith('.wav')]
        numberofrecordings.append(len(waves))

    trace = go.Bar(x=dirs, y=numberofrecordings,
                   marker=dict(color=numberofrecordings,
                               colorscale='icefire', showscale=True))
    layout = go.Layout(title='Number of recordings in given label',
                       xaxis=dict(title='Words'),
                       yaxis=dict(title='Number of recordings'))
    py.plot(go.Figure(data=[trace], layout=layout))


def load_trigger_raw_audio():
    ons = []
    backgrounds = []
    negatives = []
    print('loading ons')
    for filename in os.listdir(os.path.join(datapath, 'on')):
        if filename.endswith('wav'):
            on = AudioSegment.from_wav(os.path.join(datapath, 'on')
                                       + connector + filename)
            ons.append(on)
    print('loading negatives')
    for filename in os.listdir(os.path.join(datapath, 'negatives')):
        if filename.endswith('wav'):
            negative = AudioSegment.from_wav(os.path.join(datapath, 'negatives')
                                             + connector + filename)
            negatives.append(negative)
    print('loading background_trigger')
    for filename in os.listdir(os.path.join(datapath, 'background_trigger')):
        if filename.endswith('wav'):
            background = AudioSegment.from_wav(os.path.join(datapath, 'background_trigger')
                                               + connector + filename)
            backgrounds.append(background)
    return ons, negatives, backgrounds


def get_random_time_segment(segment_ms):
    '''gets a random time segement of duration segment_ms in a 10,000 ms audio clip
    Arguments:
    segment_ms -- duration of audio clip in milliseconds
    Returns:
    returns a tuple in ms '''
    segment_start = np.random.randint(low=0, high=10000 - segment_ms)
    segment_end = segment_start + segment_ms - 1
    return(segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    '''checks is the time of the segment overlaps with times of existing segments
    Arguements:
    segment_time is a tuple of (segment_start, segment_end)for the new segment
    previous_segments is a list of tuples of all previous times

    Returns:
    returns True is segment overlaps with any of the previous times'''
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if segment_end >= previous_start and segment_start <= previous_end:
            overlap = True
    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    '''Insert a new audio segment over the background noise at a random time
    step, ensuring that the audio segment does not overlap with existing
    segments

    Arguements:
    background -- a 10 second background recording
    audio_clip -- the audio clip being inserted
    previous_segments -- times where audio segments have been placed already

    Returns:
    new_background -- the updated background clip'''

    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_ms)

    previous_segments.append(segment_time)
    new_background = background.overlay(audio_clip, position=segment_time[0])
    return new_background, segment_time


def insert_ones(y, segment_end_ms, Ty):
    '''Update the label vector. The labels of the 40 output steps strictly
    after the end of the segment should be set to 1. By strictly the label of
    segment_end_y should be 0 while, the 40 following labels should be one
    Arguments:
    y -- numpy array of shape(1,Ty), the labels of the training examples
    segment_end_ms -- the end of time of the segment in ms
    returns:
    y -- updated labels'''
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    for i in range(segment_end_y + 1, segment_end_y + 41):
        if i < Ty:
            y[0, i] = 1
    return y


def create_trigger_dataset(background, ons, negatives, num, id, Ty):
    '''Creates wavfiles with the background , on and negatives
    Arguements:
    background -- a 60 seconds background audio recording
    ons -- a list of audio segments with the word "on"
    negatives -- a list of audio segments with randwords that are not "on"
    Returns:
    y  -- the label at each time step of the training example
    x -- spectogram of the created wav file
    '''
    background = background - 20

    y = np.zeros((1, Ty))
    previous_segments = []

    number_of_ons = np.random.randint(0, 5)
    random_indicies = np.random.randint(len(ons), size=number_of_ons)
    random_ons = [ons[i] for i in random_indicies]

    for random_on in random_ons:
        background, segment_time = insert_audio_clip(background, random_on,
                                                     previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end, Ty)
    print(y)
    number_of_negatives = np.random.randint(0, 3)
    random_indicies = np.random.randint(len(negatives),
                                        size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indicies]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative,
                                          previous_segments)

    background = match_target_amplitude(background, -20.0)
    trainpath = os.path.join(datapath, 'train') + connector
    background.export(trainpath + 'train{}{}.wav'.format(num, id), format='wav')

    x = graph_spectogram(trainpath + 'train{}{}.wav'.format(num, id))

    return x, y
