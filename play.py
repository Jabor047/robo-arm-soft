import wave
import pyaudio


def playwav(wavfile):
    chunk = 1024
    wf = wave.open(wavfile, 'rb')

    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(chunk)

    while data:
        stream.write(data)
        data = wf.readframes(chunk)

    wf.close()
    stream.close()
    p.terminate()
