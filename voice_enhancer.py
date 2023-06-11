import sys
import wave

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import write


# function to read the audio samples into a python application
def wavread(wavpath):
    wavfile = wave.open(wavpath, 'rb')
    params = wavfile.getparams()
    framerate, nframes = params[2], params[3]  # get framerate and nframes
    # Reads and returns at most n frames of audio, as a bytes object.
    wavdata = wavfile.readframes(nframes)
    wavfile.close()
    datause = np.frombuffer(wavdata, dtype=np.short)  # convert string to int
    time = np.arange(nframes) * (1.0 / framerate)  # return one list of time
    print(framerate, nframes)
    return datause, time, framerate, nframes


# function to plot normalised amplitudes vs time using a linear axis in the time domain and save the image in the current path
def namp_vs_time(wavpath):
    data = wavread(wavpath)
    wavdata, time = data[0], data[1]
    nwavdata = wavdata * 1.0 / (max(abs(wavdata)))  # normalized amplitude
    plt.figure(figsize=(12, 4.5))
    plt.plot(time, nwavdata, color='b')
    plt.xlabel("Time(s)")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.savefig("NormalizedAmplitude_vs_Time.svg", dpi=600,
                format="svg", bbox_inches='tight', pad_inches=0.1)
    return None


# function to plot amplitude (dB) vs frequency using logarithmic axis in the frequency domain and save the image in the current path
def dB_vs_freq(wavpath):
    data = wavread(wavpath)
    wavdata, framerate, nframes = data[0], data[2], data[3]
    nwavdata = wavdata * 1.0 / max(abs(wavdata))  # normalized amplitude
    ft_signal = fft(nwavdata)  # fft
    n = nframes // 2  # only need half of the fft list
    T = nframes / framerate
    # the real frequency of the signal at the sampling point
    freq = np.arange(nframes) / T
    plt.figure(figsize=(12, 4.5))
    plt.plot(freq[:n], 20 * np.log((abs(ft_signal[:n]))))
    plt.xscale('log')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("Amplitude(dB)_vs_Log(Frequency).svg", dpi=600,
                format="svg", bbox_inches='tight', pad_inches=0.1)
    return None


# function to increase the voice quality and save in the current path
def voice_enhancer(wavpath):
    sys.setrecursionlimit(1000000)
    # Set the maximum depth of the Python interpreter stack to limit.
    # This limit prevents infinite recursion from causing an overflow of the C stack and crashing Python.
    data = wavread(wavpath)
    wavdata, time, framerate, nframes = data[:4]
    nwavdata = wavdata * 1.0 / max(abs(wavdata))  # normalized amplitude
    ft_signal = fft(nwavdata)  # fft
    # frequency of signal, the same as dB_vs_freq
    freq = np.arange(nframes) / (nframes / framerate)
    # Removal of low and high frequency noise
    for i in range(nframes):
        if freq[i] < 130 or freq[i] > 4800:
            ft_signal[i] = 0
    # Removal of noise below 40dB, but after this there will be white noise as a background sound
    # dB = 20 * np.log10(abs(ft_signal))
    # for i in range(nframes):
    #     if dB[i] < 20:
    #         ft_signal[i] = 0
    ft_signal = ft_signal * 6  # increased of around 15dB
    filtered = (ifft(ft_signal)).real * max(abs(wavdata))  # ifft
    # creat 'improved.wav' and save the voice
    write('improved.wav', framerate, filtered.astype(np.int16))
    return None

# function to input and output
def main():
    wavpath = 'original.wav'
    namp_vs_time(wavpath)
    dB_vs_freq(wavpath)
    plt.show()
    voice_enhancer(wavpath)


if __name__ == '__main__':
    main()
