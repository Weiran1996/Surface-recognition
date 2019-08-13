import numpy as np  
from scipy import signal, fftpack, stats

def mean(data):

	return np.mean(data)


def std(data):

	return np.std(data)


def getmin(data):

	return min(data)


def getmax(data):

	return max(data)


def rms(data):

	rms = np.sqrt(np.mean(data**2))

	return rms


def energy_for_each_freq_band(data, ODR, band_size):
	fft = fftpack.fft(data)
	freqs = fftpack.fftfreq(len(fft)) * ODR
	band_seg = int(len(freqs) / band_size)
	energy_vector = np.empty(band_seg)

	for i in range(band_seg):
		energy_vector[i] = sum(abs(fft[band_size*i:band_size*(i+1)])**2)

	return energy_vector