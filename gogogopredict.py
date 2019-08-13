from scipy import signal
import math
from online_filter import Filter 
from basic_stuff import *
import pickle
from scipy import signal, fftpack

logger = logging.getLogger(__name__)

class Algorithm_prediction():
	def __init__(self):
		self.model_rf = pickle.load(open('demo1.sav', 'rb'))
		self.ODR = 50
		self.window_size = 200
		self.window = []
		self.window_smoothed = []
		self.var = 0
		self.feature = []
		self.result = ""

	def filter(self, data):
		sos = signal.butter(4, 1, 'highpass', fs=50, output='sos')
		filtered_data = signal.sosfilt(sos, data)

		return filtered_data

	def smooth(self, data):
		smooth_data = signal.savgol_filter(data, window_length=21, polyorder=3, mode='mirror')

		return smooth_data

	def eliminate_abnormal_value(self, new_data, orignal_data):
		window_size = 20
		threshold = 200
		new_orignal_data = []
		new_new_data = []

		for i in range(int(len(new_data)/window_size)):
			window = new_data[i*window_size:(i+1)*window_size]
			var = np.std(window)
			#print(var)
			if var < threshold:
				new_orignal_data.extend(orignal_data[i*window_size:(i+1)*window_size])
				new_new_data.extend(new_data)

		return np.array(new_new_data), np.array(new_orignal_data)

	def rms(self, data):
		rms = np.sqrt(np.mean(data**2))

		return rms

	def energy_for_each_freq_band(self, data):
		fft = fftpack.fft(data)
		freqs = fftpack.fftfreq(len(fft)) * self.ODR
		band = 50
		band_seg = int(len(freqs) / band)
		energy_vector = np.empty(band_seg)

		for i in range(band_seg):
			energy_vector[i] = sum(abs(fft[band*i:band*(i+1)])**2)

		return energy_vector

	def extract_features(self, data):
		window_size = 150
		#feature = np.empty([int(len(data)/window_size), 5])
		feature = np.empty([1, 5])

		window = data[0:window_size]
		mean = np.mean(window)
		std = np.std(window)
		feature[0, 0] = mean
		feature[0, 1] = std
		energy_vector = self.energy_for_each_freq_band(window)
		feature[0, 2] = min(window)
		feature[0, 3] = max(window)
		feature[0, 4] = self.rms(window)
		feature = np.append(feature, energy_vector)

		return np.array([feature])

	def feed(self, data):
		#print("window_size: ", self.window_size)
		if self.window_size == 0:
			data = self.window
			data = self.filter(data)
			new_data = self.smooth(data)
			new_new_data, new_orignal_data = self.eliminate_abnormal_value(new_data, data)
			if new_orignal_data.shape[0] >= 150:
				features = self.extract_features(new_orignal_data)
				self.result = self.model_rf.predict(features)
				print(self.result)
				exit()
			self.window = []
			self.window_size = 200
		else:
			self.window.append(data)
			self.window_size -= 1
			#print("collecting data")

def test_offline(fname):
    import numpy as np 
    data = np.loadtxt(fname, delimiter=';', comments='#', usecols=[2,3,4,5,6,7])
    prediction = Algorithm_prediction()
    data1=data[0:600,:]
    for ax,ay,az, mx, my, mz in data1:
        prediction.feed(ax)
        #print("\nThe vacuum cleaner is on the {}".format(prediction.result))
        

def test_online():
    import imports  # pylint: disable=unused-import
    from kx_lib.kx_board import ConnectionManager
    from kx_lib.kx_util import evkit_config
    from kmx62 import kmx62_data_logger
    from kmx62.kmx62_driver import KMX62Driver

    class CallBack(object):
        def __init__(self):
            self.prediction = Algorithm_prediction()

        def callback(self, data):
            ch, ax, ay, az, mx, my, mz, temp = data
            self.prediction.feed(ax)

            #print("\nThe vacuum cleaner is on the {}".format(self.prediction.result))

    ODR = 50
    callback_object = CallBack()

    connection_manager = ConnectionManager(board_config_json='rokix_board_rokix_sensor_node_i2c.json')
    sensor_kmx62 = KMX62Driver()
    try:
        connection_manager.add_sensor(sensor_kmx62)
        kmx62_data_logger.enable_data_logging(sensor_kmx62, odr=ODR)
        stream = kmx62_data_logger.KMX62DataStream([sensor_kmx62])
        stream.read_data_stream(console=False, callback=callback_object.callback)

    finally:
        sensor_kmx62.set_power_off()
        connection_manager.disconnect()

if __name__ == '__main__':
    #test_offline('testdata/kmx62_50Hz_accel_skid_y.csv')
    test_online()
