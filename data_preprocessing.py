import numpy as np
from scipy import signal


#### filter ####
# order is the order of filter
# fc is the cut frequency, should be array when type is 'bandpass' or 'bandstop'
# type is the type of filter, can be 'lowpass, 'highpass', 'bandpass', 'bandstop'
# fs is the sampling frequency
# output is the type of output features
def filter(data, order=4, fc=1, type='highpass', fs=50, output='sos'):
	
	sos = signal.butter(order, fc, type, fs=fs, output=output)

	filtered_data = []

	for col in range(3):
		data[:, col] = signal.sosfilt(sos, data[:, col])

	return data

### smoothing data ####
def smooth(data, window_length=21, polyorder=3, mode='mirror'):
	
	smooth_data = []

	for col in range(3):
		temp = signal.savgol_filter(data[:, col], window_length=window_length, polyorder=polyorder, mode=mode)
		smooth_data.append(temp)

	smooth_data = np.array(smooth_data).T
	#print("smooth data: ", smooth_data.shape)

	return smooth_data

#### eliminate abnormal value ####
# using to eliminate data that changes suddenly
# new_data is the data after smoothing
def eliminate_abnormal_value(new_data, orignal_data, window_size, threshold, base_on_col):
	new_orignal_data = []
	new_new_data = []
	
	for i in range(int(len(new_data[:, base_on_col-1])/window_size)):
		window = new_data[i*window_size:(i+1)*window_size, base_on_col-1]
		var = np.std(window)
		if var < threshold:
			new_orignal_data.extend(orignal_data[i*window_size:(i+1)*window_size, :])
			new_new_data.extend(new_data[i*window_size:(i+1)*window_size, :])
	

	new_orignal_data = np.array(new_orignal_data)
	new_new_data = np.array(new_new_data)
	#print("new_orignal_data: ", new_orignal_data.shape)
	#print("new_new_data: ", new_new_data.shape)

	return new_new_data, new_orignal_data
