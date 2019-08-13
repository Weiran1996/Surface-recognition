import numpy as np
from scipy import signal

class Filter(object):
    def __init__(self, b, a=1, channels=['ax','ay','az']):
        # If FIR a = 1, with IIR set A
        self.b = b 
        self.a = a
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.channels = channels
        self.axis_f_states = {key:self.zi for key in self.channels}

    def __call__(self, values):

        # values: shape (samples x channels) e.g. [1,2,3]
        result = np.zeros(len(self.channels))
        for idx, value in enumerate(values):
            channel = self.channels[idx]
            result[idx], self.axis_f_states[channel] = signal.lfilter(
                self.b, self.a, [value], zi=self.axis_f_states[channel])
        return result

    def reset_state(self, channels=None):
        if channels is None:
            self.channels = channels
        self.zi = signal.lfilter_zi(self.b, self.a)
        self.axis_f_states = {key:self.zi for key in self.channels}

