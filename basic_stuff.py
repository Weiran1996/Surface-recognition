import os
import json
import logging
import numpy as np
from dotmap import DotMap # pip install dotmap
logging.basicConfig(
    format='%(asctime)s %(levelname)s :\t%(filename)s (%(lineno)d) :\t%(funcName)s :\t%(message)s')
from settings import *

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class AlgorithmBase(object):
    def __init__(self, name = '*', configuration_directory='algo_settings'):
        self._configuration_directory = configuration_directory
        self.state = 0
        self.configuration_dictionary = {
            'foo':1,
            'bar':2,
            }
        self.config = DotMap(self.configuration_dictionary)
        self.name = name

    def configure(self,configuration_dictionary_json=None):
        try:
            if not isinstance(configuration_dictionary_json, dict):
                configuration_dictionary_json = json.loads(
                    os.path.join(self._configuration_directory, configuration_dictionary_json))
                
            self.configuration_dictionary.update(configuration_dictionary_json)
            self.config = DotMap(self.configuration_dictionary)

            logger.info('Configure algorithm {}'.format(self.configuration_dictionary))

        except Exception as e:
            logger.error(e)

    def feed(self):
        pass

class IIR_Filter1(object):
    def __init__(self, aplha):
        self.alpha = alpha
        self.prev_value = 0

    def feed(self,data):
        value = self.prev_value + self.alpha * (data - self.prev_value)
        self.prev_value = value
        return value

class IIR_Filter(object):
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.prev_value = 0

    def feed(self,data):
        value = self.prev_value + self.alpha * (data - self.prev_value)
        self.prev_value = value
        return value

class FIR_Filter(object):
    def __init__(self, weights=[0.2,0.2,0.2,0.2,0.2]):
        self.weights = weights
        self.buffer = [0] * len(weights)
        self.length = len(weights)

    def feed(self, data):

        self.buffer.pop(0)
        self.buffer.append(data)
        result = 0
        for t in range(self.length):
            result += self.weights[t]*self.buffer[t]

        return result # / self.length # do not divide when sum of weights is 1

class Filter_3D(object):
    def __init__(self, filter_class, weights):
        self.xfilter = filter_class(weights)
        self.yfilter = filter_class(weights)
        self.zfilter = filter_class(weights)
        self.x=0
        self.y=0
        self.z=0

    def feed(self,x,y,z):
        self.x, self.y, self.z= self.xfilter.feed(x), self.yfilter.feed(y), self.zfilter.feed(z)
        return self.x, self.y, self.z
