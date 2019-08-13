####
import os
from scipy import signal
import math
from online_filter import Filter 
from basic_stuff import *
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
global_settings = load_settings()
node_settings = load_settings('node_settings')

ST_INIT, ST_RUNNING = range(2)

class Algorithm_g_force(AlgorithmBase): # side to side
    def __init__(self, channel=['ax']):
        self.make_settings()

        self.settings = load_settings('algorithm_g')

        b, a = signal.butter(
            N=self.settings.filter.n,
            Wn=self.settings.filter.Wn_Hz,
            fs=global_settings.sensor.odr,
            btype='low')

        self.filter = Filter(b, a, channels=channel)
        logger.debug(b)
        logger.debug(a)
        self.state = ST_INIT
        self.oneg = 32768 / global_settings.sensor.g_range

        self.ox = node_settings.accelerometer.offset_x
        self.oy = node_settings.accelerometer.offset_y
        self.oz = node_settings.accelerometer.offset_z
        #print (node_settings)

        self.g_force = 0

    def make_settings(self):
        if not os.path.isfile('cfg/algorithm_g.json'):
            m = DotMap()
            m.g_threshold = 0.1 
            m.filter.n  = 10
            m.filter.Wn_Hz = 10
            save_settings(m, 'algorithm_g')

    def counts_to_g(self, counts):
        return float(counts / self.oneg)

    def feed(self, value):
        if self.state == ST_RUNNING:
            ax, ay, az = value

            ax += self.ox
            ay += self.oy
            az += self.oz

            # print ('aa',ax)
            # print ('ab',ay)
            # print ('ac',az)

            fx, fy, fz = self.filter([ax, ay, az]).tolist()
            tot = math.sqrt(fx*fx + fy*fy + fz*fz)

            # show g force if total acceleration != 1g
            if True:#abs(tot-self.oneg) > (self.oneg * self.settings.g_threshold):
                logger.debug(tot)
                self.g_force = self.counts_to_g(fx)
            else:
                self.g_force = 0
 
        else:
            for _ in range(100):
                fx, fy, fz = self.filter(value).tolist()

            self.g_force = self.counts_to_g(fx)
            self.state = ST_RUNNING
        

def test_offline(fname):
    import numpy as np 
    data = np.loadtxt(fname, delimiter=';', comments='#', usecols=[2,3,4,5,6,7])
    lateral_g = Algorithm_g_force(['ax','ay','az'])
    longitudinal_g = Algorithm_g_force(['ax','ay','az'])
    data1=data[0:600,:]
    for ax,ay,az,mx,my,mz in data1:
        lateral_g.feed((ay,ax,az)) # turn g force
        longitudinal_g.feed((ax,ay,az)) # accel / brake g force
        print ('%.02f\t%0.2f' % (lateral_g.g_force, longitudinal_g.g_force))

def test_online():
    import imports  # pylint: disable=unused-import
    from kx_lib.kx_board import ConnectionManager
    from kx_lib.kx_util import evkit_config
    from kmx62 import kmx62_data_logger
    from kmx62.kmx62_driver import KMX62Driver

    class CallBack(object):
        def __init__(self):
            self.lateral_g = Algorithm_g_force(['ax','ay','az'])
            self.longitudinal_g = Algorithm_g_force(['ax','ay','az'])

        def callback(self, data):
            ch, ax, ay, az, mx, my, mz, temp = data
            self.lateral_g.feed((ay,ax,az))
            self.longitudinal_g.feed((ax,ay,az))
            print ('%.02f\t%0.2f' % (self.lateral_g.g_force, self.longitudinal_g.g_force))

    callback_object = CallBack()

    connection_manager = ConnectionManager(board_config_json='rokix_board_rokix_sensor_node_i2c.json')
    sensor_kmx62 = KMX62Driver()
    try:
        connection_manager.add_sensor(sensor_kmx62)
        kmx62_data_logger.enable_data_logging(
            sensor_kmx62, 
            odr=global_settings.sensor.odr,
            max_range_acc = '%dG' % global_settings.sensor.g_range
        )

        stream = kmx62_data_logger.KMX62DataStream([sensor_kmx62])
        stream.read_data_stream(console=False, callback=callback_object.callback)

    finally:
        sensor_kmx62.set_power_off()
        connection_manager.disconnect()

if __name__ == '__main__':
    #test_offline('testdata/kmx62_50Hz_accel_skid_y.csv')
    test_online()
