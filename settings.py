from dotmap import DotMap
import json


def load_settings(fname='global_settings'):
    j = json.load(open(('cfg/%s.json' % fname),'r'))
    settings = DotMap(j)
    return settings
    
def save_settings(m, fname='global_settings'):
    json.dump(m, open('cfg/%s.json' % fname,'w'), indent=True)

def make_settings():
    m = DotMap()
    m.accelerometer.offset_x = 0
    m.accelerometer.offset_y = 0
    m.accelerometer.offset_z = 0

    m.magnetometer.offset_x = 0
    m.magnetometer.offset_y = 0
    m.magnetometer.offset_z = 0
    save_settings(m, 'node_settings')

    m = DotMap()
    m.sensor.g_range = 4
    m.sensor.odr = 100

    m.data.source = 'gui' # 'gui' or 'cli'
    m.display.fps = 50

    save_settings(m)

def test():
    make_settings()
    settings = load_settings()
    print (settings.pprint('json'))

if __name__ == '__main__':
    test()
