from basic_stuff import *
import time, random, datetime
from test1 import MyFrame, wx
from receiver_thread import UpdateBarEvent, EVT_UPDATE_BARGRAPH, CalcBarThread
from receiver_thread import UpdateDataEvent, EVT_DATA, ReceiverThread
from receiver_thread import UpdateTimeEvent, EVT_UPDATE_TIME, TimerThread
global_settings = load_settings()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestFrame(MyFrame):
    def __init__(self, *args, **kwds):
        MyFrame.__init__(self, *args, **kwds)
        self.Bind(EVT_UPDATE_BARGRAPH, self.update_bar)
        self.Bind(EVT_DATA, self.new_data)
        self.Bind(EVT_UPDATE_TIME, self.new_time)
        self.odr_counter=-1
        self.prev_odr_count_time = datetime.timedelta(minutes=0)

        self.smooth_terrain_bitmap = wx.Bitmap(".\\graphics\\terrain_smooth.bmp", wx.BITMAP_TYPE_ANY)
        self.rough_terrain_bitmap = wx.Bitmap(".\\graphics\\terrain_rough.bmp", wx.BITMAP_TYPE_ANY)
        self.terrain_bitmap_set=0

        self.car_moving_bitmap = wx.Bitmap(".\\graphics\\car_moving.bmp", wx.BITMAP_TYPE_ANY)
        self.car_stopping_bitmap = wx.Bitmap(".\\graphics\\car_stationary.bmp", wx.BITMAP_TYPE_ANY)
        self.car_status_bitmap_set=0

        self.threads = []
        self.threads.append(CalcBarThread(self, 0, 50))
        self.threads.append(TimerThread(self))
        # outcomment this if want to test without sensor connection        
        self.threads.append(ReceiverThread(self, 0, 50))

        self.fps_odr_counter = 0
        self.fps_odr_decimator = global_settings.sensor.odr / global_settings.display.fps
        
        for t in self.threads:
            t.Start()

        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

    def OnCloseWindow(self, evt):
        busy = wx.BusyInfo("One moment please, waiting for threads to die...")
        wx.Yield()

        for t in self.threads:
            t.Stop()

        running = 1

        while running:
            running = 0

            for t in self.threads:
                running = running + t.IsRunning()

            time.sleep(0.1)

        self.Destroy()

    def update_bar(self, evt):
        pass

    def new_data(self, evt):
        self.fps_odr_counter += 1

        # calculate sensor data ODR received
        if (evt.lap_curr_time - self.prev_odr_count_time).seconds > 5:
            logger.debug ('odr %f' % (self.odr_counter / 6))
            self.odr_counter=0
            self.prev_odr_count_time = evt.lap_curr_time
        else:
            self.odr_counter+=1

        # update display only with FPS, this way ODR can be higer 
        if self.fps_odr_counter < self.fps_odr_decimator:
            return
        self.fps_odr_counter = 0

        lat = max(-400,evt.lat*1000) if evt.lat < 0 else min(400, evt.lat*1000) 
        lon = max(-400, evt.lon*1000) if evt.lon < 0 else min(400, evt.lon*1000)
        #logger.debug("%f %f" % (lat,lon))

        self.speed_meter.SetSpeedValue(lat)
        self.speed_meter2.SetSpeedValue(lon)
        
        if evt.lap_best_time:
            #print ('best',evt.lap_best_time)
            self.LEDNumbers_best.SetValue(evt.lap_best_time)

        if evt.lap_prev_time:
            #print ('prev',evt.lap_prev_time)
            self.LEDNumbers_prev.SetValue(evt.lap_prev_time)

        
        #print (evt.lap_curr_time)
        self.LEDNumbers_current.SetValue(str(evt.lap_curr_time))


        ### t
        if evt.car_moving:
            if self.car_status_bitmap_set != 1:
                self.car_status_bitmap_set = 1
                self.graphics_car_status.SetBitmap(self.car_moving_bitmap)
        else:
            if self.car_status_bitmap_set != 0:
                self.car_status_bitmap_set = 0
                self.graphics_car_status.SetBitmap(self.car_stopping_bitmap)
        if evt.terrain_smooth:
            if self.terrain_bitmap_set != 1:
                self.terrain_bitmap_set = 1
                self.graphics_terrain.SetBitmap(self.smooth_terrain_bitmap)
        else:
            if self.terrain_bitmap_set !=0:
                self.terrain_bitmap_set = 0
                self.graphics_terrain.SetBitmap(self.rough_terrain_bitmap)
        ### m
        if evt.rollover_time: # is not None
            self.rollover_output.Clear()
            self.rollover_output.write('\nRollover Detected %s\n' % evt.rollover_time)
            self.rollover_output.write(str(evt.rollover_blackbox))
            self.rollover_output.write('\n')
            self.rollover_output.write('\n')

        ###
        if evt.driver_profile is not None:
            self.driver_profile.Clear()
            driver_type, max_x_g_force, max_y_g_force, lap_speed, \
                peak_x_acceleration,  peak_y_acceleration = evt.driver_profile

            self.driver_profile.write(driver_profile_str % (
                driver_type, max_x_g_force, max_y_g_force, lap_speed, \
                    peak_x_acceleration,  peak_y_acceleration
            ))
        
        if evt.lap_completed:
            self.lane.write(evt.lap_completed)
            self.lane.write('\n')



    def new_time(self, evt):
        #self.LEDNumbers_current.SetValue('%07.02f' % evt.value)
        pass

driver_profile_str = "type %ss\nmaxx %f\nmaxy %f\nlap_speed %f\n peakx %f, peaky %f"

class MyApp(wx.App):
    def OnInit(self):
        self.frame = TestFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# end of class MyApp

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
