from kmx62_sample_algorithm_jacob import Algorithm
import plot

# work around to plot...
class kwargs:
    column_separator=';'
    column_header=None
plot.kwargs = kwargs

def main():
    a=Algorithm(None, None)

    data = plot.loader('../Slot Car Tests (Jacob)/kmx62_50Hz_accel_skid_y.txt')
    for row in data.iterrows():
        row_number, data = row
        ax, ay, az, mx, my, mz, temp = data['ax'], data['ay'], data['az'], data['mx'], data['my'], data['mz'], data['temp']
        #print(row_number, ax, ay, az, mx, my, mz, temp)
        a.feed([10, ax, ay, az, mx, my, mz, temp])

if __name__ == '__main__':
    main()