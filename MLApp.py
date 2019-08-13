from tkinter import *
from tkinter.ttk import Combobox
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import data_preprocessing
import feature_extraction
import train
import time
import threading
import pickle
from sklearn.metrics import accuracy_score
import matplotlib
import gogogopredict

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

HEIGHT = 600
WIDTH = 1440

#Create a tkinter window poping out
root = Tk()
root.title("GUI")
# Set minimum size of the window in case of text distortion
root.minsize(WIDTH, HEIGHT)

canvas = Canvas(root, height=HEIGHT, width=WIDTH)
canvas.pack()

# Create frames for 
frame_pre = Frame(root, bg='#CFD8DC')
frame_pre.place(relwidth=0.3, relheight=1.0)

frame_axis = Frame(root, bg='#E0F2F1')
frame_axis.place(relx=0.3, relwidth=0.3, relheight=0.3)

frame_fe = Frame(root, bg='#E0F2F1')
frame_fe.place(relx=0.3, rely=0.3, relwidth=0.3, relheight=0.7)

frame_trn = Frame(root, bg='#B2DFDB')
frame_trn.place(relx=0.6, relwidth=0.4, relheight=0.5)

frame_tst = Frame(root, bg='#80CBC4')
frame_tst.place(relx=0.6, rely=0.5, relwidth=0.4, relheight=0.5)


##############################
#### title of each frame ####
##############################
pre = LabelFrame(frame_pre, text='Preprocessing', font=("Helvetica", "25", "bold italic"))
pre.pack(fill=BOTH, expand=True)

axis = LabelFrame(frame_axis, text='Axis Selection', font=("Helvetica", "25", "bold italic"))
axis.pack(fill=BOTH, expand= True)

fe = LabelFrame(frame_fe, text='Feature Extraction', font=("Helvetica", "25", "bold italic"))
fe.pack(fill=BOTH, expand=True)

trn = LabelFrame(frame_trn, text='Training', font=("Helvetica", "25", "bold italic"))
trn.pack(fill=BOTH, expand=True)

tst = LabelFrame(frame_tst, text='Prediction', font=("Helvetica", "25", "bold italic"))
tst.pack(fill=BOTH, expand=True)


############################
#### preprocessing ####
############################
# filter
# Create a variable to get value from the checkbox
var_filter = IntVar()

#Initialize labels and entries in the filter section
#entries are gray and non-changeable by default
lb_order = Label(pre, text='Order', font=("Helvetica", "12"))
lb_order.place(relx=0.1, rely=0.13)
en_order = Entry(pre, bd=2, state=DISABLED)
en_order.place(relx=0.22, rely=0.13)

lb_fc = Label(pre, text='fc', font=("Helvetica", "12", "italic"))
lb_fc.place(relx=0.1, rely=0.18)
en_fc = Entry(pre, bd=2, state=DISABLED)
en_fc.place(relx=0.22, rely=0.18)
lb_fc_unit = Label(pre, text='Hz', font=("Helvetica", "12", "italic"))
lb_fc_unit.place(relx=0.52, rely=0.18)

lb_type = Label(pre, text='Type', font=("Helvetica", "12"))
lb_type.place(relx=0.1, rely=0.23)
en_type = Combobox(pre, values=["lowpass", "highpass", "bandpass", "bandstop"], state=DISABLED)
en_type.place(relx=0.22, rely=0.23)

lb_fs = Label(pre, text='fs', font=("Helvetica", "12", "italic"))
lb_fs.place(relx=0.1, rely=0.28)
en_fs = Entry(pre, bd=2, state=DISABLED)
en_fs.place(relx=0.22, rely=0.28)
lb_fc_unit = Label(pre, text='Hz', font=("Helvetica", "12", "italic"))
lb_fc_unit.place(relx=0.52, rely=0.28)

def activateCheck_filter():
#entry bar is active if the checkbox is selected (NORMAL)
	if var_filter.get() == 1:
		en_order.config(state=NORMAL)
		en_fc.config(state=NORMAL)
		en_type.config(state=NORMAL)
		en_fs.config(state=NORMAL)
#entry bar is gray if the checkbox is not selected (DISABLED)
	elif var_filter.get() == 0:
		en_order.config(state=DISABLED)
		en_fc.config(state=DISABLED)
		en_type.config(state=DISABLED)
		en_fs.config(state=DISABLED)
#change status of the filter section
filterChk = Checkbutton(pre, text='Filter', font=("Helvetica", "18"), variable=var_filter, command=activateCheck_filter)
filterChk.place(relx=0.05, rely=0.05)


#Initialize labels and entries in the smoothing section
var_smooth = IntVar()

lb_wl = Label(pre, text='Window Length', font=("Helvetica", "12"))
lb_wl.place(relx=0.1, rely=0.46)
en_wl = Entry(pre, bd=2, state=DISABLED)
en_wl.place(relx=0.38, rely=0.46)

lb_po = Label(pre, text='Poly Order', font=("Helvetica", "12"))
lb_po.place(relx=0.1, rely=0.51)
en_po = Entry(pre, bd=2, state=DISABLED)
en_po.place(relx=0.38, rely=0.51)

lb_mode = Label(pre, text='Mode', font=("Helvetica", "12"))
lb_mode.place(relx=0.1, rely=0.56)
en_mode = Combobox(pre, values=["mirror", "nearest", "wrap", "constant", "interp"], state=DISABLED)
en_mode.place(relx=0.38, rely=0.56)

def activateCheck_smooth():
	if var_smooth.get() == 1:
		en_wl.config(state=NORMAL)
		en_po.config(state=NORMAL)
		en_mode.config(state=NORMAL)
		elimChk.config(state=NORMAL)
	elif var_smooth.get() == 0:
		en_wl.config(state=DISABLED)
		en_po.config(state=DISABLED)
		en_mode.config(state=DISABLED)
		elimChk.config(state=DISABLED)

smoothChk = Checkbutton(pre, text='Smoothing', font=("Helvetica", "18"), variable=var_smooth, command=activateCheck_smooth)
smoothChk.place(relx=0.05, rely=0.38)

# eliminate
var_elim = IntVar()

lb_thre = Label(pre, text='Threshold', font=("Helvetica", "12"))
lb_thre.place(relx=0.1, rely=0.74)
en_thre = Entry(pre, bd=2, state=DISABLED)
en_thre.place(relx=0.35, rely=0.74)

lb_win = Label(pre, text='Window Size', font=("Helvetica", "12"))
lb_win.place(relx=0.1, rely=0.79)
en_win = Entry(pre, bd=2, state=DISABLED)
en_win.place(relx=0.35, rely=0.79)

var_base = IntVar()
var_base.set(1)
lb_base = Label(pre, text='Criterion Axis', font=("Helvetica", "12"))
lb_base.place(relx=0.1, rely=0.84)
rb_x = Radiobutton(pre, text='X Axis', variable=var_base, value=1, state=DISABLED)
rb_y = Radiobutton(pre, text='Y Axis', variable=var_base, value=2, state=DISABLED)
rb_z = Radiobutton(pre, text='Z Axis', variable=var_base, value=3, state=DISABLED)
rb_x.place(relx=0.35, rely=0.84)
rb_y.place(relx=0.5, rely=0.84)
rb_z.place(relx=0.65, rely=0.84)

def activateCheck_elim():
	if var_elim.get() == 1:
		en_thre.config(state=NORMAL)
		en_win.config(state=NORMAL)
		rb_x.config(state=NORMAL)
		rb_y.config(state=NORMAL)
		rb_z.config(state=NORMAL)
	elif var_elim.get() == 0:
		en_thre.config(state=DISABLED)
		en_win.config(state=DISABLED)
		rb_x.config(state=DISABLED)
		rb_y.config(state=DISABLED)
		rb_z.config(state=DISABLED)

elimChk = Checkbutton(pre, text='Eliminate Abnormal Data', font=("Helvetica", "18"), variable=var_elim, command=activateCheck_elim, state=DISABLED)
elimChk.place(relx=0.05, rely=0.66)


# Plot figures after data processing
def plotFigures():
	file_idx = trn_files.curselection()
	filename = ""
	if file_idx == ():
		messagebox.showerror("Error", "Please select the training files!")
		return
	else:
		filename = trn_files.get(file_idx)


	if var_filter.get() == 1:
		if en_order.get() != "" and en_fc.get() != "" and en_type.get() != "" and en_fs.get() != "":
			filter_order = int(en_order.get())
			filter_fc = en_fc.get()
			filter_type = en_type.get()
			filter_fs = int(en_fs.get())
			if filter_type == "bandpass" or filter_type == "bandstop":
				if filter_fc.split("[")[0] != "":
					messagebox.showerror("Error", "The form of fc should be like \'[fc1, fc2]\'!")
					return

				fc = filter_fc.split("[")[1].split("]")[0].split(",")
				filter_fc = [int(fc[0]), int(fc[1])]
			else:
				filter_fc = int(filter_fc)

		else:
			messagebox.showerror("Error", "Please set all the parameters of Filter!")
			return

	if var_smooth.get() == 1:
		if en_wl.get() != "" and en_po.get() != "" and en_mode.get() != "":
			smooth_wl = int(en_wl.get())
			smooth_po = int(en_po.get())
			smooth_mode = en_mode.get()

			if smooth_wl % 2 == 0:
				messagebox.showerror("Error", "The window length should be odd!")
				return


		else:
			messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
			return

	if var_elim.get() == 1:
		if en_thre.get() != "" and en_win.get() != "":
			elim_thre = int(en_thre.get())
			elim_ws = int(en_win.get())
		else:
			messagebox.showerror("Error", "Please set all the parameters of Eliminate Abnormal Data!")
			return


	files = pd.read_csv(filename)
	num_rows = files.shape[0]

	for idx, row in files.iterrows():
		name = row['file']
		dir = row['dir']
		label = row['label']
		header = row['header']
		raw_data = pd.read_csv(dir + '\\' + '\\' + name, header=header, delimiter=';')

		ax = raw_data['ax']
		ay = raw_data['ay']
		az = raw_data['az']

		data = np.array([ax, ay, az]).T

		top = Toplevel(root)
		top.title("Figure " + str(idx+1) + " / " + str(num_rows))

		f = Figure(figsize=(14, 8), dpi=100)

		sub = var_filter.get() + var_smooth.get() + var_elim.get() + 1
		if var_elim.get() == 1:
			sub += 1

		axes = f.subplots(sub, 1)

		if sub == 1:
			axes.plot(data[:, 0], label='raw ax - ' + label, alpha=0.8)
			axes.plot(data[:, 1], label='raw ay - ' + label, alpha=0.8)
			axes.plot(data[:, 2], label='raw az - ' + label, alpha=0.8)
			axes.legend()
			axes.grid()
		else:
			axes[0].plot(data[:, 0], label='raw ax - ' + label, alpha=0.8)
			axes[0].plot(data[:, 1], label='raw ay - ' + label, alpha=0.8)
			axes[0].plot(data[:, 2], label='raw az - ' + label, alpha=0.8)
			axes[0].legend()
			axes[0].grid()
		index = 1

		if var_filter.get() == 1:
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)
			axes[index].plot(data[:, 0], label='filtered ax - ' + label, alpha=0.8)
			axes[index].plot(data[:, 1], label='filtered ay - ' + label, alpha=0.8)
			axes[index].plot(data[:, 2], label='filtered az - ' + label, alpha=0.8)
			axes[index].legend()
			axes[index].grid()
			index += 1

		if var_smooth.get() == 1:
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)
			axes[index].plot(smoothed_data[:, 0], label='smoothed ax - ' + label, alpha=0.8)
			axes[index].plot(smoothed_data[:, 1], label='smoothed ay - ' + label, alpha=0.8)
			axes[index].plot(smoothed_data[:, 2], label='smoothed az - ' + label, alpha=0.8)
			axes[index].legend()
			axes[index].grid()
			axes[index].set_ylim((-6000, 6000))
			index += 1

			if var_elim.get() == 1:
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, var_base.get())
				axes[index].plot(data[:, 0], label='after eliminate ax - ' + label, alpha=0.8)
				axes[index].plot(data[:, 1], label='after eliminate ay - ' + label, alpha=0.8)
				axes[index].plot(data[:, 2], label='after eliminate az - ' + label, alpha=0.8)
				axes[index].legend()
				axes[index].grid()
				index += 1
				axes[index].plot(new_smoothed_data[:, 0], label='after eliminate smoothed ax - ' + label, alpha=0.8)
				axes[index].plot(new_smoothed_data[:, 1], label='after eliminate smoothed ay - ' + label, alpha=0.8)
				axes[index].plot(new_smoothed_data[:, 2], label='after eliminate smoothed az - ' + label, alpha=0.8)
				axes[index].legend()
				axes[index].grid()
				index += 1

		
		canvas_fig = FigureCanvasTkAgg(f, top)
		canvas_fig.draw()
		canvas_fig.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

		toolbar = NavigationToolbar2Tk(canvas_fig, top)
		toolbar.update()
		canvas_fig._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)

		# xChk = Checkbutton(canvas_fig.get_tk_widget(), text='x axis', font=("Helvetica", "11"))
		# xChk.place(relx=0.15, rely=0.05)
		# yChk = Checkbutton(canvas_fig.get_tk_widget(), text='y axis', font=("Helvetica", "11"))
		# yChk.place(relx=0.2, rely=0.05)
		# zChk = Checkbutton(canvas_fig.get_tk_widget(), text='z axis', font=("Helvetica", "11"))
		# zChk.place(relx=0.25, rely=0.05)
		

plot = Button(pre, text='Plot', font=("Helvetica", "12", "bold"), padx=4, command=plotFigures)
plot.place(relx=0.85, rely=0.92)



########################
#### AXIS SELECTION ####
########################
var_accel_x = IntVar()
accelXChk = Checkbutton(axis, text='X Acceleration', font=("Helvetica", "15"), variable=var_accel_x)
accelXChk.place(relx=0.05, rely=0.06)

var_accel_y = IntVar()
accelYChk = Checkbutton(axis, text= 'Y Acceleration', font=("Helvetica", "15"), variable= var_accel_y)
accelYChk.place(relx= 0.05, rely=0.28)

var_accel_z = IntVar()
accelZChk = Checkbutton(axis, text= 'Z Acceleration', font=("Helvetica", "15"), variable= var_accel_z)
accelZChk.place(relx= 0.05, rely=0.5)



################################
#### feature extract ####
################################
var_mean = IntVar()
meanChk = Checkbutton(fe, text='Mean Value', font=("Helvetica", "15"), variable=var_mean)
meanChk.place(relx=0.05, rely=0.05)

var_std = IntVar()
stdChk = Checkbutton(fe, text='Standard Divation', font=("Helvetica", "15"), variable=var_std)
stdChk.place(relx=0.05, rely=0.13)

var_min = IntVar()
minChk = Checkbutton(fe, text='Minimum Value', font=("Helvetica", "15"), variable=var_min)
minChk.place(relx=0.05, rely=0.21)

var_max = IntVar()
maxChk = Checkbutton(fe, text='Maximum Value', font=("Helvetica", "15"), variable=var_max)
maxChk.place(relx=0.05, rely=0.29)

var_rms = IntVar()
rmsChk = Checkbutton(fe, text='Root Mean Square', font=("Helvetica", "15"), variable=var_rms)
rmsChk.place(relx=0.05, rely=0.37)

var_energy = IntVar()
lb_bs = Label(fe, text='Band Size', font=("Helvetica", "11"))
lb_bs.place(relx=0.1, rely=0.53)
en_bs = Entry(fe, bd=2, state=DISABLED)
en_bs.place(relx=0.28, rely=0.53)

lb_odr = Label(fe, text='ODR', font=("Helvetica", "11"))
lb_odr.place(relx=0.1, rely=0.59)
lb_odr_unit = Label(fe, text='Hz', font=("Helvetica", "11"))
lb_odr_unit.place(relx=0.58, rely=0.59)
en_odr = Entry(fe, bd=2, state=DISABLED)
en_odr.place(relx=0.28, rely=0.59)

def activateCheck_energy():
	if var_energy.get() == 1:
		en_bs.config(state=NORMAL)
		en_odr.config(state=NORMAL)
	elif var_energy.get() == 0:
		en_bs.config(state=DISABLED)
		en_odr.config(state=DISABLED)

energyChk = Checkbutton(fe, text='Energy', font=("Helvetica", "15"), variable=var_energy, command=activateCheck_energy)
energyChk.place(relx=0.05, rely=0.45)

# Set data processing window size
lb_ws = Label(fe, text='Segment Size', font=("Helvetica", "15"))
lb_ws.place(relx=0.05, rely=0.7)
en_ws = Entry(fe, bd=2)
en_ws.place(relx=0.35, rely=0.71)



# extract
def extractAndSave():
	file_idx = trn_files.curselection()
	filename = ""
	if file_idx == ():
		messagebox.showerror("Error", "Please select the training files!")
		return
	else:
		filename = trn_files.get(file_idx)


	if var_filter.get() == 1:
		if en_order.get() != "" and en_fc.get() != "" and en_type.get() != "" and en_fs.get() != "":
			filter_order = int(en_order.get())
			filter_fc = en_fc.get()
			filter_type = en_type.get()
			filter_fs = int(en_fs.get())
			if filter_type == "bandpass" or filter_type == "bandstop":
				if filter_fc.split("[")[0] != "":
					messagebox.showerror("Error", "The form of fc should be like \'[fc1, fc2]\'!")
					return

				fc = filter_fc.split("[")[1].split("]")[0].split(",")
				filter_fc = [int(fc[0]), int(fc[1])]
			else:
				filter_fc = int(filter_fc)

		else:
			messagebox.showerror("Error", "Please set all the parameters of Filter!")
			return

	if var_smooth.get() == 1:
		if en_wl.get() != "" and en_po.get() != "" and en_mode.get() != "":
			smooth_wl = int(en_wl.get())
			smooth_po = int(en_po.get())
			smooth_mode = en_mode.get()

			if smooth_wl % 2 == 0:
				messagebox.showerror("Error", "The window length should be odd!")
				return


		else:
			messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
			return

	if var_elim.get() == 1:
		if en_thre.get() != "" and en_win.get() != "":
			elim_thre = int(en_thre.get())
			elim_ws = int(en_win.get())
		else:
			messagebox.showerror("Error", "Please set all the parameters of Eliminate Abnormal Data!")
			return

	if var_energy.get() == 1:
		if en_bs.get() != "" and en_odr.get() != "":
			eng_bs = int(en_bs.get())
			eng_odr = int(en_odr.get())
		else:
			messagebox.showerror("Error", "Please set all the parameters of Energy!")
			return

	if en_ws.get() != "":
		seg_size = int(en_ws.get())
	else:
		messagebox.showerror("Error", "Please set the segment size!")
		return

	if var_accel_x.get() == 0 and var_accel_y.get() == 0 and var_accel_z.get() == 0:
		messagebox.showerror("Error", "Please select at least one axis!")
		return

	if (var_base.get() == 1 and var_accel_x.get() == 0) or (var_base.get() == 2 and var_accel_y.get() == 0) or (var_base.get() == 3 and var_accel_z.get() == 0):
		messagebox.showerror("Error", "Please select the axis that is select in Axis Selection!")
		return

	if var_mean.get() + var_std.get() + var_min.get() + var_max.get() + var_rms.get() + var_energy.get() == 0:
		messagebox.showerror("Error", "Please select at least one feature!")
		return

	save_features = filedialog.asksaveasfilename(initialdir = "/", title = "Save file", filetypes = (("numpy files","*.npy"), ("all files", "*.*")))
	if save_features == "":
		return


	files = pd.read_csv(filename)

	final_features = []
	labels = []

	for idx, row in files.iterrows():
		name = row['file']
		dir = row['dir']
		label = row['label']
		header = row['header']
		raw_data = pd.read_csv(dir + '\\' + '\\' + name, header=header, delimiter=';')

		ax = raw_data['ax']
		ay = raw_data['ay']
		az = raw_data['az']

		data = np.array([ax, ay, az]).T

		if var_filter.get() == 1:
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

		if var_smooth.get() == 1:
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

			if var_elim.get() == 1:
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, var_base.get())
			else:
				data = smoothed_data


		num = var_mean.get() + var_std.get() + var_min.get() + var_max.get() + var_rms.get()


		for col in range(3):
			feature = np.empty([int(len(data[:, col])/seg_size), num])
			new_feature = []
			for i in range(int(len(data[:, col])/seg_size)):
				window = data[i*seg_size:(i+1)*seg_size, col]
				k = 0
				if var_mean.get():
					feature[i, k] = feature_extraction.mean(window)
					k += 1;
				if var_std.get():
					feature[i, k] = feature_extraction.std(window)
					k += 1;
				if var_min.get():
					feature[i, k] = feature_extraction.getmin(window)
					k += 1;
				if var_max.get():
					feature[i, k] = feature_extraction.getmax(window)
					k += 1;
				if var_rms.get():
					feature[i, k] = feature_extraction.rms(window)
					k += 1;
				if var_energy.get():
					energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
					temp = np.append(feature[i, :], energy_vector)
					new_feature.append(temp)
				else:
					new_feature.append(feature[i, :])

			new_feature = np.array(new_feature)
			#print("new_feature: ", new_feature.shape)
			
			if col == 0:
				ex_feature = new_feature
			else:
				ex_feature = np.append(ex_feature, new_feature, axis=1)
			
			#print("ex_feature: ", ex_feature.shape)

		num_features = ex_feature.shape[1] // 3

		num_axis = var_accel_x.get() + var_accel_y.get() + var_accel_z.get()

		trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

		k = 0

		if var_accel_x.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
			k += 1
		if var_accel_y.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
			k += 1
		if var_accel_z.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
			k += 1;

		final_features.extend(trn_feature)
		for i in range(len(trn_feature)):
			labels.append(label)

	final_features = np.array(final_features)
	labels = np.array(labels)

	save_labels = save_features

	save_features = save_features.split(".")[0] + "_features.npy"
	save_labels = save_labels.split(".")[0] + "_labels.npy"

	np.save(save_features, final_features)
	np.save(save_labels, labels)

	messagebox.showinfo("Congratulations", "Features and labels are successfully saved!")

#Set up extract button and call function extractAndSave 
bt_ext = Button(fe, text='Extract', bd=2, font=("Helvetica", "12", "bold"), command=extractAndSave)
bt_ext.place(relx=0.8, rely=0.88)


########################
#### Training ####
########################
# model
lb_model = Label(trn, text='Model', font=("Helvetica", "12"))
lb_model.place(relx=0.05, rely=0.05)
en_model = Combobox(trn, values=["Random Forest", "SVM"])
en_model.place(relx=0.15, rely=0.05)

# data
scrollbar_x = Scrollbar(trn, orient=HORIZONTAL)
scrollbar_x.pack(side=BOTTOM, fill=X)
scrollbar_y = Scrollbar(trn, orient=VERTICAL)
scrollbar_y.pack(side=RIGHT, fill=Y)

trn_files = Listbox(trn, height=3, selectmode=SINGLE, width=32)
trn_files.place(relx=0.05, rely=0.30)

scrollbar_x.config(command=trn_files.xview)
scrollbar_y.config(command=trn_files.yview)
trn_files.config(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)


def addFile():
	filename = filedialog.askopenfilename(initialdir="/", title="Select training file", filetypes=(("csv files", "*.csv"), ("numpy files", "*.npy")))
	if filename == "":
		return
	trn_files.insert(END, filename)

def deleteFile():
	idx = trn_files.curselection()
	i = len(idx) - 1
	while(i>=0):
		trn_files.delete(idx[i])
		i -= 1

bt_add = Button(trn, text='Add Files', font=("Helvetica", "12"), command=addFile)
bt_add.place(relx=0.05, rely=0.18)
bt_delete = Button(trn, text='Delete Files', font=("Helvetica", "12"), command=deleteFile)
bt_delete.place(relx=0.22, rely=0.18)


# start training button
def startTrain():
	model =  en_model.get()
	if model == "":
		messagebox.showerror("Error", "Please select the training model!")
		return

	trn_ratio = var_ratio.get()

	file_idx = trn_files.curselection()
	filename = ""
	if file_idx == ():
		messagebox.showerror("Error", "Please select the training files!")
		return
	else:
		filename = trn_files.get(file_idx)

# Detect if a filter is used and pass all filter parameter to the function
	if var_filter.get() == 1:
		if en_order.get() != "" and en_fc.get() != "" and en_type.get() != "" and en_fs.get() != "":
			filter_order = int(en_order.get())
			filter_fc = en_fc.get()
			filter_type = en_type.get()
			filter_fs = int(en_fs.get())
			if filter_type == "bandpass" or filter_type == "bandstop":
				if filter_fc.split("[")[0] != "":
					messagebox.showerror("Error", "The form of fc should be like \'[fc1, fc2]\'!")
					return

				fc = filter_fc.split("[")[1].split("]")[0].split(",")
				filter_fc = [int(fc[0]), int(fc[1])]
			else:
				filter_fc = int(filter_fc)

		else:
			messagebox.showerror("Error", "Please set all the parameters of Filter!")
			return

# Detect if smoothing is used and pass all filter parameter to the function
	if var_smooth.get() == 1:
		if en_wl.get() != "" and en_po.get() != "" and en_mode.get() != "":
			smooth_wl = int(en_wl.get())
			smooth_po = int(en_po.get())
			smooth_mode = en_mode.get()

			if smooth_wl % 2 == 0:
				messagebox.showerror("Error", "The window length should be odd!")
				return

		else:
			messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
			return

	if var_elim.get() == 1:
		if en_thre.get() != "" and en_win.get() != "":
			elim_thre = int(en_thre.get())
			elim_ws = int(en_win.get())
		else:
			messagebox.showerror("Error", "Please set all the parameters of Eliminate Abnormal Data!")
			return

	if var_energy.get() == 1:
		if en_bs.get() != "" and en_odr.get() != "":
			eng_bs = int(en_bs.get())
			eng_odr = int(en_odr.get())
		else:
			messagebox.showerror("Error", "Please set all the parameters of Energy!")
			return

	if en_ws.get() != "":
		seg_size = int(en_ws.get())
	else:
		messagebox.showerror("Error", "Please set the segment size!")
		return

	if var_accel_x.get() == 0 and var_accel_y.get() == 0 and var_accel_z.get() == 0:
		messagebox.showerror("Error", "Please select at least one axis!")
		return

	if (var_base.get() == 1 and var_accel_x.get() == 0) or (var_base.get() == 2 and var_accel_y.get() == 0) or (var_base.get() == 3 and var_accel_z.get() == 0):
		messagebox.showerror("Error", "Please select the axis that is select in Axis Selection!")
		return

	if var_mean.get() + var_std.get() + var_min.get() + var_max.get() + var_rms.get() + var_energy.get() == 0:
		messagebox.showerror("Error", "Please select at least one feature!")
		return


	save_model = filedialog.asksaveasfilename(initialdir = "/", title = "Save Model", filetypes = (("sav files","*.sav"), ("all files", "*.*")))
	if save_model == "":
		return
	if len(save_model.split(".")) == 1:
		save_model = save_model + '.sav'

	### write model configuration
	parts = save_model.split('/')
	directory = '/'.join(parts[0:len(parts)-1])
	txtName = parts[len(parts)-1].split('.')[0] + '.txt'

	pb['value'] = 10
	trn.update_idletasks() 
	time.sleep(0.5)

	f = open(directory + '/' + txtName, "w+")
	f.write("param value\n")
	if var_filter.get() == 1:
		f.write("filter 1\n")
		f.write("order %d\n" %filter_order)
		f.write("fc %d\n" %filter_fc)
		f.write("type " + filter_type + "\n")
		f.write("fs %d\n" %filter_fs)
	else:
		f.write("filter 0\n")

	if var_smooth.get() == 1:
		f.write("smooth 1\n")
		f.write("windowlength %d\n" %smooth_wl)
		f.write("polyorder %d\n" %smooth_po)
		f.write("mode " + smooth_mode + "\n")
	else:
		f.write("smooth 0\n")

	if var_elim.get() == 1:
		f.write("eliminate 1\n")
		f.write("threshold %d\n" %elim_thre)
		f.write("windowsize %d\n" %elim_ws)
		f.write("baseon %d\n" %var_base.get())
	else:
		f.write("eliminate 0\n")

	f.write("X %d\n" %var_accel_x.get())
	f.write("Y %d\n" %var_accel_y.get())
	f.write("Z %d\n" %var_accel_z.get())

	f.write("mean %d\n" %var_mean.get())
	f.write("std %d\n" %var_std.get())
	f.write("min %d\n" %var_min.get())
	f.write("max %d\n" %var_max.get())
	f.write("rms %d\n" %var_rms.get())
	f.write("energy %d\n" %var_energy.get())

	if var_energy.get() == 1:
		f.write("bandsize %d\n" %eng_bs)
		f.write("odr %d\n" %eng_odr)

	f.write("segsize %d\n" %seg_size)

	f.close()


	files = pd.read_csv(filename)

	final_features = []
	labels = []

	num_rows = files.shape[0]

	for idx, row in files.iterrows():
		name = row['file']
		dir = row['dir']
		label = row['label']
		header = row['header']
		raw_data = pd.read_csv(dir + '\\' + '\\' + name, header=header, delimiter=';')

		ax = raw_data['ax']
		ay = raw_data['ay']
		az = raw_data['az']

		data = np.array([ax, ay, az]).T

		if var_filter.get() == 1:
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

		if var_smooth.get() == 1:
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

			if var_elim.get() == 1:
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, var_base.get())
			else:
				data = smoothed_data


		num = var_mean.get() + var_std.get() + var_min.get() + var_max.get() + var_rms.get()

# extract features from x y z axis respectively
		for col in range(3):
			# number of feature column is determined by selected features
			feature = np.empty([int(len(data[:, col])/seg_size), num])
			new_feature = []
			for i in range(int(len(data[:, col])/seg_size)):
				# window is one group of data whose size is seg_size
				window = data[i*seg_size:(i+1)*seg_size, col]
				# keep track of features
				k = 0
				if var_mean.get():
					feature[i, k] = feature_extraction.mean(window)
					k += 1;
				if var_std.get():
					feature[i, k] = feature_extraction.std(window)
					k += 1;
				if var_min.get():
					feature[i, k] = feature_extraction.getmin(window)
					k += 1;
				if var_max.get():
					feature[i, k] = feature_extraction.getmax(window)
					k += 1;
				if var_rms.get():
					feature[i, k] = feature_extraction.rms(window)
					k += 1;
				if var_energy.get():
					energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
					temp = np.append(feature[i, :], energy_vector)
					new_feature.append(temp)
				else:
					new_feature.append(feature[i, :])
            # features on only one axis
			new_feature = np.array(new_feature)
			#print("new_feature: ", new_feature.shape)
			# save features of all axes to ex_feature
			if col == 0:
				ex_feature = new_feature
			else:
				ex_feature = np.append(ex_feature, new_feature, axis=1)
			
			# print("ex_feature: ", ex_feature.shape)

		num_features = ex_feature.shape[1] // 3

		num_axis = var_accel_x.get() + var_accel_y.get() + var_accel_z.get()

		trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

		k = 0

		if var_accel_x.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
			k += 1
		if var_accel_y.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
			k += 1
		if var_accel_z.get() == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
			k += 1;

		final_features.extend(trn_feature)
		for i in range(len(trn_feature)):
			labels.append(label)

		pb['value'] = 10 + (idx+1)/num_rows * 80
		trn.update_idletasks() 
		time.sleep(0.5)

	final_features = np.array(final_features)
	labels = np.array(labels)

	#print(final_features.shape)
	#print(labels.shape)
	#np.savetxt('fea', final_features, delimiter=',')
	#print(labels)
	# f = open(directory + '/' + txtName, "w+")
	# f.write("param value\n")

	if res_text.get('1.0', END) != []:
		res_text.delete('1.0', END)

	if model == "Random Forest":
		acc_trn, acc_tst, acc_oob = train.RandomForest(final_features, labels, trn_ratio, save_model)
		res_text.insert(END, "Training accuracy: " + str(acc_trn) + '\n')
		res_text.insert(END, '\n' + "Testing accuracy: " + str(acc_tst) + '\n')
		res_text.insert(END, '\n' + "Out of bag accuracy: " + str(acc_oob) + '\n')

	if model == "SVM":
		acc_trn, acc_tst = train.SVM(final_features, labels, trn_ratio, save_model)
		res_text.insert(END, "Training accuracy: " + str(acc_trn) + '\n')
		res_text.insert(END, '\n' + "Test accuracy: " + str(acc_tst) + '\n')

	pb['value'] = 100
	trn.update_idletasks() 
	time.sleep(0.5)



bt_trn = Button(trn, text='START', font=("Helvetica", "15", "bold"), height=1, width=15, command=startTrain)
bt_trn.place(relx=0.05, rely=0.75)

# train set ratio
var_ratio = DoubleVar()
var_ratio.set(0.8)

scale_title = Label(trn, text='Ratio of training set: 0.8', font=("Helvetica", "12"))
scale_title.place(relx=0.05, rely=0.52)

def get_ratio(self):
	text = 'Ratio of training set: ' + str(var_ratio.get())
	scale_title.config(text=text)

scale = Scale(trn, from_=0.1, to=0.9, orient=HORIZONTAL, length=180, resolution=0.1, variable=var_ratio, showvalue=0, command=get_ratio)
scale.place(relx=0.05, rely=0.6)

# progress pane
lb_pb = Label(trn, text='Progress', font=("Helvetica", "12"))
lb_pb.place(relx=0.45, rely=0.05)
pb = ttk.Progressbar(trn, mode='determinate', orient=HORIZONTAL, length=200)
pb.place(relx=0.58, rely=0.05)

# output
result = LabelFrame(trn, text='Result', font=("Helvetica", "15", "bold italic"))
result.place(relx=0.45, rely=0.15, relheight=0.75, relwidth=0.5)
res_text = Text(result, font=("Helvetica", "11"), width=32, wrap=WORD, bd=0, height=9)
vscroll = ttk.Scrollbar(result, orient=VERTICAL, command=res_text.yview)
res_text['yscroll'] = vscroll.set
vscroll.pack(side=RIGHT, fill=Y)
res_text.place(relx=0.02, rely=0.01)


########################
#### Prediction ####
########################
offline = LabelFrame(tst, text='Offline', font=("Helvetica", "15", "bold italic"))
offline.place(relx=0, rely=0, relheight=0.5, relwidth=1.0)

online = LabelFrame(tst, text='Online', font=("Helvetica", "15", "bold italic"))
online.place(relx=0, rely=0.5, relheight=0.5, relwidth=1.0)

# offline

def start_predict_offline():
	
	if file_text.get("1.0", END) == []:
		messagebox.showerror("Error", "Please select test file!")
		return
	else:
		test_file = file_text.get("1.0", END)
		test_file = test_file.split("\n")[0]

	if model_text.get("1.0", END) == []:
		messagebox.showerror("Error", "Please select test model!")
		return
	else:
		offline_model = model_text.get("1.0", END)
		offline_model = offline_model.split("\n")[0]

	test_config = pd.read_csv(offline_model.split(".")[0] + ".txt", sep=" ")
	test_config.columns = ['param', 'value']

	test_config = pd.DataFrame(data=[test_config['value'].values], columns=test_config['param'].values)

	if test_config['filter'].values[0] == '1':
		filter_order = int(test_config['order'].values[0])
		filter_fc = test_config['fc'].values[0]
		filter_type = test_config['type'].values[0]
		filter_fs = int(test_config['fs'].values[0])
		if filter_type == "bandpass" or filter_type == "bandstop":
			fc = filter_fc.split("[")[1].split("]")[0].split(",")
			filter_fc = [int(fc[0]), int(fc[1])]
		else:
			filter_fc = int(filter_fc)

	if test_config['smooth'].values[0] == '1':
		smooth_wl = int(test_config['windowlength'].values[0])
		smooth_po = int(test_config['polyorder'].values[0])
		smooth_mode = test_config['mode'].values[0]

	if test_config['eliminate'].values[0] == '1':
		elim_thre = int(test_config['threshold'].values[0])
		elim_ws = int(test_config['windowsize'].values[0])
		elim_base = int(test_config['baseon'].values[0])

	if test_config['energy'].values[0] == '1':
		eng_bs = int(test_config['bandsize'].values[0])
		eng_odr = int(test_config['odr'].values[0])

	seg_size = int(test_config['segsize'].values[0])

	files = pd.read_csv(test_file)

	final_features = []
	labels = []

	for idx, row in files.iterrows():
		name = row['file']
		dir = row['dir']
		label = row['label']
		header = row['header']
		raw_data = pd.read_csv(dir + '\\' + '\\' + name, header=header, delimiter=';')

		ax = raw_data['ax']
		ay = raw_data['ay']
		az = raw_data['az']

		data = np.array([ax, ay, az]).T

		if test_config['filter'].values[0] == '1':
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

		if test_config['smooth'].values[0] == '1':
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

			if test_config['eliminate'].values[0] == '1':
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, elim_base)
			else:
				data = smoothed_data


		num = int(test_config['mean'].values[0]) + int(test_config['std'].values[0]) + int(test_config['min'].values[0]) + int(test_config['max'].values[0]) + int(test_config['rms'].values[0])


		for col in range(3):
			feature = np.empty([int(len(data[:, col])/seg_size), num])
			new_feature = []
			for i in range(int(len(data[:, col])/seg_size)):
				window = data[i*seg_size:(i+1)*seg_size, col]
				k = 0
				if int(test_config['mean'].values[0]):
					feature[i, k] = feature_extraction.mean(window)
					k += 1;
				if int(test_config['std'].values[0]):
					feature[i, k] = feature_extraction.std(window)
					k += 1;
				if int(test_config['min'].values[0]):
					feature[i, k] = feature_extraction.getmin(window)
					k += 1;
				if int(test_config['max'].values[0]):
					feature[i, k] = feature_extraction.getmax(window)
					k += 1;
				if int(test_config['rms'].values[0]):
					feature[i, k] = feature_extraction.rms(window)
					k += 1;
				if int(test_config['energy'].values[0]):
					energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
					temp = np.append(feature[i, :], energy_vector)
					new_feature.append(temp)
				else:
					new_feature.append(feature[i, :])

			new_feature = np.array(new_feature)
			#print("new_feature: ", new_feature.shape)
			
			if col == 0:
				ex_feature = new_feature
			else:
				ex_feature = np.append(ex_feature, new_feature, axis=1)
			
			#print("ex_feature: ", ex_feature.shape)

		num_features = ex_feature.shape[1] // 3

		num_axis = int(test_config['X'].values[0]) +int(test_config['Y'].values[0]) + int(test_config['Z'].values[0])

		trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

		k = 0

		if int(test_config['X'].values[0]) == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
			k += 1
		if int(test_config['Y'].values[0]) == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
			k += 1
		if int(test_config['Z'].values[0]) == 1:
			trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
			k += 1;

		final_features.extend(trn_feature)
		for i in range(len(trn_feature)):
			labels.append(label)

	final_features = np.array(final_features)
	labels = np.array(labels)

	test_model = pickle.load(open(offline_model, 'rb'))
	pred = test_model.predict(final_features)
	acc_tst = accuracy_score(labels, pred)

	lb_acc_tst = Label(offline, text=acc_tst, font=("Helvetica", "12", "bold italic"))
	lb_acc_tst.place(relx=0.6, rely=0.28)




global online_window
online_window = []
global online_window_size
online_window_size = 200
global flag
flag = False

def start_predict_online(data_online):
	global online_window
	global online_window_size
	global flag

	if model_text2.get("1.0", END) == []:
		messagebox.showerror("Error", "Please select test model!")
		return
	else:
		online_model = model_text2.get("1.0", END)
		online_model = online_model.split("\n")[0]

	if res_text2.get("1.0", END) == []:
		res_text2.delete("1.0", END)



	test_config = pd.read_csv(online_model.split(".")[0] + ".txt", sep=" ")
	test_config.columns = ['param', 'value']

	test_config = pd.DataFrame(data=[test_config['value'].values], columns=test_config['param'].values)

	if test_config['filter'].values[0] == '1':
		filter_order = int(test_config['order'].values[0])
		filter_fc = test_config['fc'].values[0]
		filter_type = test_config['type'].values[0]
		filter_fs = int(test_config['fs'].values[0])
		if filter_type == "bandpass" or filter_type == "bandstop":
			fc = filter_fc.split("[")[1].split("]")[0].split(",")
			filter_fc = [int(fc[0]), int(fc[1])]
		else:
			filter_fc = int(filter_fc)

	if test_config['smooth'].values[0] == '1':
		smooth_wl = int(test_config['windowlength'].values[0])
		smooth_po = int(test_config['polyorder'].values[0])
		smooth_mode = test_config['mode'].values[0]

	if test_config['eliminate'].values[0] == '1':
		elim_thre = int(test_config['threshold'].values[0])
		elim_ws = int(test_config['windowsize'].values[0])
		elim_base = int(test_config['baseon'].values[0])

	if test_config['energy'].values[0] == '1':
		eng_bs = int(test_config['bandsize'].values[0])
		eng_odr = int(test_config['odr'].values[0])

	seg_size = int(test_config['segsize'].values[0])

	#print(flag)
	if flag == False:
		online_window = []
		online_window_size = seg_size * 1.5
		flag = True


	if online_window_size == 0:

		# test_config = pd.read_csv(online_model.split(".")[0] + ".txt", sep=" ")
		# test_config.columns = ['param', 'value']

		# test_config = pd.DataFrame(data=[test_config['value'].values], columns=test_config['param'].values)

		# if test_config['filter'].values[0] == '1':
		# 	filter_order = int(test_config['order'].values[0])
		# 	filter_fc = test_config['fc'].values[0]
		# 	filter_type = test_config['type'].values[0]
		# 	filter_fs = int(test_config['fs'].values[0])
		# 	if filter_type == "bandpass" or filter_type == "bandstop":
		# 		fc = filter_fc.split("[")[1].split("]")[0].split(",")
		# 		filter_fc = [int(fc[0]), int(fc[1])]
		# 	else:
		# 		filter_fc = int(filter_fc)

		# if test_config['smooth'].values[0] == '1':
		# 	smooth_wl = int(test_config['windowlength'].values[0])
		# 	smooth_po = int(test_config['polyorder'].values[0])
		# 	smooth_mode = test_config['mode'].values[0]

		# if test_config['eliminate'].values[0] == '1':
		# 	elim_thre = int(test_config['threshold'].values[0])
		# 	elim_ws = int(test_config['windowsize'].values[0])
		# 	elim_base = int(test_config['baseon'].values[0])

		# if test_config['energy'].values[0] == '1':
		# 	eng_bs = int(test_config['bandsize'].values[0])
		# 	eng_odr = int(test_config['odr'].values[0])

		# seg_size = int(test_config['segsize'].values[0])
		
		data = np.array(online_window)
		#print(data)
		#final_features = []
		if test_config['filter'].values[0] == '1':
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

		if test_config['smooth'].values[0] == '1':
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

			if test_config['eliminate'].values[0] == '1':
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, elim_base)

		if data.shape[0] >= seg_size:
			
			num = int(test_config['mean'].values[0]) + int(test_config['std'].values[0]) + int(test_config['min'].values[0]) + int(test_config['max'].values[0]) + int(test_config['rms'].values[0])


			for col in range(3):
				feature = np.empty([int(len(data[:, col])/seg_size), num])
				new_feature = []
				for i in range(int(len(data[:, col])/seg_size)):
					window = data[i*seg_size:(i+1)*seg_size, col]
					k = 0
					if int(test_config['mean'].values[0]):
						feature[i, k] = feature_extraction.mean(window)
						k += 1;
					if int(test_config['std'].values[0]):
						feature[i, k] = feature_extraction.std(window)
						k += 1;
					if int(test_config['min'].values[0]):
						feature[i, k] = feature_extraction.getmin(window)
						k += 1;
					if int(test_config['max'].values[0]):
						feature[i, k] = feature_extraction.getmax(window)
						k += 1;
					if int(test_config['rms'].values[0]):
						feature[i, k] = feature_extraction.rms(window)
						k += 1;
					if int(test_config['energy'].values[0]):
						energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
						temp = np.append(feature[i, :], energy_vector)
						new_feature.append(temp)
					else:
						new_feature.append(feature[i, :])

				new_feature = np.array(new_feature)
				#print("new_feature: ", new_feature.shape)
				
				if col == 0:
					ex_feature = new_feature
				else:
					ex_feature = np.append(ex_feature, new_feature, axis=1)
					
					#print("ex_feature: ", ex_feature.shape)

			num_features = ex_feature.shape[1] // 3

			num_axis = int(test_config['X'].values[0]) +int(test_config['Y'].values[0]) + int(test_config['Z'].values[0])

			trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

			k = 0

			if int(test_config['X'].values[0]) == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
				k += 1
			if int(test_config['Y'].values[0]) == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
				k += 1
			if int(test_config['Z'].values[0]) == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
				k += 1;

			test_model = pickle.load(open(online_model, 'rb'))
			predict_output = test_model.predict(trn_feature)
			print(predict_output[0])
			#res_text2.insert(END, predict_output[0])

		#online_window = []
		#online_window_size = 200
		flag = False
	else:
		online_window.append(data_online)
		online_window_size -= 1
		# print(online_window_size)

	
def callback(data):
    ch, ax, ay, az, mx, my, mz, temp = data
    input_data = np.array([ax,ay,az])
    start_predict_online(input_data)

def test_online():
    import imports  # pylint: disable=unused-import
    from kx_lib.kx_board import ConnectionManager
    from kx_lib.kx_util import evkit_config
    from kmx62 import kmx62_data_logger
    from kmx62.kmx62_driver import KMX62Driver
    ODR = 50

    connection_manager = ConnectionManager(board_config_json='rokix_board_rokix_sensor_node_i2c.json')
    sensor_kmx62 = KMX62Driver()
    try:
        connection_manager.add_sensor(sensor_kmx62)
        kmx62_data_logger.enable_data_logging(sensor_kmx62, odr=ODR)
        stream = kmx62_data_logger.KMX62DataStream([sensor_kmx62])
        stream.read_data_stream(console=False, callback=callback)

    finally:
        sensor_kmx62.set_power_off()
        connection_manager.disconnect()

# def run_thread():
# 	t1 = threading.Thread(target=test_online)
# 	t1.start()
# 	t1.join()




### offline front end ###

file_text = Text(offline, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
file_text.place(relx=0.24, rely=0.1)

model_text = Text(offline, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
model_text.place(relx=0.24, rely=0.51)

def selectFile():
	filename = filedialog.askopenfilename(initialdir="/Documents", title="Select testing file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
	if filename == "":
		return

	if file_text.get("1.0", END) != []:
		file_text.delete("1.0", END)
	#file_text.insert(END, filename.split('/')[len(filename.split('/'))-1])
	file_text.insert(END, filename)
	#print("test_file: ", filename)

def selectModel1():
	filename = filedialog.askopenfilename(initialdir="/Documents", title="Select model", filetypes=(("sav files", "*.sav"), ("all files", "*.*")))
	if filename == "":
		return

	if model_text.get("1.0", END) != []:
		model_text.delete("1.0", END)
	#model_text.insert(END, filename.split('/')[len(filename.split('/'))-1])
	model_text.insert(END, filename)
	#print("offline_model: ", filename)

bt_sel = Button(offline, text='Select Files', font=("Helvetica", "12"), command=selectFile)
bt_sel.place(relx=0.05, rely=0.05)

bt_sel2 = Button(offline, text='Select Model', font=("Helvetica", "12"), command=selectModel1)
bt_sel2.place(relx=0.05, rely=0.45)

bt_start = Button(offline, text='Start', font=("Helvetica", "12", "bold"), command=start_predict_offline)
bt_start.place(relx=0.9, rely=0.6)

lb_acc = Label(offline, text='Accurancy: ', font=("Helvetica", "12", "bold italic"))
lb_acc.place(relx=0.55, rely=0.08)



bt_start2 = Button(online, text='Start', font=("Helvetica", "12", "bold"), command=test_online)
bt_start2.place(relx=0.9, rely=0.6)
#exec(open("gogogopredict.py").read())

model_text2 = Text(online, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
model_text2.place(relx=0.05, rely=0.4)

def selectModel2():
	filename = filedialog.askopenfilename(initialdir="/Documents", title="Select model", filetypes=(("sav files", "*.sav"), ("all files", "*.*")))
	if filename == "":
		return

	if model_text2.get("1.0", END) != []:
		model_text2.delete("1.0", END)
	model_text2.insert(END, filename)

bt_sel3 = Button(online, text='Select Model', font=("Helvetica", "12"), command=selectModel2)
bt_sel3.place(relx=0.05, rely=0.05)

result2 = LabelFrame(online, text='Result', font=("Helvetica", "11", "bold italic"))
result2.place(relx=0.35, rely=0, relheight=0.85, relwidth=0.5)
res_text2 = Text(result2, font=("Helvetica", "11"), width=32, wrap=WORD, bd=0, height=3)
vscroll2 = ttk.Scrollbar(result2, orient=VERTICAL, command=res_text2.yview)
res_text2['yscroll'] = vscroll2.set
vscroll2.pack(side=RIGHT, fill=Y)
res_text2.place(relx=0.02, rely=0.01)

root.mainloop()