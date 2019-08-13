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
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from multiprocessing import Process, Queue
from queue import Empty

HEIGHT = 600
WIDTH = 1440


class MLAPP():
	def __init__(self, master):
		self.master = master
		master.title("GUI")
		master.minsize(WIDTH, HEIGHT)
		self.initUI()


	def initUI(self):
		self.canvas = Canvas(self.master, height=HEIGHT, width=WIDTH)
		self.canvas.pack()

		self.frame_pre = Frame(self.master, bg='#CFD8DC')
		self.frame_pre.place(relwidth=0.3, relheight=1.0)

		self.frame_axis = Frame(self.master, bg='#E0F2F1')
		self.frame_axis.place(relx=0.3, relwidth=0.3, relheight=0.3)

		self.frame_fe = Frame(self.master, bg='#E0F2F1')
		self.frame_fe.place(relx=0.3, rely=0.3, relwidth=0.3, relheight=0.7)

		self.frame_trn = Frame(self.master, bg='#B2DFDB')
		self.frame_trn.place(relx=0.6, relwidth=0.4, relheight=0.5)

		self.frame_tst = Frame(self.master, bg='#80CBC4')
		self.frame_tst.place(relx=0.6, rely=0.5, relwidth=0.4, relheight=0.5)

		##############################
		#### title of each frame ####
		##############################
		self.pre = LabelFrame(self.frame_pre, text='Preprocessing', font=("Helvetica", "25", "bold italic"))
		self.pre.pack(fill=BOTH, expand=True)

		self.axis = LabelFrame(self.frame_axis, text='Axis Selection', font=("Helvetica", "25", "bold italic"))
		self.axis.pack(fill=BOTH, expand= True)

		self.fe = LabelFrame(self.frame_fe, text='Feature Extraction', font=("Helvetica", "25", "bold italic"))
		self.fe.pack(fill=BOTH, expand=True)

		self.trn = LabelFrame(self.frame_trn, text='Training', font=("Helvetica", "25", "bold italic"))
		self.trn.pack(fill=BOTH, expand=True)

		self.tst = LabelFrame(self.frame_tst, text='Prediction', font=("Helvetica", "25", "bold italic"))
		self.tst.pack(fill=BOTH, expand=True)

		############################
		#### preprocessing ####
		############################
		# filter
		self.var_filter = IntVar()

		self.filterChk = Checkbutton(self.pre, text='Filter', font=("Helvetica", "18"), variable=self.var_filter, command=self.activateCheck_filter)
		self.filterChk.place(relx=0.05, rely=0.05)

		self.lb_order = Label(self.pre, text='Order', font=("Helvetica", "12"))
		self.lb_order.place(relx=0.1, rely=0.13)
		self.en_order = Entry(self.pre, bd=2, state=DISABLED)
		self.en_order.place(relx=0.22, rely=0.13)

		self.lb_fc = Label(self.pre, text='fc', font=("Helvetica", "12", "italic"))
		self.lb_fc.place(relx=0.1, rely=0.18)
		self.en_fc = Entry(self.pre, bd=2, state=DISABLED)
		self.en_fc.place(relx=0.22, rely=0.18)

		self.lb_type = Label(self.pre, text='Type', font=("Helvetica", "12"))
		self.lb_type.place(relx=0.1, rely=0.23)
		self.en_type = Combobox(self.pre, values=["lowpass", "highpass", "bandpass", "bandstop"], state=DISABLED)
		self.en_type.place(relx=0.22, rely=0.23)

		self.lb_fs = Label(self.pre, text='fs', font=("Helvetica", "12", "italic"))
		self.lb_fs.place(relx=0.1, rely=0.28)
		self.en_fs = Entry(self.pre, bd=2, state=DISABLED)
		self.en_fs.place(relx=0.22, rely=0.28)


		# smooth
		self.var_smooth = IntVar()

		self.smoothChk = Checkbutton(self.pre, text='Smoothing', font=("Helvetica", "18"), variable=self.var_smooth, command=self.activateCheck_smooth)
		self.smoothChk.place(relx=0.05, rely=0.38)

		self.lb_wl = Label(self.pre, text='Window Length', font=("Helvetica", "12"))
		self.lb_wl.place(relx=0.1, rely=0.46)
		self.en_wl = Entry(self.pre, bd=2, state=DISABLED)
		self.en_wl.place(relx=0.38, rely=0.46)

		self.lb_po = Label(self.pre, text='Poly Order', font=("Helvetica", "12"))
		self.lb_po.place(relx=0.1, rely=0.51)
		self.en_po = Entry(self.pre, bd=2, state=DISABLED)
		self.en_po.place(relx=0.38, rely=0.51)

		self.lb_mode = Label(self.pre, text='Mode', font=("Helvetica", "12"))
		self.lb_mode.place(relx=0.1, rely=0.56)
		self.en_mode = Combobox(self.pre, values=["mirror", "nearest", "wrap", "constant", "interp"], state=DISABLED)
		self.en_mode.place(relx=0.38, rely=0.56)


		# eliminate
		self.var_elim = IntVar()

		self.elimChk = Checkbutton(self.pre, text='Eliminate Abnormal Data', font=("Helvetica", "18"), variable=self.var_elim, command=self.activateCheck_elim, state=DISABLED)
		self.elimChk.place(relx=0.05, rely=0.66)

		self.lb_thre = Label(self.pre, text='Threshold', font=("Helvetica", "12"))
		self.lb_thre.place(relx=0.1, rely=0.74)
		self.en_thre = Entry(self.pre, bd=2, state=DISABLED)
		self.en_thre.place(relx=0.35, rely=0.74)

		self.lb_win = Label(self.pre, text='Window Size', font=("Helvetica", "12"))
		self.lb_win.place(relx=0.1, rely=0.79)
		self.en_win = Entry(self.pre, bd=2, state=DISABLED)
		self.en_win.place(relx=0.35, rely=0.79)

		self.var_base = IntVar()
		self.var_base.set(1)
		self.lb_base = Label(self.pre, text='Base on', font=("Helvetica", "12"))
		self.lb_base.place(relx=0.1, rely=0.84)
		self.rb_x = Radiobutton(self.pre, text='X Axis', variable=self.var_base, value=1, state=DISABLED)
		self.rb_y = Radiobutton(self.pre, text='Y Axis', variable=self.var_base, value=2, state=DISABLED)
		self.rb_z = Radiobutton(self.pre, text='Z Axis', variable=self.var_base, value=3, state=DISABLED)
		self.rb_x.place(relx=0.35, rely=0.84)
		self.rb_y.place(relx=0.5, rely=0.84)
		self.rb_z.place(relx=0.65, rely=0.84)

		# plot figures
		self.plot = Button(self.pre, text='Plot', font=("Helvetica", "12", "bold"), padx=4, command=self.plotFigures)
		self.plot.place(relx=0.85, rely=0.92)

		########################
		#### AXIS SELECTION ####
		########################
		self.var_accel_x = IntVar()
		self.accelXChk = Checkbutton(self.axis, text='X Acceleration', font=("Helvetica", "15"), variable= self.var_accel_x)
		self.accelXChk.place(relx=0.05, rely=0.06)

		self.var_accel_y = IntVar()
		self.accelYChk = Checkbutton(self.axis, text= 'Y Acceleration', font=("Helvetica", "15"), variable= self.var_accel_y)
		self.accelYChk.place(relx= 0.05, rely=0.28)

		self.var_accel_z = IntVar()
		self.accelZChk = Checkbutton(self.axis, text= 'Z Acceleration', font=("Helvetica", "15"), variable= self.var_accel_z)
		self.accelZChk.place(relx= 0.05, rely=0.5)

		################################
		#### feature extract ####
		################################
		self.var_mean = IntVar()
		self.meanChk = Checkbutton(self.fe, text='Mean Value', font=("Helvetica", "15"), variable=self.var_mean)
		self.meanChk.place(relx=0.05, rely=0.05)

		self.var_std = IntVar()
		self.stdChk = Checkbutton(self.fe, text='Standard Divation', font=("Helvetica", "15"), variable=self.var_std)
		self.stdChk.place(relx=0.05, rely=0.13)

		self.var_min = IntVar()
		self.minChk = Checkbutton(self.fe, text='Minimum Value', font=("Helvetica", "15"), variable=self.var_min)
		self.minChk.place(relx=0.05, rely=0.21)

		self.var_max = IntVar()
		self.maxChk = Checkbutton(self.fe, text='Maximum Value', font=("Helvetica", "15"), variable=self.var_max)
		self.maxChk.place(relx=0.05, rely=0.29)

		self.var_rms = IntVar()
		self.rmsChk = Checkbutton(self.fe, text='Root Mean Square', font=("Helvetica", "15"), variable=self.var_rms)
		self.rmsChk.place(relx=0.05, rely=0.37)

		self.var_energy = IntVar()

		self.energyChk = Checkbutton(self.fe, text='Energy', font=("Helvetica", "15"), variable=self.var_energy, command=self.activateCheck_energy)
		self.energyChk.place(relx=0.05, rely=0.45)

		self.lb_bs = Label(self.fe, text='Band Size', font=("Helvetica", "11"))
		self.lb_bs.place(relx=0.1, rely=0.53)
		self.en_bs = Entry(self.fe, bd=2, state=DISABLED)
		self.en_bs.place(relx=0.28, rely=0.53)

		self.lb_odr = Label(self.fe, text='ODR', font=("Helvetica", "11"))
		self.lb_odr.place(relx=0.1, rely=0.59)
		self.en_odr = Entry(self.fe, bd=2, state=DISABLED)
		self.en_odr.place(relx=0.28, rely=0.59)

		# window size
		self.lb_ws = Label(self.fe, text='Segment Size', font=("Helvetica", "15"))
		self.lb_ws.place(relx=0.05, rely=0.7)
		self.en_ws = Entry(self.fe, bd=2)
		self.en_ws.place(relx=0.35, rely=0.71)

		# extract button
		self.bt_ext = Button(self.fe, text='Extract', bd=2, font=("Helvetica", "12", "bold"), command=self.extractAndSave)
		self.bt_ext.place(relx=0.8, rely=0.88)

		########################
		#### Training ####
		########################
		# model
		self.lb_model = Label(self.trn, text='Model', font=("Helvetica", "12"))
		self.lb_model.place(relx=0.05, rely=0.05)
		self.en_model = Combobox(self.trn, values=["Random Forest", "SVM"])
		self.en_model.place(relx=0.15, rely=0.05)

		# file box
		self.scrollbar_x = Scrollbar(self.trn, orient=HORIZONTAL)
		self.scrollbar_x.pack(side=BOTTOM, fill=X)
		self.scrollbar_y = Scrollbar(self.trn, orient=VERTICAL)
		self.scrollbar_y.pack(side=RIGHT, fill=Y)

		self.trn_files = Listbox(self.trn, height=3, selectmode=SINGLE, width=32)
		self.trn_files.place(relx=0.05, rely=0.30)

		self.scrollbar_x.config(command=self.trn_files.xview)
		self.scrollbar_y.config(command=self.trn_files.yview)
		self.trn_files.config(xscrollcommand=self.scrollbar_x.set, yscrollcommand=self.scrollbar_y.set)

		# add file button
		self.bt_add = Button(self.trn, text='Add Files', font=("Helvetica", "12"), command=self.addFile)
		self.bt_add.place(relx=0.05, rely=0.18)
		# delete file button
		self.bt_delete = Button(self.trn, text='Delete Files', font=("Helvetica", "12"), command=self.deleteFile)
		self.bt_delete.place(relx=0.22, rely=0.18)

		# train set ratio
		self.var_ratio = DoubleVar()
		self.var_ratio.set(0.8)

		self.scale_title = Label(self.trn, text='Ratio of training set: 0.8', font=("Helvetica", "12"))
		self.scale_title.place(relx=0.05, rely=0.52)

		self.scale = Scale(self.trn, from_=0.1, to=0.9, orient=HORIZONTAL, length=180, resolution=0.1, variable=self.var_ratio, showvalue=0, command=self.get_ratio)
		self.scale.place(relx=0.05, rely=0.6)
		
		# start train button
		self.bt_trn = Button(self.trn, text='START', font=("Helvetica", "15", "bold"), height=1, width=15, command=self.startTrain)
		self.bt_trn.place(relx=0.05, rely=0.75)

		# progress pane
		self.lb_pb = Label(self.trn, text='Progress', font=("Helvetica", "12"))
		self.lb_pb.place(relx=0.45, rely=0.05)
		self.pb = ttk.Progressbar(self.trn, mode='determinate', orient=HORIZONTAL, length=200)
		self.pb.place(relx=0.58, rely=0.05)

		# output
		self.result = LabelFrame(self.trn, text='Result', font=("Helvetica", "15", "bold italic"))
		self.result.place(relx=0.45, rely=0.15, relheight=0.75, relwidth=0.5)
		self.res_text = Text(self.result, font=("Helvetica", "11"), width=32, wrap=WORD, bd=0, height=9)
		self.vscroll = ttk.Scrollbar(self.result, orient=VERTICAL, command=self.res_text.yview)
		self.res_text['yscroll'] = self.vscroll.set
		self.vscroll.pack(side=RIGHT, fill=Y)
		self.res_text.place(relx=0.02, rely=0.01)


		########################
		#### Prediction ####
		########################
		self.offline = LabelFrame(self.tst, text='Offline', font=("Helvetica", "15", "bold italic"))
		self.offline.place(relx=0, rely=0, relheight=0.5, relwidth=1.0)

		self.online = LabelFrame(self.tst, text='Online', font=("Helvetica", "15", "bold italic"))
		self.online.place(relx=0, rely=0.5, relheight=0.5, relwidth=1.0)

		### offline front end ###
		self.file_text = Text(self.offline, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
		self.file_text.place(relx=0.24, rely=0.1)

		self.model_text = Text(self.offline, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
		self.model_text.place(relx=0.24, rely=0.51)

		self.bt_sel = Button(self.offline, text='Select Files', font=("Helvetica", "12"), command=self.selectFile)
		self.bt_sel.place(relx=0.05, rely=0.05)

		self.bt_sel2 = Button(self.offline, text='Select Model', font=("Helvetica", "12"), command=lambda: self.selectModel(1))
		self.bt_sel2.place(relx=0.05, rely=0.45)

		self.bt_start = Button(self.offline, text='Start', font=("Helvetica", "12", "bold"), command=self.start_predict_offline)
		self.bt_start.place(relx=0.9, rely=0.6)

		self.lb_acc = Label(self.offline, text='Accuracy: ', font=("Helvetica", "12", "bold italic"))
		self.lb_acc.place(relx=0.55, rely=0.08)

		# online
		self.bt_start2 = Button(self.online, text='Start', font=("Helvetica", "12", "bold"), command=self.start_predict_online)
		self.bt_start2.place(relx=0.9, rely=0.6)

		self.model_text2 = Text(self.online, font=("Helvetica", "11"), width=20, wrap=WORD, bd=0, height=1)
		self.model_text2.place(relx=0.05, rely=0.4)

		self.bt_sel3 = Button(self.online, text='Select Model', font=("Helvetica", "12"), command=lambda: self.selectModel(2))
		self.bt_sel3.place(relx=0.05, rely=0.05)

		self.result2 = LabelFrame(self.online, text='Result', font=("Helvetica", "11", "bold italic"))
		self.result2.place(relx=0.35, rely=0, relheight=0.85, relwidth=0.5)
		self.res_text2 = Text(self.result2, font=("Helvetica", "11"), width=32, wrap=WORD, bd=0, height=3)
		self.vscroll2 = ttk.Scrollbar(self.result2, orient=VERTICAL, command=self.res_text2.yview)
		self.res_text2['yscroll'] = self.vscroll2.set
		self.vscroll2.pack(side=RIGHT, fill=Y)
		self.res_text2.place(relx=0.02, rely=0.01)

		self.online_window_size = 0
		self.online_window = []
		self.flag = False


	def activateCheck_filter(self):
		if self.var_filter.get() == 1:
			self.en_order.config(state=NORMAL)
			self.en_fc.config(state=NORMAL)
			self.en_type.config(state=NORMAL)
			self.en_fs.config(state=NORMAL)
		elif self.var_filter.get() == 0:
			self.en_order.config(state=DISABLED)
			self.en_fc.config(state=DISABLED)
			self.en_type.config(state=DISABLED)
			self.en_fs.config(state=DISABLED)

	def activateCheck_smooth(self):
		if self.var_smooth.get() == 1:
			self.en_wl.config(state=NORMAL)
			self.en_po.config(state=NORMAL)
			self.en_mode.config(state=NORMAL)
			self.elimChk.config(state=NORMAL)
		elif self.var_smooth.get() == 0:
			self.en_wl.config(state=DISABLED)
			self.en_po.config(state=DISABLED)
			self.en_mode.config(state=DISABLED)
			self.elimChk.config(state=DISABLED)

	def activateCheck_elim(self):
		if self.var_elim.get() == 1:
			self.en_thre.config(state=NORMAL)
			self.en_win.config(state=NORMAL)
			self.rb_x.config(state=NORMAL)
			self.rb_y.config(state=NORMAL)
			self.rb_z.config(state=NORMAL)
		elif self.var_elim.get() == 0:
			self.en_thre.config(state=DISABLED)
			self.en_win.config(state=DISABLED)
			self.rb_x.config(state=DISABLED)
			self.rb_y.config(state=DISABLED)
			self.rb_z.config(state=DISABLED)

	def activateCheck_energy(self):
		if self.var_energy.get() == 1:
			self.en_bs.config(state=NORMAL)
			self.en_odr.config(state=NORMAL)
		elif self.var_energy.get() == 0:
			self.en_bs.config(state=DISABLED)
			self.en_odr.config(state=DISABLED)


	def addFile(self):
		filename = filedialog.askopenfilename(initialdir="/", title="Select training file", filetypes=(("csv files", "*.csv"), ("numpy files", "*.npy")))
		if filename == "":
			return
		self.trn_files.insert(END, filename)

	def deleteFile(self):
		idx = self.trn_files.curselection()
		i = len(idx) - 1
		while(i>=0):
			self.trn_files.delete(idx[i])
			i -= 1

	def get_ratio(self, arg):
		text = 'Ratio of training set: ' + str(self.var_ratio.get())
		self.scale_title.config(text=text)

	def selectFile(self):
		filename = filedialog.askopenfilename(initialdir="/Documents", title="Select testing file", filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
		if filename == "":
			return

		if self.file_text.get("1.0", END) != []:
			self.file_text.delete("1.0", END)
		self.file_text.insert(END, filename)


	def selectModel(self, arg):
		filename = filedialog.askopenfilename(initialdir="/Documents", title="Select model", filetypes=(("sav files", "*.sav"), ("all files", "*.*")))
		if filename == "":
			return

		if arg == 1:
			if self.model_text.get("1.0", END) != []:
				self.model_text.delete("1.0", END)
			self.model_text.insert(END, filename)
		if arg == 2:
			if self.model_text2.get("1.0", END) != []:
				self.model_text2.delete("1.0", END)
			self.model_text2.insert(END, filename)

	def plotting(self):
		file_idx = self.trn_files.curselection()
		filename = ""
		if file_idx == ():
			messagebox.showerror("Error", "Please select the training files!")
			return
		else:
			filename = self.trn_files.get(file_idx)


		if self.var_filter.get() == 1:
			if self.en_order.get() != "" and self.en_fc.get() != "" and self.en_type.get() != "" and self.en_fs.get() != "":
				if self.en_type.get() == "bandpass" or self.en_type.get() == "bandstop":
					if self.en_fc.get().split("[")[0] != "":
						messagebox.showerror("Error", "The form of fc should be like \'[fc1, fc2]\'!")
						return
			else:
				messagebox.showerror("Error", "Please set all the parameters of Filter!")
				return

		if self.var_smooth.get() == 1:
			if self.en_wl.get() != "" and self.en_po.get() != "" and self.en_mode.get() != "":
				if int(self.en_wl.get()) % 2 == 0:
					messagebox.showerror("Error", "The window length should be odd!")
					return
			else:
				messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
				return

		if self.var_elim.get() == 1:
			if self.en_thre.get() == "" or self.en_win.get() == "":
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

			top = Toplevel(self.master)
			top.title("Figure " + str(idx+1) + " / " + str(num_rows))

			f = Figure(figsize=(14, 8), dpi=100)

			canvas_fig = FigureCanvasTkAgg(f, top)
			canvas_fig.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

			toolbar = NavigationToolbar2Tk(canvas_fig, top)
			canvas_fig._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
			threading.Thread(target=self.plotter(data, f, canvas_fig, toolbar, label)).start()

	def plotter(self, data, f, canvas_fig, toolbar, label):
		sub = self.var_filter.get() + self.var_smooth.get() + self.var_elim.get() + 1
		if self.var_elim.get() == 1:
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

		if self.var_filter.get() == 1:
			filter_order = int(self.en_order.get())
			filter_fc = self.en_fc.get()
			filter_type = self.en_type.get()
			filter_fs = int(self.en_fs.get())
			if filter_type == "bandpass" or filter_type == "bandstop":
				fc = filter_fc.split("[")[1].split("]")[0].split(",")
				filter_fc = [int(fc[0]), int(fc[1])]
			else:
				filter_fc = int(filter_fc)
			data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)
			axes[index].plot(data[:, 0], label='filtered ax - ' + label, alpha=0.8)
			axes[index].plot(data[:, 1], label='filtered ay - ' + label, alpha=0.8)
			axes[index].plot(data[:, 2], label='filtered az - ' + label, alpha=0.8)
			axes[index].legend()
			axes[index].grid()
			index += 1

		if self.var_smooth.get() == 1:
			smooth_wl = int(self.en_wl.get())
			smooth_po = int(self.en_po.get())
			smooth_mode = self.en_mode.get()
			smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)
			axes[index].plot(smoothed_data[:, 0], label='smoothed ax - ' + label, alpha=0.8)
			axes[index].plot(smoothed_data[:, 1], label='smoothed ay - ' + label, alpha=0.8)
			axes[index].plot(smoothed_data[:, 2], label='smoothed az - ' + label, alpha=0.8)
			axes[index].legend()
			axes[index].grid()
			axes[index].set_ylim((-6000, 6000))
			index += 1

			if self.var_elim.get() == 1:
				elim_thre = int(self.en_thre.get())
				elim_ws = int(self.en_win.get())
				new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, self.var_base.get())
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

		canvas_fig.draw()
		toolbar.update()

	def plotFigures(self):
		file_idx = self.trn_files.curselection()
		filename = ""
		if file_idx == ():
			messagebox.showerror("Error", "Please select the training files!")
			return
		else:
			filename = self.trn_files.get(file_idx)


		if self.var_filter.get() == 1:
			if self.en_order.get() != "" and self.en_fc.get() != "" and self.en_type.get() != "" and self.en_fs.get() != "":
				filter_order = int(self.en_order.get())
				filter_fc = self.en_fc.get()
				filter_type = self.en_type.get()
				filter_fs = int(self.en_fs.get())
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

		if self.var_smooth.get() == 1:
			if self.en_wl.get() != "" and self.en_po.get() != "" and self.en_mode.get() != "":
				smooth_wl = int(self.en_wl.get())
				smooth_po = int(self.en_po.get())
				smooth_mode = self.en_mode.get()

				if smooth_wl % 2 == 0:
					messagebox.showerror("Error", "The window length should be odd!")
					return


			else:
				messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
				return

		if self.var_elim.get() == 1:
			if self.en_thre.get() != "" and self.en_win.get() != "":
				elim_thre = int(self.en_thre.get())
				elim_ws = int(self.en_win.get())
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

			top = Toplevel(self.master)
			top.title("Figure " + str(idx+1) + " / " + str(num_rows))

			f = Figure(figsize=(14, 8), dpi=100)

			sub = self.var_filter.get() + self.var_smooth.get() + self.var_elim.get() + 1
			if self.var_elim.get() == 1:
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

			if self.var_filter.get() == 1:
				data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)
				axes[index].plot(data[:, 0], label='filtered ax - ' + label, alpha=0.8)
				axes[index].plot(data[:, 1], label='filtered ay - ' + label, alpha=0.8)
				axes[index].plot(data[:, 2], label='filtered az - ' + label, alpha=0.8)
				axes[index].legend()
				axes[index].grid()
				index += 1

			if self.var_smooth.get() == 1:
				smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)
				axes[index].plot(smoothed_data[:, 0], label='smoothed ax - ' + label, alpha=0.8)
				axes[index].plot(smoothed_data[:, 1], label='smoothed ay - ' + label, alpha=0.8)
				axes[index].plot(smoothed_data[:, 2], label='smoothed az - ' + label, alpha=0.8)
				axes[index].legend()
				axes[index].grid()
				axes[index].set_ylim((-6000, 6000))
				index += 1

				if self.var_elim.get() == 1:
					new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, self.var_base.get())
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

	def extractAndSave(self):
		file_idx = self.trn_files.curselection()
		filename = ""
		if file_idx == ():
			messagebox.showerror("Error", "Please select the training files!")
			return
		else:
			filename = self.trn_files.get(file_idx)


		if self.var_filter.get() == 1:
			if self.en_order.get() != "" and self.en_fc.get() != "" and self.en_type.get() != "" and self.en_fs.get() != "":
				filter_order = int(self.en_order.get())
				filter_fc = self.en_fc.get()
				filter_type = self.en_type.get()
				filter_fs = int(self.en_fs.get())
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

		if self.var_smooth.get() == 1:
			if self.en_wl.get() != "" and self.en_po.get() != "" and self.en_mode.get() != "":
				smooth_wl = int(self.en_wl.get())
				smooth_po = int(self.en_po.get())
				smooth_mode = self.en_mode.get()

				if smooth_wl % 2 == 0:
					messagebox.showerror("Error", "The window length should be odd!")
					return


			else:
				messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
				return

		if self.var_elim.get() == 1:
			if self.en_thre.get() != "" and self.en_win.get() != "":
				elim_thre = int(self.en_thre.get())
				elim_ws = int(self.en_win.get())
			else:
				messagebox.showerror("Error", "Please set all the parameters of Eliminate Abnormal Data!")
				return

		if self.var_energy.get() == 1:
			if self.en_bs.get() != "" and self.en_odr.get() != "":
				eng_bs = int(self.en_bs.get())
				eng_odr = int(self.en_odr.get())
			else:
				messagebox.showerror("Error", "Please set all the parameters of Energy!")
				return

		if self.en_ws.get() != "":
			seg_size = int(self.en_ws.get())
		else:
			messagebox.showerror("Error", "Please set the segment size!")
			return

		if self.var_accel_x.get() == 0 and self.var_accel_y.get() == 0 and self.var_accel_z.get() == 0:
			messagebox.showerror("Error", "Please select at least one axis!")
			return

		if (self.var_base.get() == 1 and self.var_accel_x.get() == 0) or (self.var_base.get() == 2 and self.var_accel_y.get() == 0) or (self.var_base.get() == 3 and self.var_accel_z.get() == 0):
			messagebox.showerror("Error", "Please select the axis that is select in Axis Selection!")
			return

		if self.var_mean.get() + self.var_std.get() + self.var_min.get() + self.var_max.get() + self.var_rms.get() + self.var_energy.get() == 0:
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

			if self.var_filter.get() == 1:
				data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

			if self.var_smooth.get() == 1:
				smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

				if self.var_elim.get() == 1:
					new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, self.var_base.get())
				else:
					data = smoothed_data


			num = self.var_mean.get() + self.var_std.get() + self.var_min.get() + self.var_max.get() + self.var_rms.get()


			for col in range(3):
				feature = np.empty([int(len(data[:, col])/seg_size), num])
				new_feature = []
				for i in range(int(len(data[:, col])/seg_size)):
					window = data[i*seg_size:(i+1)*seg_size, col]
					k = 0
					if self.var_mean.get():
						feature[i, k] = feature_extraction.mean(window)
						k += 1;
					if self.var_std.get():
						feature[i, k] = feature_extraction.std(window)
						k += 1;
					if self.var_min.get():
						feature[i, k] = feature_extraction.getmin(window)
						k += 1;
					if self.var_max.get():
						feature[i, k] = feature_extraction.getmax(window)
						k += 1;
					if self.var_rms.get():
						feature[i, k] = feature_extraction.rms(window)
						k += 1;
					if self.var_energy.get():
						energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
						temp = np.append(feature[i, :], energy_vector)
						new_feature.append(temp)
					else:
						new_feature.append(feature[i, :])

				new_feature = np.array(new_feature)

				
				if col == 0:
					ex_feature = new_feature
				else:
					ex_feature = np.append(ex_feature, new_feature, axis=1)


			num_features = ex_feature.shape[1] // 3

			num_axis = self.var_accel_x.get() + self.var_accel_y.get() + self.var_accel_z.get()

			trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

			k = 0

			if self.var_accel_x.get() == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
				k += 1
			if self.var_accel_y.get() == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
				k += 1
			if self.var_accel_z.get() == 1:
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

	def startTrain(self):
		model =  self.en_model.get()
		if model == "":
			messagebox.showerror("Error", "Please select the training model!")
			return

		trn_ratio = self.var_ratio.get()

		file_idx = self.trn_files.curselection()
		filename = ""
		if file_idx == ():
			messagebox.showerror("Error", "Please select the training files!")
			return
		else:
			filename = self.trn_files.get(file_idx)


		if self.var_filter.get() == 1:
			if self.en_order.get() != "" and self.en_fc.get() != "" and self.en_type.get() != "" and self.en_fs.get() != "":
				filter_order = int(self.en_order.get())
				filter_fc = self.en_fc.get()
				filter_type = self.en_type.get()
				filter_fs = int(self.en_fs.get())
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

		if self.var_smooth.get() == 1:
			if self.en_wl.get() != "" and self.en_po.get() != "" and self.en_mode.get() != "":
				smooth_wl = int(self.en_wl.get())
				smooth_po = int(self.en_po.get())
				smooth_mode = self.en_mode.get()

				if smooth_wl % 2 == 0:
					messagebox.showerror("Error", "The window length should be odd!")
					return


			else:
				messagebox.showerror("Error", "Please set all the parameters of Smoothing!")
				return

		if self.var_elim.get() == 1:
			if self.en_thre.get() != "" and self.en_win.get() != "":
				elim_thre = int(self.en_thre.get())
				elim_ws = int(self.en_win.get())
			else:
				messagebox.showerror("Error", "Please set all the parameters of Eliminate Abnormal Data!")
				return

		if self.var_energy.get() == 1:
			if self.en_bs.get() != "" and self.en_odr.get() != "":
				eng_bs = int(self.en_bs.get())
				eng_odr = int(self.en_odr.get())
			else:
				messagebox.showerror("Error", "Please set all the parameters of Energy!")
				return

		if self.en_ws.get() != "":
			seg_size = int(self.en_ws.get())
		else:
			messagebox.showerror("Error", "Please set the segment size!")
			return

		if self.var_accel_x.get() == 0 and self.var_accel_y.get() == 0 and self.var_accel_z.get() == 0:
			messagebox.showerror("Error", "Please select at least one axis!")
			return

		if (self.var_base.get() == 1 and self.var_accel_x.get() == 0) or (self.var_base.get() == 2 and self.var_accel_y.get() == 0) or (self.var_base.get() == 3 and self.var_accel_z.get() == 0):
			messagebox.showerror("Error", "Please select the axis that is select in Axis Selection!")
			return

		if self.var_mean.get() + self.var_std.get() + self.var_min.get() + self.var_max.get() + self.var_rms.get() + self.var_energy.get() == 0:
			messagebox.showerror("Error", "Please select at least one feature!")
			return


		save_model = filedialog.asksaveasfilename(initialdir = "/", title = "Save Model", filetypes = (("sav files","*.sav"), ("all files", "*.*")))
		if save_model == "":
			return
		if len(save_model.split(".")) == 1:
			save_model = save_model + '.sav'


		self.pb['value'] = 10
		self.trn.update_idletasks() 
		time.sleep(0.5)

		### write model configuration
		parts = save_model.split('/')
		directory = '/'.join(parts[0:len(parts)-1])
		txtName = parts[len(parts)-1].split('.')[0] + '.txt'

		f = open(directory + '/' + txtName, "w+")
		f.write("param value\n")
		if self.var_filter.get() == 1:
			f.write("filter 1\n")
			f.write("order %d\n" %filter_order)
			f.write("fc %d\n" %filter_fc)
			f.write("type " + filter_type + "\n")
			f.write("fs %d\n" %filter_fs)
		else:
			f.write("filter 0\n")

		if self.var_smooth.get() == 1:
			f.write("smooth 1\n")
			f.write("windowlength %d\n" %smooth_wl)
			f.write("polyorder %d\n" %smooth_po)
			f.write("mode " + smooth_mode + "\n")
		else:
			f.write("smooth 0\n")

		if self.var_elim.get() == 1:
			f.write("eliminate 1\n")
			f.write("threshold %d\n" %elim_thre)
			f.write("windowsize %d\n" %elim_ws)
			f.write("baseon %d\n" %self.var_base.get())
		else:
			f.write("eliminate 0\n")

		f.write("X %d\n" %self.var_accel_x.get())
		f.write("Y %d\n" %self.var_accel_y.get())
		f.write("Z %d\n" %self.var_accel_z.get())

		f.write("mean %d\n" %self.var_mean.get())
		f.write("std %d\n" %self.var_std.get())
		f.write("min %d\n" %self.var_min.get())
		f.write("max %d\n" %self.var_max.get())
		f.write("rms %d\n" %self.var_rms.get())
		f.write("energy %d\n" %self.var_energy.get())

		if self.var_energy.get() == 1:
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

			if self.var_filter.get() == 1:
				data = data_preprocessing.filter(data, filter_order, filter_fc, filter_type, filter_fs)

			if self.var_smooth.get() == 1:
				smoothed_data = data_preprocessing.smooth(data, smooth_wl, smooth_po, smooth_mode)

				if self.var_elim.get() == 1:
					new_smoothed_data, data = data_preprocessing.eliminate_abnormal_value(smoothed_data, data, elim_ws, elim_thre, self.var_base.get())
				else:
					data = smoothed_data


			num = self.var_mean.get() + self.var_std.get() + self.var_min.get() + self.var_max.get() + self.var_rms.get()


			for col in range(3):
				feature = np.empty([int(len(data[:, col])/seg_size), num])
				new_feature = []
				for i in range(int(len(data[:, col])/seg_size)):
					window = data[i*seg_size:(i+1)*seg_size, col]
					k = 0
					if self.var_mean.get():
						feature[i, k] = feature_extraction.mean(window)
						k += 1;
					if self.var_std.get():
						feature[i, k] = feature_extraction.std(window)
						k += 1;
					if self.var_min.get():
						feature[i, k] = feature_extraction.getmin(window)
						k += 1;
					if self.var_max.get():
						feature[i, k] = feature_extraction.getmax(window)
						k += 1;
					if self.var_rms.get():
						feature[i, k] = feature_extraction.rms(window)
						k += 1;
					if self.var_energy.get():
						energy_vector = feature_extraction.energy_for_each_freq_band(window, eng_odr, eng_bs)
						temp = np.append(feature[i, :], energy_vector)
						new_feature.append(temp)
					else:
						new_feature.append(feature[i, :])

				new_feature = np.array(new_feature)

				
				if col == 0:
					ex_feature = new_feature
				else:
					ex_feature = np.append(ex_feature, new_feature, axis=1)


			num_features = ex_feature.shape[1] // 3

			num_axis = self.var_accel_x.get() + self.var_accel_y.get() + self.var_accel_z.get()

			trn_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

			k = 0

			if self.var_accel_x.get() == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
				k += 1
			if self.var_accel_y.get() == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
				k += 1
			if self.var_accel_z.get() == 1:
				trn_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
				k += 1;

			final_features.extend(trn_feature)
			for i in range(len(trn_feature)):
				labels.append(label)


			self.pb['value'] = 10 + (idx+1)/num_rows * 80
			self.trn.update_idletasks() 
			time.sleep(0.5)

		final_features = np.array(final_features)
		labels = np.array(labels)


		if self.res_text.get('1.0', END) != []:
			self.res_text.delete('1.0', END)

		if model == "Random Forest":
			acc_trn, acc_tst, acc_oob = train.RandomForest(final_features, labels, trn_ratio, save_model)
			self.res_text.insert(END, "training accuracy: " + '\n' + str(acc_trn) + '\n' + '\n')
			self.res_text.insert(END, "test accuracy: " + '\n' + str(acc_tst) + '\n' + '\n')
			self.res_text.insert(END, "out of bag accuracy: " + '\n' + str(acc_oob) + '\n')

		if model == "SVM":
			acc_trn, acc_tst = train.SVM(final_features, labels, trn_ratio, save_model)
			self.res_text.insert(END, "training accuracy: " + '\n' + str(acc_trn) + '\n' + '\n')
			self.res_text.insert(END, "test accuracy: " + '\n' + str(acc_tst) + '\n' + '\n')

		self.pb['value'] = 100
		self.trn.update_idletasks() 
		time.sleep(0.5)

	def start_predict_offline(self):
		if self.file_text.get("1.0", END) == "\n":
			messagebox.showerror("Error", "Please select test file!")
			return
		else:
			test_file = self.file_text.get("1.0", END)
			test_file = test_file.split("\n")[0]

		if self.model_text.get("1.0", END) == "\n":
			messagebox.showerror("Error", "Please select test model!")
			return
		else:
			offline_model = self.model_text.get("1.0", END)
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

				
				if col == 0:
					ex_feature = new_feature
				else:
					ex_feature = np.append(ex_feature, new_feature, axis=1)


			num_features = ex_feature.shape[1] // 3

			num_axis = int(test_config['X'].values[0]) +int(test_config['Y'].values[0]) + int(test_config['Z'].values[0])

			tst_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

			k = 0

			if int(test_config['X'].values[0]) == 1:
				tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
				k += 1
			if int(test_config['Y'].values[0]) == 1:
				tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
				k += 1
			if int(test_config['Z'].values[0]) == 1:
				tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
				k += 1;

			final_features.extend(tst_feature)
			for i in range(len(tst_feature)):
				labels.append(label)

		final_features = np.array(final_features)
		labels = np.array(labels)

		test_model = pickle.load(open(offline_model, 'rb'))
		pred = test_model.predict(final_features)
		acc_tst = accuracy_score(labels, pred)

		lb_acc_tst = Label(self.offline, text=acc_tst, font=("Helvetica", "12", "bold italic"))
		lb_acc_tst.place(relx=0.6, rely=0.28)


	def start_predict_online(self, data_online):
		if self.model_text2.get("1.0", END) == "\n":
			messagebox.showerror("Error", "Please select test model!")
			return
		else:
			online_model = self.model_text2.get("1.0", END)
			online_model = online_model.split("\n")[0]

		if self.res_text2.get("1.0", END) == "\n":
			self.res_text2.delete("1.0", END)



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

		if self.flag == False:
			self.online_window = []
			self.online_window_size = seg_size * 1.5
			self.flag = True


		if self.online_window_size == 0:
			
			data = np.array(self.online_window)

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

					
					if col == 0:
						ex_feature = new_feature
					else:
						ex_feature = np.append(ex_feature, new_feature, axis=1)
						

				num_features = ex_feature.shape[1] // 3

				num_axis = int(test_config['X'].values[0]) +int(test_config['Y'].values[0]) + int(test_config['Z'].values[0])

				tst_feature = np.empty([ex_feature.shape[0], num_axis*num_features])

				k = 0

				if int(test_config['X'].values[0]) == 1:
					tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, 0:num_features]
					k += 1
				if int(test_config['Y'].values[0]) == 1:
					tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features:num_features*2]
					k += 1
				if int(test_config['Z'].values[0]) == 1:
					tst_feature[:, k*num_features:(k+1)*num_features] = ex_feature[:, num_features*2:num_features*3]
					k += 1;

				test_model = pickle.load(open(online_model, 'rb'))
				predict_output = test_model.predict(tst_feature)
				self.res_text2.insert(END, predict_output[0])


			self.flag = False
		else:
			self.online_window.append(data_online)
			self.online_window_size -= 1


	def run_thread(self):
		return


if __name__ == '__main__':
	root = Tk()
	app = MLAPP(root)
	root.mainloop()