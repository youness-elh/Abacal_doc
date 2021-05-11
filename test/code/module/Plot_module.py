'''
This module allow the creation of graphs of the temperature profiles for each iteration of the optimization process.

.. comment 

	import numpy as np
	import math
	import sys
	import os
	import re
	import shutil #deleting a directory
	from numpy import savetxt
	import time
	import matplotlib.pyplot as plt
	import config as cfg

	directory = cfg.directories["main"]
	time_weld = int(cfg.Abaqus["time"])
	speed_weld = float(cfg.Abaqus["speed"])


	start_time_plot = time.time()

	###########################################################
	################## Parameters to set ######################
	###########################################################
	directory = directory
	report_file= os.path.join(directory,'T_t extraction.txt') #exporting data to a created text file
	plot_profiles = os.path.join(directory,"Plot/T-t profiles"+str(sys.argv[1])+str('.png'))
	radius = 76.2

	######################################################################
	############ load text file containing the T-t profiles  #############
	######################################################################
	profiles = np.loadtxt(report_file,delimiter=",")			
	#print(profiles)
	######################################################################
	####################### plot the T-t profiles and save it  #######################
	######################################################################
	dim = profiles.shape[1]-3 #number of frames
	mesurements = profiles.shape[0]
	Label = profiles[:,0]
	speed = speed_weld
	time_max = time_weld
	degree = np.arange(1,dim+1)#1 to 371
	arc = 2*math.pi*radius*degree/360. # arc lengt
	Time =np.linspace(0,time_max,dim)#arc/speed #time

	plt.figure()
	for i in range(mesurements):
		plt.plot(Time,profiles[i,3:],"--",label = str(Label[i])+" mm")
		plt.ylim(19, 1.2*max(profiles[i,3:]))

	plt.title("Profile of the temperature for different measurement points")

	plt.ylabel("Temperature [C]")

	plt.xlabel("Time [s]")
	#plt.xlabel("Degree [°]")

	plt.legend(loc=0)

	plt.savefig(plot_profiles,format='png')#the name of the saved file contains float numbers which provokes a warning msg
	#
	#plt.show()

	end_time_plot = time.time()
	elapsed_time_plot = end_time_plot - start_time_plot 
	#print("Elapsed time to plot the profiles is "+str(elapsed_time_plot)+" seconds")
	print("--------------The profiles are plotted in the Plot folder--------------")
'''

