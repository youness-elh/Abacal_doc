import numpy as np
import math
import sys
import os
import re
import shutil #deleting a directory
from numpy import savetxt
import time
import matplotlib.pyplot as plt
import sys
import config as cfg
directory = cfg.directories["main"]

start_time = time.time()

###########################################################
################## Parameters to set ######################
###########################################################
directory = directory+'/test'#'//isi/w/elh/Optimizer_tool/test'#'//isi/w/ditt/Optimizer_tool/test'
param_line = cfg.pre_Abaqus["lines_parameters"]# array of the line number of the position of the parameter - 1
Parameter_name = cfg.pre_Abaqus["Parameters_name"]
DFLUX_file = 'SP013_X6CR.f'
modify_file= os.path.join(directory,DFLUX_file) #file to modify	
number_parameters = len(sys.argv)
if number_parameters >= 2:
	n = number_parameters-1
	new_par = np.zeros(n)
	######################################################################
	##################### modify the old parameters  #####################
	######################################################################
	print('\nThe number of changed parameters is: '+str(n))
	for i in range(n):
		######################################################################
		##################### open file and read lines  ######################
		######################################################################
		oldfile = open(modify_file,'r')
		text = oldfile.readlines()
		new_par[i] = float(sys.argv[i+1])
		
		line_tomodify= text[param_line[i]]
		try:
			if str(Parameter_name[i]) in line_tomodify:
				old = text[param_line[i]]
				text[param_line[i]] = '\t'+ str(Parameter_name[i])+' = '+str(new_par[i])+" \n"
				
				newfile = open(modify_file,'w')
				text = newfile.writelines(text)
				print("\n"+str(old)+ "\t \t become: "+str(new_par[i]))
			else: 
				print("Given line of the parameter "+ str(Parameter_name[i])+ " is not detected")
		except KeyError:
			print("The number of arguments/parameters should be 5 at maximum!!")

		newfile.close()


	end_time = time.time()
	elapsed_time = end_time - start_time 
	#print("Elapsed time to update the parameters "+str(elapsed_time)+" seconds")
	
else:
	print('Please specify new values!')


