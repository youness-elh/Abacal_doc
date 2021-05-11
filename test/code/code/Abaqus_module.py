import numpy as np
import os
import subprocess 
import time
import config as cfg
directory = cfg.directories["main"]


############################################################
#################### inputs ################################
#############################################################
directory = directory#'//isi/w/elh'#'//isi/w/ditt'

#odb files
inp_file_name = 'test/SP013_X6CR.inp'
f_file_name = 'test/SP013_X6CR.f'
f_file = os.path.join(directory,f_file_name)
inp_file = os.path.join(directory,inp_file_name)

#file having the state of the odb model
state_odb_name = 'test/SP013_X6CR.sta'
state_odb = os.path.join(directory,state_odb_name)

#file having the state of the extraction process
state_file_name = 'state.txt'
state_file = os.path.join(directory,state_file_name)

odb_dir = 'test'
path_odb_dir = os.path.join(directory,odb_dir)



############################################################
#################### functionss ############################
#############################################################

def clean_files():
	files = [i for i in os.listdir(path_odb_dir) if i not in ('SP013_X6CR.inp', 'SP013_X6CR.f', 'WSModell_Kinematic_new.ssc','SP013_X6CR.fem')]#'21L_R60_t15_3d_L19_heat.odb','21L_R60_t15_3d_L19_heat.sta',
	#print(i)
	subprocess.call(['rm','-rf'] + files)
	
def exist(state_path):
	return os.path.exists(state_path)

def wait_step(state,keyword='SUCCESSFULLY'):
	i = 0
	while not exist(state):
		if i == 0:
			print('\n---Waiting for odb.sta to appear and the start of iteration!---\n')
			i+=1
	
	file = open(state,'r')
	text = file.readlines()
	file.close()
	while(len(text) < 2):
		#print('Waiting more time!!')
		file = open(state,'r')
		text = file.readlines()
		file.close()
	
	file = open(state,'r')
	text = file.readlines()
	last_line = text[-1]
	file.close()
	k=0
	while keyword not in last_line:
		#print('last_line',last_line)
		file = open(state,'r')
		text = file.readlines()
		last_line = text[-1]
		file.close()
			
		if keyword == 'SUCCESSFULLY':
			if k==0:
				print('------------Waiting for odb for around 20 minutes!--------------')
				k+=1
		else:
			pass
		file.close()
	if keyword == 'SUCCESSFULLY':
		print('--------Ready to start extraction!-------')
	else:
		pass			

##########################################################
#################### call ################################
##########################################################

	
print('---------Clean old job outputs----------')
clean_files()
print('----------------------------------------')
print('-------------Run abaqus job-------------')
print('----------------------------------------')
subprocess.call(['abaqus_2019 -j SP013_X6CR -cpus 8 user=SP013_X6CR'], shell=True)
# wait for odb
wait_step(state_odb,'SUCCESSFULLY')
#extraction
print('--------Extraction: starting now--------')
with open(state_file,'a') as file:
	file.write("\n -----------------Extraction: start now----------------")




