import numpy as np
import os
import subprocess 
import time
import config as cfg

####################################
##############Inputs################
####################################
directory = cfg.directories["main"]
#Abaqus post processing
Postprocessing = cfg.directories['modules']['Postprocessing_module']
Postprocessing_path = os.path.join(directory,Postprocessing)

#state file and profile files
state_file = cfg.output['state_file']
state_file = os.path.join(directory,state_file)

def wait_step(text,keyword='end now'):
	print('\n \t Waiting for extraction to be done! Check state.txt for details!!! \n')	
	while (keyword not in str(text[-1])):
		oldfile = open(state_file,'r+')
		text = oldfile.readlines()
		#print(str(text[-1]))
		#print('\n \t Waiting for extraction to be done! Check state.txt for details!!! \n')	
		#time.sleep(30)
		oldfile.close()
		
print('\n---------------Ongoing extraction of temperature profiles----------------------- \n' )
oldfile = open(state_file,'r+')
text = oldfile.readlines()
if 'start now' in str(text[-1]):
	subprocess.call(['abaqus_2019 viewer noGUI='+str(Postprocessing_path) ],shell=True)
oldfile.close()
##wait
wait_step(text)
