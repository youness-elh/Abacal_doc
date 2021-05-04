import numpy as np
import os
import subprocess 
import time
import config as cfg

directory = cfg.directories["main"]
script = 'code/extraction_module_new'
script_path = os.path.join(directory,script)#'//isi/w/elh/Optimizer_tool/code/extraction_module.py'
comments_file = os.path.join(directory,'state.txt')

def wait_step(text,keyword='end now'):
	print('\n \t Waiting for extraction to be done! Check state.txt for details!!! \n')	
	while (keyword not in str(text[-1])):
		oldfile = open(comments_file,'r+')
		text = oldfile.readlines()
		#print(str(text[-1]))
		#print('\n \t Waiting for extraction to be done! Check state.txt for details!!! \n')	
		#time.sleep(30)
		oldfile.close()
		
print('\n---------------Ongoing extraction of temperature profiles----------------------- \n' )
oldfile = open(comments_file,'r+')
text = oldfile.readlines()
if 'start now' in str(text[-1]):
	subprocess.call(['abaqus_2019 viewer noGUI='+str(script_path) ],shell=True)
oldfile.close()
##wait
wait_step(text)
