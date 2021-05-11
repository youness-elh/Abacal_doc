import numpy as np
import math
import sys
import os
import re
import shutil #deleting a directory
from numpy import savetxt
import time
import matplotlib.pyplot as plt
import subprocess

######################################################################
### Testing the nodule to change parameters inside the user file  ####
######################################################################
#we need the directory the module to test and insert it in the line 31

def unpackarraytostring(Q,space=' '):
	out=''
	for arg in Q:
		out += str(arg)+space
	return out

def modify_Q_in_f(Q):
	#for multi parameters: the order of the parameters is important 
	#for this 2D version the parameters are ordered as follow QT,  
	if type(Q) != int:
		Q = unpackarraytostring(Q)
	print(Q)
	subprocess.call('python '+'../code'+'/change_parameters.py '+Q,shell=True)
	

print('---------------test--------------------')
Q=[30000,1,2]
modify_Q_in_f(Q)