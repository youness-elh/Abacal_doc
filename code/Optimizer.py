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

start_time = time.time()

###########################################################
################## Parameters to set ######################
###########################################################
directory = '//isi/w/elh/T_model_2Dopt'#'//isi/w/elh/Optimizer_tool'#'//isi/w/ditt/Optimizer_tool'

#abaqus script
scripts = 'code'
script_path = os.path.join(directory,scripts)
#results
results = os.path.join(directory,'results')
#path to simulation output
report_file= os.path.join(directory,"T_t extraction.txt")#'T_t extraction.txt') #exporting data to a created text file

#state file
state_file = os.path.join(directory,"state.txt")

#path to simulation for step 1&2 to calculate the first gradient estimation
step1_profile_file = os.path.join(directory,"Initial_data/step1_Q1000.txt")
step2_profile_file = os.path.join(directory,"Initial_data/step2_Q10000.txt")
#values of heat intensity for step 1&2
Q_step1 = 1000.0
Q_step2 = 10000.0
nb_col_profiles =80#370 #number of evaluated degrees+3
#path to the target profiles
targets_file = os.path.join(directory,"Initial_data/target_ref.txt")

######################################################################
############## load inputs containing the T-t profiles  ##############
######################################################################

#Profiles = np.loadtxt(report_file,delimiter=",")		
#nb_col_profiles =370 #Profiles.shape[1]#
profiles_step1 = np.loadtxt(step1_profile_file,delimiter=",")#
profiles_step1 = profiles_step1[:,:nb_col_profiles]

print('\n------checking shapes---------\n')
print(profiles_step1.shape)
profiles_step2 = np.loadtxt(step2_profile_file,delimiter=",")
profiles_step2 = profiles_step2[:,:nb_col_profiles]
print(profiles_step2.shape)
Targets = np.loadtxt(targets_file,delimiter=",")
Targets = Targets[:,:nb_col_profiles]
print(Targets.shape)

###########################################################
############ Loss function and its gradient ###############
###########################################################

def profile_target_L2(profile,target):
    profile = profile[:,3:]#skip coordinates
    target  = target[:,3:]#skip coordinates
	
    output = profile - target
    output = output**2
    output = 1*np.sum(output)#/(nb_col_profiles-3) #rectangle rule (width = 1 degree)
    return output

def Loss(Q,profile,target,penalisation=1e-30):
    J = profile_target_L2(profile,target) + penalisation*Q**2
    return J

def Gradient_loss(Q1,Q2,profile1,profile2,target,penalisation=1e-30):
    step = Q2 - Q1
    #assert step != 0, "Step should be != 0!"
    grad = Loss(Q2,profile2,target,penalisation)-Loss(Q1,profile1,target,penalisation)
    grad /= step
    return grad

def modify_Q_in_f(Q):
	Q = str(Q)
	subprocess.call('python '+script_path+'/change_parameters.py '+Q,shell=True)
	time.sleep(1)

def plot_profile(Q):
	Q = str(Q)
	subprocess.call('python '+script_path+'/plot_T-t_profiles.py '+Q,shell=True)
	time.sleep(1)
	
def Abaqus(Q):
	modify_Q_in_f(Q)
	subprocess.call('python '+script_path+'/Extract.py',shell=True)
	time.sleep(200)###########make sure extracted data is READY#############
	with open(report_file,'r') as f:
		Profiles = np.loadtxt(f,delimiter=",")
	f.close()
	plot_profile(Q)
	
	return Profiles

######################################################################
################ Calculate initial gradient and loss  ################
######################################################################
gradient_init = Gradient_loss(Q_step1,Q_step2,profiles_step1,profiles_step2,Targets)
print("\n")
print('Initial gradient---------------------------------------> '+str(gradient_init))
L1 = Loss(Q_step1,profiles_step1,Targets)
L2 = Loss(Q_step2,profiles_step2,Targets)
print("Initial Loss of step 1 and initial Loss of step 2------> "+str(L1)+ " & " +str(L2))


print("\n \n---------------------------------------------------")
print("-------Starting gradient descent algorithm----------")
print("----------------------------------------------------\n \n")
######################################################################
################### Algorithm of gradient descent  ###################
######################################################################

delta = L2-L1
error = 1.
error_abs = 1
tol = 1.e-1
max_iter = 24 # (~three hours per cycle)
max_iter_step = 24 # 
step = 19800
Loss_list = [L1,L2]
Q_list = [Q_step1,Q_step2]
Profiles_list = [profiles_step1,profiles_step2]
gradient_list = []
delta_list = []
error_list = []
error_abs_list = []
step_list = []
iter = 0
total_iter = 0

while ((error_abs > tol) and (iter < max_iter)):
	if iter == 1:
		step = 200
	print('-----------------------------------------------------')
	print('-----------Optimal Q in iteration '+str(iter)+' = '+str(Q_list[-1])+'-----------')
	print('-----------------------------------------------------\n')

	gradient =  np.sign(Gradient_loss(Q_list[-2], Q_list[-1],Profiles_list[-2],Profiles_list[-1],Targets))
	print('----------------------',gradient)
	Q = max(Q_list[-1] - step*gradient,0)
	print('-------------',gradient,step)
	profiles = Abaqus(Q)
	with open(report_file,'r') as f:
		profiles = np.loadtxt(f,delimiter=",")
	f.close()
	L_new = Loss(Q,profiles,Targets)
	
	#step of descent direction
	count = 0
	Q_old = Q_list[-1]
	delta = L_new - Loss_list[-1]
	while ((count < max_iter_step)  and (delta >=0)):
		print('----------------------------------------------------------')
		print('-----Looking for descent direction for iteration '+str(count+1)+'------')
		print('----------------------------------------------------------\n')

		step /= 1.3
		
		Q = max(Q_old - step*gradient,0)
		profiles = Abaqus(Q)
		L_new = Loss(Q,profiles,Targets)
		delta = L_new - Loss_list[-1]
		count +=1
		#save
		Q_list.append(Q)
		Loss_list.append(L_new)
		delta_list.append(delta)
		step_list.append(step)
		Profiles_list.append(profiles)
		
		#save
		with open(results+'/Profiles_list.txt','w') as f:
			np.savetxt(f, np.array(Profiles_list),delimiter=",", fmt='%s')
		
		with open(results+'/Loss_list.txt','w') as f:
			np.savetxt(f, np.array(Loss_list),delimiter=",")
		
		with open(results+'/Q_list.txt','w') as f:
			np.savetxt(f, np.array(Q_list),delimiter=",")
			
		with open(results+'/delta_list.txt','w') as f:
			np.savetxt(f, np.array(delta_list),delimiter=",")
			
		with open(results+'/step_list.txt','w') as f:
			np.savetxt(f, np.array(step_list),delimiter=",")


	total_iter += total_iter+ count + iter
	#save
	if count == 0:
		Q_list.append(Q)
		Loss_list.append(L_new)
		Profiles_list.append(profiles)
		delta_list.append(delta)
		step_list.append(step)
		
	gradient_list.append(gradient)

	error = abs(Loss_list[-1]-Loss_list[-2])/Loss_list[-2] if Loss_list[-2] != 0 else 0
	error_list.append(error)
	
	error_abs = profile_target_L2(profiles,Targets)
	error_abs_list.append(error_abs)
	iter += 1
	print("For iteration "+str(iter)+" we obtain Q = "+str( Q)+ " with a relative error of "+str(error))
	print('----------------------------------------------------')
	print('----------------Save results on files---------------')
	print('----------------------------------------------------')
	
	with open(results+'/Loss_list.txt','w') as f:
		np.savetxt(f, np.array(Loss_list),delimiter=",")
		
	with open(results+'/Q_list.txt','w') as f:
		np.savetxt(f, np.array(Q_list),delimiter=",")
		
	with open(results+'/Profiles_list.txt','w') as f:
		np.savetxt(f, np.array(Profiles_list),delimiter=",", fmt='%s')
	
	with open(results+'/gradient_list.txt','w') as f:
		np.savetxt(f, np.array(gradient_list),delimiter=",")
		
	with open(results+'/error_list.txt','w') as f:
		np.savetxt(f, np.array(error_list),delimiter=",")
		
	with open(results+'/delta_list.txt','w') as f:
		np.savetxt(f, np.array(delta_list),delimiter=",")
		
	with open(results+'/step_list.txt','w') as f:
		np.savetxt(f, np.array(step_list),delimiter=",")
		
	with open(results+'/error_abs_list.txt','w') as f:
		np.savetxt(f, np.array(error_abs_list),delimiter=",")
		
	print('-------------------------------------------------------------')
	print('----------results saved in '+str(results)+'---------')
	print('-------------------------------------------------------------\n')

print('--------------------------------------------------------------------')
print('----------------------------Done!-----------------------------------')
print('--------------------------------------------------------------------')
total_iter=0
#time estimation
end_time = time.time()
elapsed_time = end_time - start_time
with open(state_file,'a') as file:
	file.write('\n-----------------------------------------------------------------------------------------------\n')
	file.write("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations including search for descent direction and " +str(iter)+" main iterations ")
	file.write('\n------------------------------------------------------------------------------------------------\n')

