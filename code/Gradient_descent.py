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
import config as cfg

start_time = time.time()

###########################################################
################## Parameters to set ######################
###########################################################
directory = cfg.directories["main"]

#abaqus script
odb_dir = 'test'
path_odb_dir = os.path.join(directory,odb_dir)

scripts = 'code'
script_path = os.path.join(directory,scripts)
#results
results = os.path.join(directory,'results')
#path to simulation output
report_file= os.path.join(directory,"T_t extraction.txt")#'T_t extraction.txt') #exporting data to a created text file
#extraction of profiles
extract = 'code/extraction_module_new'
extract_path = os.path.join(directory,extract)
#state file
state_file = os.path.join(directory,"state.txt")

#path to simulation for step 1&2 to calculate the first gradient estimation
step1_profile_file = os.path.join(directory,"Initial_data/step1_Q1000.txt")
step2_profile_file = os.path.join(directory,"Initial_data/step2_Q10000.txt")
#values of heat intensity for step 1&2
Q_step1 = np.array([1000.0,1.,2.5])
Q_step2 = np.array([10000.0,1.,5.])
nb_col_profiles =cfg.post_Abaqus["extracted_collumns"]+3#number of evaluated degrees+3
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

def Loss(Q,profile,target,penalisation1=1e-30):#penalisation can be also an array for multi parameters regulation
    J = profile_target_L2(profile,target) + sum(penalisation1*Q**2) # sum -->in case Q is an array (multi parameters)
    return J

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
	Q = str(Q)
	subprocess.call('python '+script_path+'/change_parameters.py '+Q,shell=True)

def plot_profile(Q):
	if type(Q) != int:
		Q = unpackarraytostring(Q,'_')
		Q = Q[:-1]
	Q = str(Q)
	subprocess.call('python '+script_path+'/plot_T-t_profiles.py '+Q,shell=True)
	
def Abaqus(Q):
	modify_Q_in_f(Q)
	print('--------Abaqus job to submit--------')
	subprocess.call('python '+script_path+'/Extract.py',shell=True)
	time.sleep(2)
	
def Extract(Q):
	subprocess.call('python '+script_path+'/Extract_profiles.py',shell=True)
	time.sleep(2)
	with open(report_file,'r') as f:
		Profiles = np.loadtxt(f,delimiter=",")
	f.close()
	plot_profile(Q)

	return Profiles
	
def get_profiles(Q):
	Abaqus(Q) #modify input files and generate odb
	#Extract the profile from the odb
	print('\n-----------------------------------------------------------')
	print('----------------------Extraction starts now----------------')
	print('-----------------------------------------------------------\n')
	profiles = Extract(Q)
	return profiles
	
def Gradient_Loss(Q1,Q2,target,penalisation=1e-30):
	#calculate profiles using abqus cmd
	assert (Q1 != Q2).any(), "The initial parameter set should be not the same!"
	assert len(Q1) == len(Q2), "The initial values of parameter array for both steps should be the same!"
	print('\n---------------------------------------------------------------------')
	print('----------------------Gradient estimation--------------------------')
	print('----------------------------------------------------------------------\n')
	if type(Q2) != int:
		n = len(Q2)
		grad = np.zeros(n)
		loss = np.zeros(n+1)
		#Calculate the initial Loss (of Q1)
		profile1 = get_profiles(Q1)
		loss[0] = Loss(Q1,profile1,target,penalisation)
		#print('the initial loss is=',loss[0])
		
		for i in range(n):
			#run Abqus simulation for each parameter constellation
			Q = Q1.copy()
			Q[i] = Q2[i] 
			if Q1[i] == Q2[i]:
				grad[i]= 0.1
			else:
				Profiles = get_profiles(Q)
				#Calculate the loss
				loss[i+1] = Loss(Q,Profiles,target,penalisation)
				grad[i] = loss[i+1]-loss[0]
				grad[i] /= Q2[i]-Q1[i]
		
		grad[0]=grad[0]*1000
		grad[1:-1]=grad[1:-1]/3000
		print('The gradient is ',grad)
		print('The loss is ',loss)
		
	else:
		print('use 1D script')
	
	return grad
		
		
####################################################################
grad_init = Gradient_Loss(Q_step1,Q_step2,Targets,1e-30)

######################################################################
################ Calculate initial gradient and loss  ################
######################################################################
gradient_init = grad_init
print("\n")
print('Initial gradient---------------------------------------> '+str(gradient_init))
L1 = Loss(Q_step1,profiles_step1,Targets,1e-30)
L2 = Loss(Q_step2,profiles_step2,Targets,1e-30)
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
max_iter = 10 # (~20 min per cycle)
max_iter_step = 24 # 
step = 1.#np.array([19800,0.5,0.5,0.5,0.5])#################tochange#####################
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

###########################################################
##################adapt to dimension#######################
###########################################################
dimension = len(Q_step1) 
maximum_par = np.array([50000,6,6,6,6])[:dimension]
while ((error_abs > tol) and (iter < max_iter)):
		
	print('-----------------------------------------------------')
	print('-----------Optimal Q in iteration '+str(iter)+' = '+str(Q_list[-1])+'-----------')
	print('-----------------------------------------------------\n')
	if iter ==0:
		gradient = gradient_init
	else:
		gradient =  Gradient_Loss(Q_list[-2], Q_list[-1],Targets,1e-30)
	print('----------------------',gradient)
	Q = np.minimum(np.maximum(np.array(Q_list[-1]) - step*gradient,0.9*np.ones(dimension)),maximum_par)
	print('-------gradient and step------',gradient,step)
	profiles = get_profiles(Q)
	
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
		Q = np.minimum(np.maximum(np.array(Q_old) - step*gradient,0.9*np.ones(dimension)),maximum_par)
		profiles = get_profiles(Q)
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
			
		with open(results+'/gradient_list.txt','w') as f:
			np.savetxt(f, np.array(gradient_list),delimiter=",")


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

