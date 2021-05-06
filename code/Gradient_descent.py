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
odb_dir = cfg.directories['odb_folder']
path_odb_dir = os.path.join(directory,odb_dir)

scripts = cfg.directories['code_folder']
script_path = os.path.join(directory,scripts)

results_folder =cfg.directories['results_folder']
results = os.path.join(directory,results_folder)
#########################################################################
###############################modules###################################
#########################################################################

#update module
update_module = cfg.directories['modules']['update_module']
update_path = os.path.join(directory,update_module)

#visualization module
Plot_module = cfg.directories['modules']['Plot_module']
Plot_path = os.path.join(directory,Plot_module)

#visualization module
Abaqus_module = cfg.directories['modules']['Abaqus_module']
Abaqus_path = os.path.join(directory,Abaqus_module)

#extraction of profiles
Extract = cfg.directories['modules']['extraction_module']
Extract_path = os.path.join(directory,Extract)
###########################################################################
###########################################################################

#state file and profile files
state_file = cfg.output['state_file']
state_file = os.path.join(directory,state_file)

#path to simulation output
profiles_file = cfg.output['export_file']
report_file= os.path.join(directory,profiles_file)#exporting data to a created text file

#values of parameters for step 1&2
Q_step1 = cfg.Gradient_algo['Q_0']
Q_step2 = cfg.Gradient_algo['Q_1']

nb_col_profiles =cfg.post_Abaqus["extracted_columns"]+3#number of evaluated degrees+3
#path to the target profiles
targets_file = cfg.directories['target_file']
targets_file = os.path.join(directory,targets_file)

######################################################################
############## load inputs containing the target T-t profiles  ##############
######################################################################
def Load_profile(targets_file,nb_col_profiles):
	Targets = np.loadtxt(targets_file,delimiter=",")
	Targets = Targets[:,:nb_col_profiles]
	return Targets

Targets = Load_profile(targets_file,nb_col_profiles)
print('\nThe target profile has the following shape: \
'+str(Targets.shape)+'.\nThe lines representing different position of measurements!\n\
The first 3 columns representing the temperature measurements coordinates!\
\nThe other columns representing the temperature value for different time frames!\n')

###########################################################
############ Loss function and its gradient ###############
###########################################################

def profile_target_L2(profile,target):
    profile = profile[:,3:]#skip coordinates
    target  = target[:,3:]#skip coordinates
	
    output = profile - target
    output = output**2
    output = 1*np.sum(output)#rectangle rule (width = 1 degree)
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
	subprocess.call('python '+update_path+' '+Q,shell=True)

def test():
	#should include the error msg for number of parameters!
	modify_Q_in_f(np.array([1000.0,1.,1.,1.,1.]))
#test()

def plot_profile(Q):
	if type(Q) != int:
		Q = unpackarraytostring(Q,'_')
		Q = Q[:-1]
	Q = str(Q)
	subprocess.call('python '+Plot_path+' '+Q,shell=True)
	
def Abaqus(Q):
	modify_Q_in_f(Q)
	print('--------Abaqus job to submit--------')
	subprocess.call('python '+Abaqus_path,shell=True)
	time.sleep(2)
	
def Extract(Q):
	subprocess.call('python '+Extract_path,shell=True)
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
	#assert (Q1 != Q2).any(), "The initial parameter set should be not the same!"
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
				grad[i]= 0.01
			else:
				Profiles = get_profiles(Q)
				#Calculate the loss
				loss[i+1] = Loss(Q,Profiles,target,penalisation)
				grad[i] = loss[i+1]-loss[0]
				grad[i] /= Q2[i]-Q1[i]
		
		grad[0]=grad[0]*800
		grad[1:-1]=grad[1:-1]/5000
		print('The gradient is ',grad)
		print('The loss is ',loss)
		
	else:
		print('use 1D script')
	
	return grad
		
	

######################################################################
################ Calculate initial gradient and loss  ################
######################################################################

grad_init = Gradient_Loss(Q_step1,Q_step2,Targets,1e-30)
gradient_init = grad_init
print("\n")
print('Initial gradient---------------------------------------> '+str(gradient_init))

print("\n \n-------------------------------------------------")
print("-------Starting gradient descent algorithm----------")
print("-------------------------------------------------\n \n")
######################################################################
################### Algorithm of gradient descent  ###################
######################################################################

delta = -1
error = 1.
error_abs = 1
tol = 1.e-1
max_iter = 10 # (~20 min per cycle)
max_iter_step = 24 # 
step = 1.#np.array([19800,0.5,0.5,0.5,0.5])#################tochange#####################
Loss_list = []
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
	L_old = Loss_list[-1]
	delta = L_new - L_old 
	while ((count < max_iter_step)  and (delta >=0)):
		print('----------------------------------------------------------')
		print('-----Looking for descent direction for iteration '+str(count+1)+'------')
		print('----------------------------------------------------------\n')

		step /= 1.3
		Q = np.minimum(np.maximum(np.array(Q_old) - step*gradient,0.9*np.ones(dimension)),maximum_par)
		profiles = get_profiles(Q)
		L_new = Loss(Q,profiles,Targets)
		delta = L_new - L_old
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

