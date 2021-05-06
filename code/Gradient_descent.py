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
'''
This is a proposed minimization algorithm for the calibration of the heat source parameters i.e Goldak model in the welding process model. 
The model is simulated by Abaqus CAE and the temperature profiles are driven from the obtained Abaqus odb. 
The temperature profiles are then compared to to the target profiles and the dependent cost function is minimized using this proposed optimization algorithm. 
In fact, it is the gradient descent algorithm using the backtracking line search based on the **Armijo_Goldstein condition** in order to determine the step.
'''

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

#########################################################################
###############################Results###################################
#########################################################################

#Loss_list
Loss_list_path = cfg.output['Results']['Loss_list']
#Profiles_list
Profiles_list_path = cfg.output['Results']['Profiles_list']
#Q_list
Q_list_path = cfg.output['Results']['Q_list']
#delta_list
delta_list_path = cfg.output['Results']['delta_list']
#step_list
step_list_path = cfg.output['Results']['step_list']
#gradient_list
gradient_list_path = cfg.output['Results']['gradient_list']
#error_list
error_list_path = cfg.output['Results']['error_list']
#error_abs_list
error_abs_list_path = cfg.output['Results']['error_abs_list']

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

#########################################################################
#########################Gradient parameters#############################
#########################################################################

######################################
tol = cfg.Gradient_algo['tolernce']
max_iter = cfg.Gradient_algo['max_iter']
max_iter_step = cfg.Gradient_algo['max_iter_step']
step = cfg.Gradient_algo['step']
#####################################

###########################################################################
###########################################################################

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
	
def Loss_calc(Q1,target,penalisation=1e-30):
	print('\n--------------------------------------------------------------------')
	print('----------------------Calculation of Loss--------------------------')
	print('--------------------------------------------------------------------\n')
	#Calculate the initial Loss (of Q1)
	profile = get_profiles(Q1)
	loss = Loss(Q1,profile,target,penalisation)
	return loss,profile
	
#loss_0 = Loss_init(Q_step1,Targets,1e-30)
#print('the initial loss is=',loss_0)

	
def Gradient_Loss(L1,Q1,Q2,target,penalisation=1e-30):
	#calculate profiles using abqus cmd
	#assert (Q1 != Q2).any(), "The initial parameter set should be not the same!"
	assert len(Q1) == len(Q2), "The initial values of parameter array for both steps should be the same!"
	print('\n---------------------------------------------------------------------')
	print('----------------------Gradient estimation--------------------------')
	print('----------------------------------------------------------------------\n')

	n = len(Q2)
	grad = np.zeros(n)
	loss = np.zeros(n+1)
	loss[0] = L1
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
	
	grad[0]=grad[0]*800
	grad[1:-1]=grad[1:-1]/5000
	print('The gradient is ',grad)
	print('The loss is ',loss)

	
	return grad
		
	
def save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,ok=True):
	print('----------------------------------------------------')
	print('----------------Save results on files---------------')
	print('----------------------------------------------------\n')
	if ok:
		file = os.path.join(directory,Q_list_path)
		with open(file,'w') as f:
			np.savetxt(f, np.array(Q_list),delimiter=",")
			
		file = os.path.join(directory,Loss_list_path)
		with open(file,'w') as f:
			np.savetxt(f, np.array(Loss_list),delimiter=",")
		
	file = os.path.join(directory,Profiles_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(Profiles_list),delimiter=",", fmt='%s')
	
	file = os.path.join(directory,gradient_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(gradient_list),delimiter=",")
		
	file = os.path.join(directory,error_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(error_list),delimiter=",")
		
	file = os.path.join(directory,delta_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(delta_list),delimiter=",")
		
	file = os.path.join(directory,step_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(step_list),delimiter=",")
		
	file = os.path.join(directory,error_abs_list_path)
	with open(file,'w') as f:
		np.savetxt(f, np.array(error_abs_list),delimiter=",")
	
	print('-------------------------------------------------------------')
	print('----------results saved in '+str(results)+'---------')
	print('-------------------------------------------------------------\n')
######################################################################
################ Calculate initial gradient and loss  ################
######################################################################
def grad_init():
	gradient_init = Gradient_Loss(Q_step1,Q_step2,Targets,1e-30)
	print("\n")
	print('Initial gradient---------------------------------------> '+str(gradient_init))
	return gradient_init

print("\n \n-------------------------------------------------")
print("-------Starting gradient descent algorithm----------")
print("-------------------------------------------------\n \n")
######################################################################
################### Algorithm of gradient descent  ###################
######################################################################

######################################
tol = tol
max_iter = max_iter
max_iter_step = max_iter_step
step = step
#####################################

######################################################################
#######################Initialization#################################
######################################################################
error = 1.
error_abs = 1
iter = 0
total_iter = 0
Q_list = [Q_step1,Q_step2]
L1,profile1 = Loss_calc(Q_list[-2],Targets,1e-30)
L2,profile2 = Loss_calc(Q_list[-1],Targets,1e-30)
Loss_list = [L1,L2]
Profiles_list = [profile1,profile2]
gradient_list = []
delta_list = []
error_list = []
error_abs_list = []
step_list = []
###########################################################
##################adapt to dimension#######################
###########################################################
dimension = len(Q_step1) 
maximum_par = cfg.Gradient_algo['Parameters_interval']['max'][:dimension]
minimum_par = cfg.Gradient_algo['Parameters_interval']['min'][:dimension]
while ((error_abs > tol) and (iter < max_iter)):
		
	print('-----------------------------------------------------')
	print('-----------Optimal Q in iteration '+str(iter)+' = '+str(Q_list[-1])+'-----------')
	print('-----------------------------------------------------\n')
	gradient =  Gradient_Loss(Loss_list[-2],Q_list[-2], Q_list[-1],Targets,1e-30)
	print('----------------------',gradient)
	Q = np.minimum(np.maximum(np.array(Q_list[-1]) - step*gradient,minimum_par),maximum_par)
	print('-------gradient and step------',gradient,step)
	L_new,profiles = Loss_calc(Q,Targets,1e-30)
	
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
		Q = np.minimum(np.maximum(np.array(Q_old) - step*gradient,minimum_par),maximum_par)
		L_new,profiles = Loss_calc(Q,Targets,1e-30)
		delta = L_new - L_old
		count +=1
		
		#save
		delta_list.append(delta)
		step_list.append(step)
		Profiles_list.append(profiles)
		error_abs_list.append(error_abs)
		
		#save
		save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,False)

	total_iter += count + iter
	#save
	if count == 0:
		Q_list.append(Q)
		Loss_list.append(L_new)
		Profiles_list.append(profiles)
		delta_list.append(delta)
		step_list.append(step)
	else:
		Q_list.append(Q)
		Loss_list.append(L_new)
		
		
	gradient_list.append(gradient)

	error = abs(Loss_list[-1]-Loss_list[-2])/Loss_list[-2] if Loss_list[-2] != 0 else 0
	error_list.append(error)
	
	error_abs = profile_target_L2(profiles,Targets)
	error_abs_list.append(error_abs)
	iter += 1
	print("For iteration "+str(iter)+" we obtain Q = "+str( Q)+ " with a relative error of "+str(error))
	#save
	save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,True)

print('--------------------------------------------------------------------')
print('----------------------------Done!-----------------------------------')
print('--------------------------------------------------------------------')

#time estimation
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations.\
\n The search for descent direction took "+str(count)+" in " +str(iter)+" main iterations ")
with open(state_file,'a') as file:
	file.write('\n-----------------------------------------------------------------------------------------------\n')
	file.write("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations including search for descent direction and " +str(iter)+" main iterations ")
	file.write('\n------------------------------------------------------------------------------------------------\n')

