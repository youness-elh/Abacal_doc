'''

This is a proposed minimization algorithm for the fitting of the heat source parameters i.e Goldak model in the welding process simulation. 
The model is simulated by Abaqus CAE and the temperature profiles are driven from the obtained Abaqus odb. 
The temperature profiles are then compared to to the target profiles and the dependent cost function is minimized using this proposed optimization algorithm. 
In fact, it is the gradient descent algorithm using the backtracking line search based on the **Armijo_Goldstein condition** in order to determine the step. For more detail about this method one can consult the fllowing wikipedia page: `here <https://en.wikipedia.org/wiki/Backtracking_line_search>`_.

The following draft summerrize the used implementation of both the gradient descent and it searching algorithm for the step.

.. figure:: file:///home/you/Documents/Master%20thesis/test/code/module/img/gradient_0.PNG
    :align: center
    :alt: The implemeted gradient descent algorithm
    :figclass: align-center

    The implemeted gradient descent algorithm. **Left loop** for the backtracking line search. **Right loop** describing the gradient descent implementation overview.

Requirements and imports
########################

The following libraries need to be imported into this module:

.. seealso:: One can use 'setup.py' file for installing the requirements. 

.. warning:: it is not yet done!

.. code-block:: python

	import sys
	import os
	import re
	import time
	import math
	import shutil 
	import subprocess	
	import numpy as np
	from numpy import savetxt
	import matplotlib.pyplot as plt
		&
	import config as cfg

The Config module is used as a configuration file for the usage of this optimizer. it will be explained later on the next section how one can use the calibration tool using the config.py file and few instructions to know.

Performance
############

This section describe the used methids to optimize the presented tool.

Time module
***********
The time() method of the imported time module is used to measure the elapsed time in seconds as floating-point number.

:Example:

>>> start_time = time.time()
>>> Some running script!
>>> end_time = time.time()
>>> elapsed_time = end_time - start_time
>>> print("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations. The search for descent direction took "+str(count)+" in " +str(iter)+" main iterations ")

Parallel computing
******************

.. warning:: In this module version no parallelisation was implemented. 

In the coming versions, there would be a possibility of parallelizing in the following bullet points for instance:

.. todo:: 
	* The Abaqus job submission.
	* The extraction of the temperature profiles.
	* The searching algorithm for the optimal step e.g Bisection method.

Inputs
######

The inputs are imported from the `config` module and structured in six groups:

* Directories like the source code path, abaqus files path or target file path
* Pre processing parameters like the name of the parameters to calibrate and their respective position line
* Abaqus inputs regarding the welding process parameters like the welding speed or the welding simulation time
* Post processing parameters like the number of the measurement positions, their coordinates and the number of frames to be extracted 
* Gradient descent algorithm settings like initial parameter set, their interval, tolerance of the square absolute error, the step, etc
* Outputs files and paths to store results and check simulation state for instance

For more details about the `configuration` file one can check the following link:

.. Seealso:: config module

The following code snipet is representing the input aquisition according to the `configuration' module settings: 

.. code-block:: python

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
	assert len(Q_step1)==len(Q_step1), 'The inputs should be of the same shape!'
	assert len(Q_step1)<6, 'The inputs shape should contains 5 value at maximum!'

	nb_col_profiles =cfg.post_Abaqus["extracted_columns"]+3#number of evaluated degrees+3
	#path to the target profiles
	targets_file = cfg.directories['target_file']
	targets_file = os.path.join(directory,targets_file)

	###########################################################
	##################adapt to dimension#######################
	###########################################################
	dimension = len(Q_step1) 
	maximum_par = cfg.Gradient_algo['Parameters_interval']['max'][:dimension]
	minimum_par = cfg.Gradient_algo['Parameters_interval']['min'][:dimension]

	#########################################################################
	#########################Gradient parameters#############################
	#########################################################################

	################################################################
	tol = cfg.Gradient_algo['tolerance']
	max_iter = cfg.Gradient_algo['max_iter']
	max_iter_step = cfg.Gradient_algo['max_iter_step']
	step = cfg.Gradient_algo['step']
	gradient_scaling = cfg.Gradient_algo['gradient_scaling'][:dimension]
	regulization = cfg.Gradient_algo['regulization'][:dimension]
	armijo_step = cfg.Gradient_algo['armijo_step']

	################################################################

	###########################################################################
	###########################################################################

.. comment:

	######################################################################
	############## load inputs containing the target T-t profiles  ##############
	######################################################################

Methods
#######

'''
def Load_profile(targets_file,nb_col_profiles):

	'''
	Returns the temperature profiles of the target measurements as a numpy array.

	.. warning:: The first three returned columns contains the coordinates of each position of the temperature sensor.

	:Example:

        >>> Targets = Gradient_descent.Load_profile(targets_file,nb_col_profiles)
	>>> print('The target profile has the following shape: '+str(Targets.shape)+'.The linesrepresenting different position of measurements!The first 3 columns representing the temperature measurements coordinates!The other columns representing the temperature value for different time frames!')

        :param targets_file: Absolute path to the target file
        :type targets_file: string
        :param nb_col_profiles: Number of columns indicating number of extracted frames from the Abaqus model
        :type nb_col_profiles: integer
        :return:  A matrix of shape (number of measurements, number of frames)
        :rtype: numpy::array
        :raises: None

	'''
	'''
.. code-block:: python
	Targets = np.loadtxt(targets_file,delimiter=",")
	Targets = Targets[:,:nb_col_profiles]
	'''

	return Targets
'''

###########################################################
############ Loss function and its gradient ###############
###########################################################


'''
def profile_target_L2(profile,target):

	'''
	Returns the square of the L2 norm of the difference between the simulated temperature profile and the targeted measured temperature profile. The Temperature profile is time dependant over the welding process and measured in several positions. The return value is calculated based on the following expression: 

	.. math::

		||Profile_{Simulated} - Profile_{Target}||_{L_2}^2 = \\int_{t=0}^{Max} (Profile_{Simulated} - Profile_{Target})^2 dt 


	Which is estimated using the rectangle rule:

	.. math::

		\\sum_{i=0}^{Frames} (T_{Simulated}[:,i] - T_{Target}[:,i])^2 

	This is implemented method to estimate the integral is described in the following wikipedia page: `here <https://en.wikipedia.org/wiki/Numerical_integration>`_.

	:Example:

        >>> L2 = Gradient_descent.profile_target_L2(profile,target)


        :param profile: The obtained simulated temperature profile
        :type profile: numpy::array
        :param target: The measured target temperature profile
        :type target: numpy::array
        :return:  A scalar
        :rtype: float
        :raises: None

	'''
	'''
.. code-block:: python
	profile = profile[:,3:]#skip coordinates
	target  = target[:,3:]#skip coordinates

	output = profile - target
	output = output**2
	output = 1*np.sum(output)#rectangle rule (width = 1 degree)
	'''
	return output


def Loss(Q,profile,target):
	'''
	Returns the value of the cost function expressed as follows:

	.. math::
	
		J = ||T_{Simulated} - T_{Target}||_{L_2}^2 + \\sum_{i=0}^{n-1} regulization[i]\\times||Q[i]||_{L_2}^2

	The regulization array is a variable imported from the `config` file in order to regulize the cost function in case of unsettled measurements.

	For information about the mathematical background one can check the following wikipedia page: `here <https://en.wikipedia.org/wiki/Lp_space>`_.

	:Example:

        >>> regulization = cfg.Gradient_algo['regulization'][:dimension]
	>>> J = Gradient_descent.Loss(Q,profile,target)

        :param Q: The n-ary vecteur representing the parameter set to calibrate of dimension n e.g for n = 5, "QT","AF", "AR", "B"and "C" have to be fitted
        :type Q: numpy::array
        :param profile: The obtained simulated temperature profile
        :type profile: numpy::array
        :param target: The measured target temperature profile
        :type target: numpy::array
        :return:  A scalar
        :rtype: float
        :raises: None

	'''
	'''
		
.. code-block:: python
	J = profile_target_L2(profile,target) + sum(regulization*Q**2) 
	'''
	return J


def unpackarraytostring(Q,space=' '):

	'''
	Takes an array as a first argument and return a string containing the same elements of the array seperated by the second argument e.g space.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Q = Gradient_descent.unpackarraytostring(Q,'_')

	Output::

    		`1_2_3`

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :param space: Seperation string
        :type space: string
        :return: string
        :rtype: string
        :raises: None

	'''
	'''
		
.. code-block:: python
	
	out=''
	for arg in Q:
		out += str(arg)+space
	'''
	return out


def modify_Q_in_f(Q):

	'''
	Prepares the parameters using `unpackarraytostring()` method in order to run them as a shell first argument using the method subprocess.call() as if we would run the following::
	
	 $ python update_module.py  Q

	The module `update_module` is described in the following section and has the main goal to update the abaqus user subroutine file with the new parameter set before running the Abaqus job.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Gradient_descent.modify_Q_in_f(Q)

	Output::

    		`Parameters are modified successfully!`

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :return: None
        :rtype: None
        :raises: None

	'''
	'''

.. code-block:: python
 
	if type(Q) != int:
		Q = unpackarraytostring(Q)
	Q = str(Q)
	subprocess.call('python '+update_path+' '+Q,shell=True)

	'''


'''
def test():
	#should include the error msg for number of parameters!
	modify_Q_in_f(np.array([1000.0,1.,1.,1.,1.]))
#test()

	'''

def plot_profile(Q):

	'''
	Prepares the parameters using `unpackarraytostring()` method in order to run them as a shell first argument using the method subprocess.call() as if we would run the following::
	
	 $ python plot_module.py  Q

	The module `plot_module` is described within the next sections and has the main goal to create a plot of the simulated temperature profiles for each iteration.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Gradient_descent.plot_profile(Q)

	Output::

    		`The profile graph is made successfully!`

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :return: None
        :rtype: None
        :raises: None

	'''
	'''

.. code-block:: python
 
	if type(Q) != int:
		Q = unpackarraytostring(Q,'_')
		Q = Q[:-1]
	Q = str(Q)
	subprocess.call('python '+Plot_path+' '+Q,shell=True)

	'''
	
def Abaqus(Q):


	'''
	Prepares the Abaqus user subroutine using `modify_Q_in_f()` method in order to run a shell command using the method subprocess.call() as if we would run the following::
	
	 $ python Abaqus_module.py

	The module `Abaqus_module` is described within the next sections and has the main goal to submit the Abaqus analysis on the cluster for each iteration.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Gradient_descent.Abaqus(Q)

	Output::

    		`The Abaqus job is lunched successfully!`

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :return: None
        :rtype: None
        :raises: None

	'''
	'''

.. code-block:: python

	modify_Q_in_f(Q)
	print('--------Abaqus job to submit--------')
	subprocess.call('python '+Abaqus_path,shell=True)
	time.sleep(2)
	'''

def Extract(Q):


	'''
	Calling the `Gradient_descent.plot_profile()` method after running the post processing phase using `Postprocessing_module` via a shell command using the method subprocess.call() as if we would run the following::
	
	 $ python Postprocessing_module.py

	The module `Postprocessing_module` is described within the next sections and has the main goal to extract the temperature profiles from the Abaqus model.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Gradient_descent.Extract(Q)

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :return: Temperature profiles
        :rtype: numpy::array
        :raises: None

	'''
	'''

.. code-block:: python
	subprocess.call('python '+Extract_path,shell=True)
	time.sleep(2)
	with open(report_file,'r') as f:
		Profiles = np.loadtxt(f,delimiter=",")
	f.close()
	plot_profile(Q)

	'''
	return Profiles

	
def get_profiles(Q):


	'''
	 Returns temperature profiles after calling the `Gradient_descent.Abaqus()` method followed by calling `Gradient_descent.Extract()` method.

	The methods are described above within this module.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Gradient_descent.get_profiles(Q)

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
        :return: Temperature profiles
        :rtype: numpy::array
        :raises: None

	'''
	'''

.. code-block:: python
	Abaqus(Q) #modify input files and generate odb
	#Extract the profile from the odb
	print('\n-----------------------------------------------------------')
	print('----------------------Extraction starts now----------------')
	print('-----------------------------------------------------------\n')
	profiles = Extract(Q)

	'''

	return profiles

	
def Loss_calc(Q1,target):

	'''
	Returns temperature profiles and the cost function after calling the `Gradient_descent.get_profiles()` method followed by calling `Gradient_descent.Loss()` method.

	The methods are described within this module.

	:Example:

        >>> Q = np.array([1,2,3])
	>>> Targets = Gradient_descent.Load_profile(targets_file,nb_col_profiles)
	>>> L,profile = Gradient_descent.Loss_calc(Q,Targets)

        :param Q: The n-ary vecteur representing the parameter set for instance
        :type Q: numpy::array
	:param target: The target temperature profile
        :type target: numpy::array
        :return: A tuple of temperature profiles and the cost function evaluation on the given parameter set
        :rtype: (float,numpy::array)
        :raises: None

	'''
	'''

.. code-block:: python

	print('\n--------------------------------------------------------------------')
	print('----------------------Calculation of Loss--------------------------')
	print('--------------------------------------------------------------------\n')
	#Calculate the initial Loss (of Q1)
	profile = get_profiles(Q1)
	loss = Loss(Q1,profile,target)

	'''
	return loss,profile

'''	
#loss_0 = Loss_init(Q_step1,Targets,1e-30)
#print('the initial loss is=',loss_0)
'''
	
def Gradient_Loss(L1,Q1,Q2,target):
	'''
	Returns the gradient of the cost function estimated two initial parameter set.
 	The partial derivatives are approximated by backward difference.

	Given the direction and the orthogonal basis:

	.. math::

		 h^i = (0,..,h_i,...,0) \, , e_{i \in (1,..,n)} 
	
	With:

	.. math::

		h_i =  <Q_2-Q_1 , e_i>_{R^n}

	We approximate the partial derivatives of J(x0,...,xi,...,xn) by backward difference as follows:

	.. math::

		\partial_{x_i} J = \\frac{J(Q_1 + h^i) - J(Q_1)}{h_i}

	   
	For more explanations one can check the following wikipedia pages: `here <https://en.wikipedia.org/wiki/Partial_derivative>`_ or `here <https://en.wikipedia.org/wiki/Derivative>`_.

	
	.. warning::

		When Q1 is equal to Q2 a default value of the gradient is set in the method. It can be changed in follwing line::
		
		>>> if Q1[i] == Q2[i]:
		>>>	grad[i]= 0.1

	:Example:

        >>> Q1 = np.array([1,2,3])
	>>> Q2 = np.array([1.1,2.1,3.1])
	>>> Targets = Gradient_descent.Load_profile(targets_file,nb_col_profiles)
	>>> L1,profile1 = Gradient_descent.Loss_calc(Q1,Targets)
	>>> grad = Gradient_Loss(L1,Q1,Q2,target)

	:param L1: The value of the cost function on the initial parameter set
        :type L1: numpy::array
        :param Q1: The n-ary vecteur representing the initial parameter set
        :type Q1: numpy::array
	:param Q2: The n-ary vecteur representing the neighbouring second guess parameter set
        :type Q2: numpy::array
	:param target: The target temperature profile
        :type target: numpy::array
        :return: The gradient of the cost function on the initial parameter set 
        :rtype: (float,numpy::array)
        :raises: None


	'''
	'''

.. code-block:: python

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
	
	grad[0]=grad[0]*gradient_scaling[0]
	if n>1:
		grad[1:]=grad[1:]/gradient_scaling[1:]
	print('The gradient is ',grad)
	print('The loss is ',loss)

	'''
	return grad
		

	
def save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,ok=True):
	'''
	Save the relevant output data to txt files eventually for each iteration.
		
	:Example:

        >>> save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,True)

	Output::

    		`The outputs data is saved successfully!`

        :param Q_list: A list of n-ary vecteur representing the parameter set for each iteration
        :type Q_list: list
        :param Loss_list: A list of the obtained value of the cost function for each iteration
        :type Loss_list: list
        :param Profiles_list: A list of the obtained simulated temperature profiles for each iteration
        :type Profiles_list: list
        :param step_list: A list of the found steps for each iteration
        :type step_list: list
        :param gradient_list: A list of the numerical gradients for each iteration
        :type gradient_list: list
        :param delta_list: A list of the difference between the new and the old value of the cost function for each iteration
        :type delta_list: list
        :param error_abs_list: A list of the L2 norm of the simulated and targetted temperature profiles for each iteration
	:type error_abs_list: list
        :param ok: A boolean indicating if the obtained parameter set and cost function value should be saved or not. Default is `True` for saving all. 
	:type ok: bool
        :return: None
        :rtype: None
        :raises: None

	'''
	'''

.. code-block:: python	

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
	'''
'''
#############################################################
################ Calculate initial gradient  ################
#############################################################
def grad_init():
	gradient_init = Gradient_Loss(Q_step1,Q_step2,Targets,regulization)
	print("\n")
	print('Initial gradient---------------------------------------> '+str(gradient_init))
	return gradient_init

######################################################################
################### Algorithm of gradient descent  ###################
######################################################################
'''
def Gradient_descent(tol,max_iter,max_iter_step,step,armijo_step):

	'''
	This method is an implementation of the Gradient descent which is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. The idea is to take repeated steps in the opposite direction of the approximate gradient. For more explanations one can check the following wikipedia pages: `here <https://en.wikipedia.org/wiki/Gradient_descent.>`_.

	:Example:

	>>> tol = cfg.Gradient_algo['tolerance']
	>>> max_iter = cfg.Gradient_algo['max_iter']
	>>> max_iter_step = cfg.Gradient_algo['max_iter_step']
	>>> step = cfg.Gradient_algo['step']
	>>> gradient_scaling = cfg.Gradient_algo['gradient_scaling'][:dimension]
	>>> regulization = cfg.Gradient_algo['regulization'][:dimension]
	>>> armijo_step = cfg.Gradient_algo['armijo_step']
	>>> iters,total_iter = Gradient_descent.Gradient_descent(tol,max_iter,max_iter_step,step,armijo_step)

	:param tol: This is indicating the stoping condition given by the residue
        :type tol: float
        :param max_iter: The maximum number of gradient descent iterations
        :type max_iter: integer
	:param max_iter_step: The maximum number of the backtracking line search iterations
        :type max_iter: integer
	:param step: The value of the step of the gradient descent convergeance
        :type step: float
	:param armijo_step: The coefficient of the backtracking line search 
        :type armijo_step: float
	:param gradient_scaling: The coefficients to scale the gradient magnitude
        :type gradient_scaling: numpy::array
	:param regulization: The coefficients related respectively to the parameter set in order to regulize the cost function
        :type regulization: numpy::array
        :return: a tuple (number  of iteration, total number of iterations including the backtracking line search iterations) 
        :rtype: (integer,interger)
        :raises: None


	'''
	'''

.. code-block:: python

	######################################################################
	#######################Initialization#################################
	######################################################################
	Q_list = [Q_step1,Q_step2]
	L1,profile1 = Loss_calc(Q_list[-2],Targets,regulization)
	L2,profile2 = Loss_calc(Q_list[-1],Targets,regulization)
	Loss_list = [L1,L2]
	Profiles_list = [profile1,profile2]
	gradient_list = []
	delta_list = []
	error_list = []
	error_abs_list = []
	step_list = []
	######################
	error = 1.			##
	error_abs = 1.		##
	iter = 0			##
	total_iter = 0		##
	######################
	while ((error_abs > tol) and (iter < max_iter)):
		
		print('-----------------------------------------------------')
		print('-----------Optimal Q in iteration '+str(iter)+' = '+str(Q_list[-1])+'-----------')
		print('-----------------------------------------------------\n')
		gradient =  Gradient_Loss(Loss_list[-2],Q_list[-2], Q_list[-1],Targets,regulization)
		print('----------------------',gradient)
		Q = np.minimum(np.maximum(np.array(Q_list[-1]) - step*gradient,minimum_par),maximum_par)
		print('-------gradient and step------',gradient,step)
		L_new,profiles = Loss_calc(Q,Targets,regulization)

		#step of descent direction
		count = 0
		Q_old = Q_list[-1]
		L_old = Loss_list[-1]
		delta = L_new - L_old 
		#################################################################
		##################Backtracking line search#######################
		#################################################################
		while ((count < max_iter_step)  and (delta >=0)):
			print('----------------------------------------------------------')
			print('-----Looking for descent direction for iteration '+str(count+1)+'------')
			print('----------------------------------------------------------\n')

			step /= armijo_step
			Q = np.minimum(np.maximum(np.array(Q_old) - step*gradient,minimum_par),maximum_par)
			L_new,profiles = Loss_calc(Q,Targets,regulization)
			delta = L_new - L_old
			count +=1
			
			#save
			delta_list.append(delta)
			step_list.append(step)
			Profiles_list.append(profiles)
			error_abs_list.append(error_abs)
			
			#save
			save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,False)
		
		iter += 1
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
		print("For iteration "+str(iter)+" we obtain Q = "+str( Q)+ " with a relative error of "+str(error))
		#save
		save_results(Q_list,Loss_list,Profiles_list,step_list,gradient_list,delta_list,error_list,error_abs_list,True)
	'''	
	return iter,total_iter
'''

print("\n \n-------------------------------------------------")
print("-------Starting gradient descent algorithm----------")
print("-------------------------------------------------\n \n")
##########################################################################
################### Algorithm of gradient descent  call###################
##########################################################################
######################################
tol = tol						######
max_iter = max_iter				######
max_iter_step = max_iter_step	######
step = step						######
armijo_step = armijo_step	    ######
##########################################################################
iter,total_iter = Gradient_descent(tol,max_iter,max_iter_step,step,armijo_step)  #####
##########################################################################
##########################################################################
print('--------------------------------------------------------------------')
print('----------------------------Done!-----------------------------------')
print('--------------------------------------------------------------------')

#time estimation
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations.\
\n The search for descent direction took " +str(iter)+" main iterations ")
with open(state_file,'a') as file:
	file.write('\n-----------------------------------------------------------------------------------------------\n')
	file.write("Elapsed time to converge "+str(elapsed_time)+" seconds in "+str(total_iter)+" total iterations including search for descent direction and " +str(iter)+" main iterations ")
	file.write('\n------------------------------------------------------------------------------------------------\n')
'''

