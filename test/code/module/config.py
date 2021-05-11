'''
This is where the tool settings are set. Those setting are structured in six groups:

* Input paths
* Output paths
* Preprocessing parameters 
* Abaqus inputs 
* Postprocessing parameters 
* Gradient descent algorithm settings 


Paths of input and ouput files
#####################################

.. comment import numpy as np

This part of the `config` module has to be ultered when the location of the simulation inputs are changed.

:Example:

.. code-block:: python

	input = {	"root":"//isi/w/elh/Gradient_descent_config",
			'odb_folder':'test',
			'code_folder': 'code',
			'results_folder':'results',
			'odb_file': 'test/SP013_X6CR.odb',
			'target_file':"Initial_data/target_ref.txt",
			'modules':{'update_module':'code/update_module.py',
				   'Postprocessing_module':'code/Postprocessing_module.py',
				   'Plot_module':'code/Plot_module.py',
				   'Abaqus_module':'code/Abaqus_module.py',
				   'extraction_module':'code/Extract_module.py',
				   }
	}



The same applies to the desired outputs destination in case the user would like to change the path.  


:Example:

.. code-block:: python

	output=  {'export_file':'T_t extraction.txt',
		  'state_file':'state.txt',
		  'Results':{'Loss_list':'results/Loss_list.txt',
			     'Profiles_list':'results/Profiles_list.txt',
			     'Q_list':'results/Q_list.txt',
			     'delta_list':'results/delta_list.txt',
			     'step_list':'results/step_list.txt',
			     'gradient_list':'results/gradient_list.txt',
			     'error_list':'results/error_list.txt',
			     'error_abs_list':'results/error_abs_list.txt',
					
					}				
	}

.. Note::
	
	In this Abacal project version the directories have the following structure:

	.. code-block:: python

		Abacal project	
		|-- Abacal tool (root)
		|	|-- code
		|	|   |-- config.py
		|	|   |-- Gradient_module.py
		|	|   |-- update_module.py
		|	|   |-- Abaqus_module.py
		|	|   |-- Postprocessing_module.py
		|	|   |-- Plot_module.py
		|	|-- test
		|	|   |-- SP013_X6CR.inp
		|	|   |-- SP013_X6CR.fem
		|	|   |-- SP013_X6CR.f
		|	|   |-- WSModell_Kinematic_new.ssc
		|	|-- resulsts
		|	|   |-- Q_list.txt
		|	|   |-- Loss_list
		|	|   |-- gradient_list
		|	|   |-- error_list
		|	|   |-- step_list
		|	|   |-- ...
		|	|-- Plots
		|	|   |-- For each iteration
		|	|--Readme.md
		|--Abacal tool doc
		|	|-- Abacal
		|	|   |-- config.py
		|	|   |-- Gradient_module.py
		|	|   |-- update_module.py
		|	|   |-- Abaqus_module.py
		|	|   |-- Postprocessing_module.py
		|	|   |-- Plot_module.py
		|	|   |-- img
		|	|       |-- ...
		|	|-- docs
		|	|   |-- build
		|	|   | 	|-- doctrees
		|	|   |	|   |-- ...
		|	|   |   |-- html
		|	|   |	|   |-- index.html
		|	|   |	|   |-- ...
		|	|   |-- source
		|	|   | 	|-- conf.py
		|	|   |   |-- _static
		|	|   |	|   |-- basic.css
		|	|   |   |-- ...
		|	|   |-- ...
		|       |--Readme.md
		|--Readme.md


Preprocessing settings
###########################

This part of the config file provide the tool with name of the parameters to be updated for each iteration. in order to modify the parameters automatically the script needs the position of the respective parameters in the user subroutine file e.g SP013_X6CR.f.

:Example:

.. code-block:: python

	pre_Abaqus = {"lines_parameters":np.array([25,34,36,41,43]),
		      "Parameters_name": {0: "QT",
					  1: "AF",
					  2: "AR",
					  3: "B",
					  4: "C",
					}
	}


* The given lines in the **`lines_parameters`** variable has to mention the line in the user subroutine - 1. 
* The variable **`Parameters_name`** can change according to the user subroutine heat source parameters.

Abaqus processing settings
###########################

Those setting are related to the welding process conditions. There are found and ultered when needed in the Abaqus input file e.g SP013_X6CR.inp.

:Example:

.. code-block:: python


	Abaqus = {"time":20,
		  "speed":5.,
	}
* **`time`** variable is indicating the time of the welding simulation
* **`speed`** variable is related to velocity of the moving heat source

Postprocessing settings
###########################

Once the simulation is done, the profiles can be extracted from the abaqus model e.g SP013_X6CR.odb via Abaqus viewer using the postprocessing module. the variabes below need to be set adequatly. 

:Example:

.. code-block:: python

	post_Abaqus = {'measurements_number':6,
		       "extracted_columns":77,
		       'path_x':np.array([1,2,3,4,5,6]),
		       'path_y':np.ones(6)*-4,
		       'path_z':np.ones(6)*5,
	}

* **measurements_number** is the number of the sensors providing several measurement postions on the model
* **extracted_columns** are the number of frames during which the temperature profile is extracted 
* **path_x/y/z** indicates the coordinates of the thermal sensors providing measured temperature

.. warning::

	The path cordinates have to be adapted to the tested model in Abaqus e.g x -> z , y -> x and z becomes y.

Gradient descent algorithm settings
####################################

This the key settings of the presented tool. Those values can be kept for the same model but have to be adapted to a new one.

.. code-block:: python

	Gradient_algo = {'Q_0': np.array([25000.0,1.5]),
			 'Q_1': np.array([26000.0,2.5]),
			 'Parameters_interval':{'max':np.array([50000,6,6,6,6]),
						'min':np.array([1000.,0.8,0.8,0.8,0.8]),
					},
			 'tolerance': 1.e-1,
			 'max_iter':10,
			 'max_iter_step':10,
			 'step':1.,
			 'armijo_step':2.3,
			 'regulization':np.array([1e-30,1e-30,1e-30,1e-30,1e-30]),
			 'gradient_scaling':np.array([900,5000,5000,5000,5000]),
					
	}
* **Q_0** and **Q_1** are the first and second initial guess allowing the two-point estimation of partial derivatives

* **tolerance** is indicating the stoping condition given by the residue

* **max_iter** is the maximum number of gradient descent iterations

* **max_iter_step** is the maximum number of the backtracking line search iterations

* **step** is the value of the step of the gradient descent convergeance

* **armijo_step** is the coefficient of the backtracking line search

* **gradient_scaling** are the coefficients to scale the gradient magnitude

* **regulization** are the coefficients related respectively to the parameter set in order to regulize the cost function


'''
