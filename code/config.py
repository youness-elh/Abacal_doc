import numpy as np
directories = {
				"main":"//isi/w/elh/Gradient_descent_config",
				'odb_folder':'test',
				'code_folder': 'code',
				'results_folder':'results',
				'odb_file': 'test/SP013_X6CR.odb',
				'target_file':"Initial_data/target_ref.txt",
				'modules':{
						'update_module':'code/update_module.py',
						'Postprocessing_module':'code/Postprocessing_module.py',
						'Plot_module':'code/Plot_module.py',
						'Abaqus_module':'code/Abaqus_module.py',
						'extraction_module':'code/Extract_module.py',
				
				
				}
}

pre_Abaqus = {
				"lines_parameters":np.array([25,34,36,41,43]),
				"Parameters_name": {
						0: "QT",
						1: "AF",
						2: "AR",
						3: "B",
						4: "C",
				}
}

Abaqus = {     
				"time":20,
				"speed":5.,
}

post_Abaqus = {
				'measurements_number':6,
				"extracted_columns":77,
				'path_x':np.array([1,2,3,4,5,6]),
				'path_y':np.ones(6)*-4,
				'path_z':np.ones(6)*5,
}

Gradient_algo = {
				'Q_0': np.array([1000.0,1.,1.]),
				'Q_1': np.array([10000.0,2.,2.]),
}


output=  {
				'export_file':'T_t extraction.txt',
				'state_file':'state.txt',
				
}
