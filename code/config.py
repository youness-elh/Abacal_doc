import numpy as np
directories = {
				"main":"//isi/w/elh/Gradient_descent",
}

pre_Abaqus = {
				"lines_parameters":np.array([25,34,36,41,43]),
				"Parameters_name": {
						0: "QT",
						1: "AF",
						2: "AR",
						3: "B",
						4: "C"
				}
}

Abaqus = {
				"time":20,
				"speed":5.,
}

post_Abaqus = {
				"extracted_collumns":77,
}