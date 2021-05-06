from abaqus import *
from odbAccess import * 
from abaqusConstants import * 
#import displayGroupOdbToolset as dgo
from itertools import count
import visualization
import numpy as np
import xyPlot
import math
import sys
import os
import re
import shutil #deleting a directory
from numpy import savetxt
#import subprocess 
import time
import os


os.chdir('../code')
import config as cfg

def print_contour(path_destination):
	session.printToFile(fileName=path_destination, format=PNG, 
			canvasObjects=(session.viewports['Viewport: 1'], ))

def Extract_profiles(limit=None):
	start_time_extraction = time.time()
	###########################################################
	################## Parameters to set ######################
	###########################################################
	directory = cfg.directories["main"]
	odb_file = cfg.directories["odb_file"]
	stpath = os.path.join(directory,odb_file)
	export_file = cfg.output["export_file"]
	report_file = os.path.join(directory,export_file) #exporting data to a created text file
	
	state_file = cfg.output["state_file"]
	comments_file = os.path.join(directory,state_file)
	######################################################################
	############ opening the odb and creating a (y,x) viewport  ##########
	######################################################################

	odb = session.openOdb(name='odb',path=stpath, readOnly=True) #opening the ODB file
	session.viewports['Viewport: 1'].setValues(displayedObject=odb) 
	session.viewports['Viewport: 1'].view.setValues(session.views['Front']) #set the viw to the front view
	session.viewports['Viewport: 1'].view.rotate(xAngle=-90,yAngle=0,zAngle=0) #to have(y,x) plan
	session.viewports['Viewport: 1'].view.zoom(zoomFactor= 1.5,mode=ABSOLUTE,drawImmediately=True) #fitview and zoom
	#print_contour(path_destination)
	################
	## todo ########
	################
	#better zoom and translate

	########################################################################
	############################# Add path #################################
	########################################################################
	Stp = 0# len(odb.steps.values())-1
	Frame = len(odb.steps.values()[0].frames) if limit is None else limit
	nb = len(odb.steps.values())-1
	with open(comments_file,'a') as file:
		file.write("\n We will extract data from the step number " +str(nb)+" for all its "+str(Frame)+" frames \n")
			
	print("--------------------------------------------------------------------------------------------------------")
	print("We will extract data from the step number", len(odb.steps.values())-1," for all its ",Frame, " frames")
	print("--------------------------------------------------------------------------------------------------------")
	
	nb = cfg.post_Abaqus['measurements_number']
	nodes = np.empty([nb,3])
				
	nodes[:,0] = cfg.post_Abaqus['path_x']
	nodes[:,1] = cfg.post_Abaqus['path_y']
	nodes[:,2] = cfg.post_Abaqus['path_z']

	print( "\n nodes 0 = \n",nodes)
		
	npath = session.Path(name="Path-grid 0 ",type=POINT_LIST,expression=nodes)

	output_all_frames = np.zeros([nb,3+Frame])
	output_all_frames[:,0:3] = nodes[:,:]

	for Frme in range(Frame):
		Frme_1 = Frme+1
		print(" ---------------Frame number ",Frme_1, " out of ",Frame , " frames-----------------")
		with open(comments_file,'a') as file:
			#subprocess.call(['echo' ,"-----------------------------------------" ],stdout=file)
			#subprocess.call(['echo' ," ---------------Frame number ",'$Frme_1', " out of ",'$Frame' , " frames-----------------" ],stdout=file)
			#file.write("---------------------------------------------------------")
			if Frme_1 % 10 == 0:
				file.write(" \n---------------Frame number "+str(Frme_1)+" out of "+str(Frame)+" frames-----------------\n")
		
		xy_data = session.XYDataFromPath(path=npath, name='xy-Data', includeIntersections=False, 
				shape=UNDEFORMED,pathStyle=PATH_POINTS,labelType= X_COORDINATE,
				step=Stp,frame=Frme,variable=('NT11',NODAL))
		output = []
		for sequence in xy_data:
			output.append(sequence)
		output = np.array(output)
		
		output_all_frames[:,3+Frme] = output[:,1]
		
	end_time_extraction = time.time()
	elapsed_extraction_time = end_time_extraction - start_time_extraction
	with open(report_file,'w') as f:
		np.savetxt(f, output_all_frames,delimiter=",")#,header = "start of new extraction",footer = " elapsed time for extraction is "+str(elapsed_extraction_time)+" seconds")			
	f.close()
	print("T-t extraction done successfully!! and saved in ",  report_file)
	with open(comments_file,'a') as file:
		file.write("\n Elapsed time for extraction is "+str(elapsed_extraction_time)+" seconds \n")
		file.write("---------------------------------extraction done successfully!----------------------------------------- \n")
		file.write("---------------------------------end now-----------------------------------------")

if __name__ == "__main__":
	extrected_columns = cfg.post_Abaqus['extracted_columns']
	Extract_profiles(extrected_columns)

