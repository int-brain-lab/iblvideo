import pandas   	

df=pandas.read_hdf('pupilDeepCut_resnet50_pupilsep20shuffle1_200000.h5')

#To get a list of the available points, type:

df.keys()
	
'''
For a particular bodypart, you can access the time series of the 'x', 'y' coordinate and detection certainty 'likelihood'
E.g. to get the x position of the 'pupil_top_r' point, type:
'''

s=df[(df.keys()[0][0], 'pupil_top_r', 'x')].values



	

