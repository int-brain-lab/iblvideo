from pylab import *
import pandas   
ion()	

d=pandas.read_hdf('pupilDeepCut_resnet50_pupilsep20shuffle1_200000.h5')

#E.g. you can save that as an excel sheet to see its structure:
d.to_excel('pupil_tracking.xlsx')	
	
# Then you can save them in a dictionary or whatever by getting the values as an array:
MyArray=d.values
	
#And plot: e.g. shape(d.values), (17967, 12)
for i in range(12):
 plot(d.values[:,i])
