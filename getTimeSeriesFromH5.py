import pandas   
from pylab import*

df=pandas.read_hdf('pupilDeepCut_resnet50_pupilsep20shuffle1_200000.h5')

#To get a list of the available points, type:

df.keys()
	
'''
For a particular bodypart, you can access the time series of the 'x', 'y' coordinate and detection certainty 'likelihood'
E.g. to get the x position of the 'pupil_top_r' point, type:
'''

s=df[(df.keys()[0][0], 'pupil_top_r', 'x')].values

'''
In practice it is important only to consider the coordinates where the network was certain, i.e. with a detection likelihood above 0.9
'''

part=pupil_top_r',

x_values=df[(df.keys()[0][0], part, 'x')].values
y_values=df[(df.keys()[0][0], part, 'y')].values
likelyhoods=df[(df.keys()[0][0], part, 'likelihood')].values

mx = ma.masked_where(likelyhoods<0.9, x_values)
x=ma.compressed(mx)
my = ma.masked_where(likelyhoods<0.9, y_values)
y=ma.compressed(my)	

