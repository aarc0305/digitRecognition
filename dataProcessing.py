import numpy as np
import csv
l=[]  
with open('train.csv') as file:  
	lines=csv.reader(file)
	for line in lines:
		l.append(line)
l.remove(l[0])
l=np.array(l)
label=l[:,0]
data=l[:,1:] 
print type(data)
print data.shape
