import numpy as np
np.random.seed(1337)  # for reproducibility
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop
import struct
from scipy.special import expit
import sys
import csv
def toInt(array):  
	array=np.mat(array)  
	m,n=np.shape(array)  
	newArray=np.zeros((m,n))  
	for i in range(m):	 
		for j in range(n):	 
			newArray[i,j]=int(array[i,j])  
	return newArray	 
l=[]
with open('train.csv') as file:
        lines=csv.reader(file)
        for line in lines:
                l.append(line)
l.remove(l[0])
l=np.array(l)
label=l[:,0]
data=l[:,1:]
data = toInt(data)
label = toInt(label)
#print type(data)
#print data.shape
X_train = data.reshape(data.shape[0], -1)/255.
y_train = np_utils.to_categorical(label, num_classes=10)
print X_train.shape
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)            
testarr=[]
with open('test.csv') as file: 
        lines=csv.reader(file)  
        for line in lines:
                testarr.append(line) #28001*784  
testarr.remove(testarr[0]) 
testnpArr=np.array(testarr) 
testnpArr = toInt(testnpArr)
print type(testnpArr)
results = model.predict(testnpArr)
print results
print type(results)
outputresult = []
for result in results:
	max = 0
	index = 0
	for i in range(10):
		if result[i] > max:
			max = result[i]
			index = i
	outputresult.append(index)
print outputresult
	
with open ('result.csv', mode='w') as write_file:
	writer = csv.writer(write_file)
	writer.writerow(["ImageId","Label"])
	for i in range(len(outputresult)):
		writer.writerow([i+1,outputresult[i]])

