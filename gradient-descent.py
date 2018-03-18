from sklearn import datasets
import numpy as np 
import matplotlib.pyplot as plt

boston=datasets.load_boston() # get reference to boston data set

data=boston.data #get a reference to the input vector (called data) from the boston dataset
target= boston.target # get a refernce to the target vector (called target) from the boston dataset

trainingX = [] #starting empty training input vector
trainingY =[] # starting empty training target vector 
testX =[] # starting empty test  input vector
testY =[] # starting empty test target vector

# using loops to construct datasets from original dataset 
for i in range (0, data.shape[0]):
		#we want to put 10% into a testing set that is not use to to train the model
		
		if(i % 10 == 0): #put every tenth row into the test set
			testX.append(data[i,:])
			#the test set needs to be constructed from the corresponding target variables
			testY.append(target[i])
			
		else:
			#put into the training set
			trainingX.append(data[i,:])
			#the test set needs to be constructed from the corresponding target variables
			trainingY.append(target[i])
		

#in this "data refining" step we convert all the test and training vectors into numpy arrays for efficent processing...
trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
testX = np.array(testX)
testY = np.array(testY)

#normalizing the training dataset created above

for i in range(0,trainingX.shape[1]):
	minvalue = min(trainingX[:,i])
	maxvalue = max(trainingX[:,i])
	
	for j in range(0,trainingX.shape[0]):
		trainingX[j,i] = (trainingX[j,i] - minvalue)/float(maxvalue-minvalue)

#normalizing the test dataset created above

for i in range(0,testX.shape[1]):
	minvalue = min(testX[:,i])
	maxvalue = max(testX[:,i])
	
	for j in range(0,testX.shape[0]):
			testX[j,i] = (testX[j,i] - minvalue)/float(maxvalue-minvalue)

# defining the model for 13 features
def model(b,x):
	y = b[0]
	for i in range(0,trainingX.shape[1]):             
	  	y += b[i+1]*x[i]                   
  	return y 
#training the model
b = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] 

learning_rate = [0.00001,0.0001,0.001,0.01,0.1,1]

for learning in range(0,len(learning_rate)):
	rms_arr = []
	for epochs in range(0,10):
		for i in range(0,trainingX.shape[0]):
		  	error = model(b, trainingX[i]) - trainingY[i]
			b[0] = b[0] - learning_rate[learning]*error*1

			for j in range(0,trainingX.shape[1]): 
		  		b[j+1] = b[j+1] - learning_rate[learning]*error*trainingX[i][j]


		#creating prediction array of test set 
		predictions = []

		for i in range(0,testX.shape[0]):                                      
			  prediction = model(b,testX[i])
			  predictions.append(prediction)

		#calculating RMSE
		r= 0.0

		for i in range(0,testY.shape[0]):
			r += (testY[i] - predictions[i]) ** 2

		rms = (r/testY.shape[0]) ** (0.5)
		rms_arr.append(rms)

	#plotting the graph
	plt.plot(range(0,10),rms_arr) 
	plt.show()


	
	
	
 	
	

	








 







