from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

# storing the m_depth list with values 
max_nodes=[2,4,8,16,32,64,128,256]
max_depth=[0,2,4,8,16]

#reading file to determine number of rows and columns in file to intitiate vectors
file = open("training_set.csv","r")
read = file.readlines()
nrows = len(read)
fields = read[0].split(',')
ncolumns = len(fields)

#initiate a dataset with no of rows and one less column number in csv file
#since id in the csv file is not needed we create the list of one column less than dimension on of csv file

dtree = np.zeros((nrows,ncolumns - 1))
target = np.zeros(nrows)

counterEarlyLate = [0,0] # to store count of early adopters and late adopters


trainingX = [] #starting empty training input vector
trainingY = [] # starting empty training target vector 
testX = [] # starting empty test input vector
testY = [] # starting empty test target vector

#define list of pruned training and test vectors
trainingX_pruned = []
trainingY_pruned = []
testX_pruned = []
testY_pruned = []
 
#adding numeric values for text in dataset
file = open("training_set.csv","r")
read = file.readlines()

for j in range(0,len(read)):
	data = read[j].strip()
	fields = data.split(',')

	for i in range(1,len(fields)):

		#Gender
		if fields[i] == "F":
			dtree[j][i-1] = 1
		elif fields[i] == "M":
			dtree[j][i-1] = 0

		#Martial Status
		if fields[i] == "Single":
			dtree[j][i-1] = 0
		elif fields[i] == "Married":
			dtree[j][i-1] = 1

		#Type of Use
		if fields[i] == "Low":
			dtree[j][i-1] = 0
		elif fields[i] == "Medium":
			dtree[j][i-1] = 1
		elif fields[i] == "Heavy":
			dtree[j][i-1] = 2
		elif fields[i] == "PrePaid":
			dtree[j][i-1] = 3

		#Automatic/Non-Automatic
		if fields[i] == "Automatic":
			dtree[j][i-1] = 0
		elif fields[i] == "Non-Automatic":
			dtree[j][i-1] = 1

		#Contract/Non-Contract
		if fields[i] == "No Contract":
			dtree[j][i-1] = 0
		elif fields[i] == "12 Months":
			dtree[j][i-1] = 1
		elif fields[i] == "24 Months":
			dtree[j][i-1] = 2
		elif fields[i] == "36 Months":
			dtree[j][i-1] = 3

		#Yes / No Encoding
		if fields[i] == "N":
			dtree[j][i-1] = 0
		elif fields[i] == "Y":
			dtree[j][i-1] = 1

		#adding values to target vector
		if (fields[i] == "Early" or fields[i] == "Very Early"):
			dtree[j][i-1] = 1
			counterEarlyLate[0] += 1
	
		elif (fields[i] == "Late" or fields[i] == "Very Late"):
			dtree[j][i-1] = 2
			counterEarlyLate[1] += 1

		#Age
		dtree[j][1] = fields[2] # after analyzing the third column
	
	
	target[j] = dtree[j,8]	

# Splitting the input dataset into training and test 
for i in range(0,dtree.shape[0]):
	new_row = dtree[i][0:8] # read from 0 to 7 to avoid reading target variable
	if (i % 10 == 0):
		testX.append(new_row)
		testY.append(target[i])
	else:	
		trainingX.append(new_row)
		trainingY.append(target[i])

# Converting list to numpay arrays for analysis
trainingX = np.array(trainingX)
trainingY = np.array(trainingY)
testX = np.array(testX)
testY = np.array(testY)

# Creating pruned data vectors 

prunedArraySize = 2 * min(counterEarlyLate)
if (counterEarlyLate[1] < counterEarlyLate[0]):
	prunedtempdataset = sorted(dtree,key=lambda x: x[8],reverse = True)
else:
	prunedtempdataset = sorted(dtree,key=lambda x: x[8])

prunedtempdataset = np.array(prunedtempdataset)
np.random.shuffle(prunedtempdataset)

pruneddtree = np.zeros((2*min(counterEarlyLate),8))
prunedtarget = np.zeros(min(counterEarlyLate))

pruneddtree = prunedtempdataset[0:prunedArraySize,0:8]
prunedtarget = prunedtempdataset[0:prunedArraySize,8]

# Splitting Pruned data set into 90% training and 10 % test data set 
for i in range(0,pruneddtree.shape[0]):
	new_row = pruneddtree[i][:] # read from 0 to 7 to avoid reading target variable
	if (i % 10 == 0):
		testX_pruned.append(new_row)
		testY_pruned.append(target[i])
	else:	
		trainingX_pruned.append(new_row)
		trainingY_pruned.append(target[i])

#Converting new created lists into numpy arrays
trainingX_pruned = np.array(trainingX_pruned)
trainingY_pruned = np.array(trainingY_pruned)
testX_pruned = np.array(testX_pruned)
testY_pruned = np.array(testY_pruned)

# Creating DecisionTree

for i in  range(0,len(max_depth)):
	accuracy_list = []
	for j in range(0,len(max_nodes)):
		if(max_depth[i] == 0):
			clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_nodes[j])
			clf.fit(trainingX,trainingY)
		else:
			clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_nodes[j],max_depth = max_depth[i])
			clf.fit(trainingX,trainingY)

		correct = 0
		incorrect = 0
		pred = clf.predict(testX)

		for index in range(0,pred.shape[0]):
			if pred[index] == testY[index]:		
				correct = correct + 1
			else:
				incorrect = incorrect + 1
	
		accuracy = float(correct) / float(correct+incorrect)

		accuracy_list.append(accuracy)

	#plotting the graph
	plt.plot(max_nodes, accuracy_list,color='red')
	plt.xticks(range(0,max_nodes[7],25))
	plt.xlabel("Number of Nodes")
	plt.ylabel("Accuracy")
	if (max_depth[i] == 0):
		plt.title("Normal Model - Depth : None")
	else: 
		plt.title("%s \n Depth : %.1f " % ("Normal Model",float(max_depth[i])))
	plt.show()


# Creating pruned Desicion Tree

for i in  range(0,len(max_depth)):
	accuracy_list = []
	for j in range(0,len(max_nodes)):
		if(max_depth[i] == 0):
			clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_nodes[j])
			clf.fit(trainingX_pruned,trainingY_pruned)
		else:
			clf = tree.DecisionTreeClassifier(max_leaf_nodes = max_nodes[j],max_depth = max_depth[i])
			clf.fit(trainingX_pruned,trainingY_pruned)

		correct = 0
		incorrect = 0
		pred = clf.predict(testX_pruned)

		for index in range(0,pred.shape[0]):
			if pred[index] == testY_pruned[index]:		
				correct = correct + 1
			else:
				incorrect = incorrect + 1
	
		accuracy = float(correct) / float(correct+incorrect)

		accuracy_list.append(accuracy)

	#plotting the graph
	plt.plot(max_nodes, accuracy_list,color='red')
	plt.xticks(range(0,max_nodes[7],25))
	plt.xlabel("Number of Nodes")
	plt.ylabel("Accuracy")
	if (max_depth[i] == 0):
		plt.title("Pruned Model - Depth : None")
	else: 
		plt.title("%s \n Depth : %.1f " % ("Pruned Model",float(max_depth[i])))
	plt.show()




 

