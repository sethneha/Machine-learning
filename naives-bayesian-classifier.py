import csv
import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter, defaultdict

# script to import csv file into a database - part b of question
def create_database(x):
	inputfile = x
	conn = sqlite3.connect('database.db')
	c = conn.cursor()
	csv_data = csv.reader(file(inputfile))

	c.execute('CREATE TABLE IF NOT EXISTS DatabaseTable(obs INTEGER, y INTEGER, x1 INTEGER,x2 INTEGER,x3 INTEGER,x4 INTEGER,x5 INTEGER)')
	
	c.execute('SELECT count(*) from DatabaseTable')
	lengthOfDatabase = c.fetchone()
	if lengthOfDatabase[0] == 0:	
		for row in csv_data:
			c.execute('INSERT INTO DatabaseTable VALUES(?,?,?,?,?,?,?)',tuple(row))

	c.execute('SELECT * from DatabaseTable')		
	h = c.fetchall()
	
	conn.commit()
	conn.close()
	return h

h = create_database('Flying_Fitness.csv')


lol = [list(elem) for elem in h]

# Part C of question - Creating function to open a database and use its data to train and plot a ROC curve for the classifier 

def roc(lol):

	# Creating array of data 
	data=[]
	 

	for i in range(1,len(lol)):
		j=lol[i][2:7] # dropping observations column and target variable
		data.append(j)

	data = np.array(data)

	# Creating array of Target variable
	target=[]
	CountZeroTarget=0
	CountOneTarget=0

	for i in range(1,len(lol)):
		j=lol[i][1]
		if j == 1:
			CountOneTarget += 1		
		else:
			CountZeroTarget += 1 
		target.append(j)

	target = np.array(target)

	# Defining function to calculate probablity from occurrences
	def occurrences(target):
	    no_of_examples = len(target)
	    prob = dict(Counter(target))
	    for key in prob.keys():
		prob[key] = prob[key] / float(no_of_examples)
	    return prob


	#Calculating the Conditional Probabilities for the variables
	classes     = np.unique(target)
	rows, cols  = np.shape(data)
	likelihoods = {} #initializing the dictionary

	for cls in classes:
	    
	    likelihoods[cls] = defaultdict(list)

	for cls in classes:
	    #taking samples of only 1 class at a time
	    row_indices = np.where(target == cls)[0]
	    subset      = data[row_indices, :]
	    r, c        = np.shape(subset)
	    for j in range(0, c):
		likelihoods[int(cls)][j] += list(subset[:,j])

	for cls in classes:
	    for j in range(0, c):
		likelihoods[int(cls)][j] = occurrences(likelihoods[int(cls)][j])

	# Calculating bayesian probabilities for each sample in database

	Probability_table = []
	class_probabilities = occurrences(target)

	for i in range(0,len(data)):
	    result ={} 	
	    for cls in classes:
	    	class_probability = class_probabilities[cls]
		for j in range(0,len(data[i])):
		    relative_feature_values = likelihoods[cls][j]
		    if data[i][j] in relative_feature_values.keys():
			class_probability *= relative_feature_values[data[i][j]]
		    else:
			class_probability *= 0
		    result[cls] = class_probability
	    Probability_table.append(result)


	# Creating datalist for computing the ROC 
	zeros=[]
	ones=[]


	for i in Probability_table:
	    ones.append(i.get(1))


	# Creating ROC data
	thresholdlist = list(ones)

	thresholdlist.insert(0,0)

	thresholdlist.sort()

	TruePositiveRate = 0
	FalsePositiveRate =0

	TruePositiveRateList = []
	FalsePositiveRateList = []
	

	for index in range(0,len(thresholdlist)):
		TruePositiveCount = 0
		FalsePositiveCount = 0
		for inputValueIndex in range(0,len(ones)):
			if ones[inputValueIndex] >= thresholdlist[index]:
				if target[inputValueIndex] == 1:
			
					TruePositiveCount = TruePositiveCount + 1
				else:
					FalsePositiveCount = FalsePositiveCount + 1
	
		TruePositiveRate = float(TruePositiveCount)/CountOneTarget
		FalsePositiveRate = float(FalsePositiveCount)/CountZeroTarget
		
		TruePositiveRateList.append(TruePositiveRate)
		FalsePositiveRateList.append(FalsePositiveRate)
		


	# Creating ROC graph
	plt.plot(FalsePositiveRateList, TruePositiveRateList,color='red')
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Curve")

	plt.gca().set_xlim([-0.01,1.01])
	plt.gca().set_ylim([-0.01,1.01])
	plt.plot(plt.gca().get_xlim(),plt.gca().get_ylim(),color = 'green')
	plt.show()

# Plotting the Roc curve for database lol
roc(lol)











    
	
