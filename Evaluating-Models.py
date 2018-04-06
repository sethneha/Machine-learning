import json, operator
import sys
from collections import Counter,  OrderedDict
import numpy as np
from sklearn import cluster, datasets , preprocessing, tree, linear_model
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
import random
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# reading data from json file
f = open('assess2_data.json') 
data = json.load(f)

#Question b: Writing script to find if there are duplicate name and printing them

seen = OrderedDict()
duplicate = OrderedDict()

for record in data:
	oname = record["NAME"] 
	if oname not in seen:
		seen[oname] = record
	else:		
		duplicate[oname] = record

for record in duplicate:
	print 'The duplicate name records are:',record


#Question c: Scrip to find missing values in data

count_missing_values=0 #defining counter
newdata = [] # creating dataset without enteries with missing values  

#Finding the rows with missing values and not writing them to new data set

for i in range(0,len(data)):
	isMissing = False
	for key in data[i].keys():
		if str(data[i][key]) == '-9999':
			count_missing_values = count_missing_values + 1
			isMissing = True
			break		
	if isMissing == False:
		newdata.append(data[i])
print '\nNumber of rows missing data in dataset are : ',count_missing_values

#Question d: Finding the values and frequency of the fields RFA_2F and RFA_2A and values of Wealth Index

sample=[]
for i in range(0,len(newdata)): # creating array of json data
	sample.append([int(newdata[i]['RFA_2F']),str(newdata[i]['RFA_2A']),int(newdata[i]['TARGET_B']),int(newdata[i]['LASTGIFT']),float(newdata[i]['AVGGIFT']),str(newdata[i]['PEPSTRFL']),int(newdata[i]['LASTDATE']),float(newdata[i]['WEALTH_INDEX']),int(newdata[i]['INCOME']),int(newdata[i]['FISTDATE']),str(newdata[i]['NAME'])])

sample=np.array(sample)

print '\nCount of Values of and frequency in field RFA_2F is :', Counter(sample[:,0]) 
print '\nCount of Values of and frequency in field RFA_2A is :', Counter(sample[:,1]) 
print '\nValues in field Wealth Index is: ', Counter(sample[:,7])
print '\nType of values in field Wealth Index is :', type(sample[:,7])


#Question e : Calculating the proprotion of the targeted consumers that responded
count_target_zero = 0 
count_target_one = 0

for i in range(0,sample.shape[0]):
	if  sample[i][2] == '0':
		count_target_zero = count_target_zero + 1
		
	if  sample[i][2] == '1':
		count_target_one = count_target_one + 1
		

total = count_target_zero +count_target_one
print '\nThe proportion of the targeted customers that responded is :' + str(float(count_target_one)/total * 100) + '%' 


# Creating the independent variables and dependent variable dataset 

indep_var = [] 

for i in range(0,sample.shape[0]):
	indep_var.append([float(sample[i][0]),(sample[i][1]),float(sample[i][3]),float(sample[i][4]),(sample[i][5]),float(sample[i][6]),float(sample[i][7]),float(sample[i][8]),float(sample[i][9])])

indep_var =np.array(indep_var)


depen_var = []
for i in range(0,sample.shape[0]):
	depen_var.append(int(sample[i][2]))

depen_var = np.array(depen_var)

#Normalizing data in independent variable dataset


for i in [0,2,3,5,6,7,8]:
	indep_var[:,i]=preprocessing.normalize([indep_var[:,i]], norm='max')

for i in [1,4]:
	le = preprocessing.LabelEncoder()
	le.fit(indep_var[:,i])
	indep_var[:,i]=le.transform(indep_var[:,i])

# Creating dendogram dataset	
dendro_data=[]
dendro_data =indep_var[:5000]

#Question f: Create a dendogram to show evident clusters by performing hierarchial clustering
Z = linkage(dendro_data, method='ward', metric='euclidean') # distance between clusters and metric
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
#creates a dendrogram hierarchial plot
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

#display dendogram
plt.axhline(y=12)
plt.show()

#Question g: Plot the distribution of WEALTH_INDEX for responders and non-responders on a graph to show similarities and differences

#Create dataset for responders and non responders 

Responder_Wealth = []
Non_Responder_Wealth = []

for i in range(0,sample.shape[0]):
	if sample[i][2] == "0":
		Non_Responder_Wealth.append(float(sample[i][7]))
	elif sample[i][2] =="1":
		Responder_Wealth.append(float(sample[i][7]))


# bins width ie,range of field WEALTH_INDEX 
bins = np.linspace(-10,50,20)
Non_Responder_Wealth

#plotting number of people in a bin 
plt.hist(Non_Responder_Wealth,bins,alpha=0.5,label='Non Responders')
plt.hist(Responder_Wealth,bins,alpha=0.5,label='Responders')
plt.legend()
plt.title('Histogram distribution of Responders and Non Responders')
plt.xlabel('Wealth Index bin')
plt.ylabel('Number of people in bin')
plt.show()

#plotting percentage of people in bins i.e. number of people of a segment/Total people in that segment
plt.hist(Non_Responder_Wealth,bins,alpha=0.5,label='Non Responders',normed=True)
plt.hist(Responder_Wealth,bins,alpha=0.5,label='Responders',normed=True)
plt.legend()
plt.title('Histogram Percent distribution of Responders and Non Responders')
plt.xlabel('Wealth Index bin')
plt.ylabel('Percent of particular segment of people in bin')
plt.show()

#Question h: Alphabetizing the records by the NAME Field
#Printing records from 1-10 and 20,000-20010
Alphabetized_Name_List= sorted(newdata, key=lambda k: k['NAME'])
for i in range(0,9):
	print Alphabetized_Name_List[i]['NAME']

for i in range(19999,20009):
	print Alphabetized_Name_List[i]['NAME']


#Question i: Alphabetizing the records by individual's last name
#Printing records from 1-10 and 20,000-20010
Alphabetized_Last_Name_List= sorted(newdata, key=lambda k:str.split(str( k['NAME']))[-1])
for i in range(0,9):
	print Alphabetized_Last_Name_List[i]['NAME']
for i in range(19999,20009):
	print Alphabetized_Last_Name_List[i]['NAME']


#Question j

# Creating a oversampled dataset with equal number of target responses of 0 and 1 as the proportion of responders is only 6.31% in the sample data

def oversample_data(indep_var, depen_var):
	training_data_X=[]
	training_data_Y=[]
	test_data_X=[]
	test_data_Y=[]
	resp_ctr = 0
	for i in range(len(indep_var)):		
		temp_x=[]
		if depen_var[i]==1:
			if resp_ctr%4==0:
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[i][j]))
				test_data_X.append(temp_x)
				test_data_Y.append(depen_var[i])
			else:
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[i][j]))
				training_data_X.append(temp_x)
				training_data_Y.append(depen_var[i])

			resp_ctr+=1
	non_resp_ctr = 0
	while non_resp_ctr <= resp_ctr:		
		temp_x=[]
		ptr = random.randint(0,len(indep_var)-1)
		if depen_var[ptr]==0:
			if non_resp_ctr%4 == 0:
				for j in range(len(indep_var[ptr])):
					temp_x.append(float(indep_var[ptr][j]))
				test_data_X.append(temp_x)
				test_data_Y.append(depen_var[ptr])
			else:
				for j in range(len(indep_var[i])):
					temp_x.append(float(indep_var[ptr][j]))
				training_data_X.append(temp_x)
				training_data_Y.append(depen_var[ptr])
			non_resp_ctr+=1

	return training_data_X,training_data_Y, test_data_X,test_data_Y

def plot_roc(test_data_Y, predicted_Y, caption):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i in range(len(test_data_Y)):
	    fpr[i], tpr[i], _ = roc_curve(test_data_Y, predicted_Y)
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(test_data_Y, predicted_Y)
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	plt.figure()
	lw = 2
	plt.plot(fpr[2], tpr[2], color='red',
		 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Q j. ROC for '+caption)
	plt.legend(loc="lower right")
	plt.show()

def fit_classifier(clf, training_data_X, training_data_Y, test_data_X):
	clf.fit(training_data_X, training_data_Y)
	predicted_Y=[]
	for i in range(len(test_data_X)):	
		predicted_Y.append(clf.predict([test_data_X[i]]))
	return predicted_Y

# Defining data sets before training models
training_data_X,training_data_Y, test_data_X,test_data_Y=oversample_data(indep_var, depen_var)
	
# Training Model 1 - Decision Tree
clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(max_depth=10)
predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
plot_roc(test_data_Y,predicted_Y,"Decision Tree")

# Training Model 2 - Neural Network
clf = MLPClassifier()
predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
plot_roc(test_data_Y,predicted_Y, "Neural Network")

# Training Model 3 - Logistic Regression
clf = linear_model.LogisticRegression()
predicted_Y=fit_classifier(clf, training_data_X, training_data_Y, test_data_X)
plot_roc(test_data_Y,predicted_Y, "Logistic Regression")
 	
		


 
