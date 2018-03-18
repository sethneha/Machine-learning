import json
import random
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster, datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# reading data from json file
f = open('titanic.json','r')
d = json.load(f) 
 

# create data arrays from the read json file for - age , fare ,Companions count = combined Sibling/Spouse Parent/Child count , embarked location , sex amd marker for clusters

x=[] # initializing data array to be created from json file

for i in range (0,len(d)):
	x.append([str(d[i]['Age']),float(d[i]['Fare']),int(d[i]['SiblingsAndSpouses'])+int(d[i]['ParentsAndChildren']),str(d[i]['Embarked']),str(d[i]['Sex']),int(d[i]['Survived']),0])

# find missing values and replacing for Age

miss=0
sum=0
count=0

for i in range (0,len(x)): 
	if x[i][0] == "":
		miss = miss + 1
		count = count + 1
	else:
		x[i][0] = float(x[i][0])		
		sum = sum + x[i][0]
		count = count + 1

age_avg = float(sum)/count

for i in range (0,len(x)):	
	if x[i][0] == "": 
		x[i][0] = age_avg


# Replacing missing values in variable Embarked by the most common value

countC = 0 
countQ = 0 
countS = 0 

for i in range (0,len(x)):
	if x[i][3] == "Q":
		countQ = countQ + 1
	elif x[i][3] == "S":
		countS = countS + 1
	elif x[i][3] == "C":
		countC = countC + 1

maximum = max(countQ, countS, countC)

for i in range (0,len(x)):
	if x[i][3] == "":
		x[i][3] = maximum


# adding numeric values for text in dataset

for i in range (0,len(x)):
	
	#Embarked
	if x[i][3] == "S":
		x[i][3] = 0
	elif x[i][3] == "C":
		x[i][3] = 0.5
	elif x[i][3] == "Q":
		x[i][3] = 1

	#Sex
	if x[i][4] == "male":
		x[i][4] = 0
	else:
		x[i][4] = 1

#normalizing the data

x = np.array(x)

for i in range(0,3):
	minvalue = min(x[:,i])
	maxvalue = max(x[:,i])
	
	for j in range(0,len(x)):
			x[j,i] = (x[j,i] - minvalue)/float(maxvalue-minvalue)

# deleting  fare variable
x = np.delete(x,np.s_[1:2],1)

# deleting Embarked variable
x = np.delete(x,np.s_[2:3],1)

#perform hierarchial clustering
Z = linkage(x[:,:3], method='ward', metric='euclidean') # distance between clusters and metric
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('clusters')
plt.ylabel('distance')

#creates a dendrogram hierarchial plot
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)

plt.axhline(y=5, c='k') # creating the threshold for the clusters
 
#display dendrogram
plt.show()

# randomly intialize the clusters
initial_c1 = np.random.random(3)
initial_c2 = np.random.random(3) 

# 2 kmeans clustering - clustering.py plot_cluster_iris.py

def distance(a,b):
	return np.linalg.norm(a-b)


fig = plt.figure()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=134)
ax.scatter(x[:,0],x[:, 1], x[:, 2], c='m')
ax.scatter(initial_c1[0],initial_c1[1],initial_c1[2],c= 'r',marker = "*", s=500)
ax.scatter(initial_c2[0],initial_c2[1],initial_c2[2],c= 'b',marker = "*", s=500)
ax.set_xlabel('Age')
ax.set_ylabel('Companion')
ax.set_zlabel('Sex')
plt.savefig('InitialCluster.png')
plt.close()

# reassign Clusters

for iterations in range(10):
	for i in range(x.shape[0]):
		if distance(x[i,:3],initial_c1) < distance(x[i,:3],initial_c2):
			x[i][4] = 1
		else:
			x[i][4] = 2

	# Calculating Cluster centroids 
	Cluster_1 = []
	Cluster_2 = []
	for i in range(len(x)):
		if x[i][4] == 1:		
			Cluster_1.append(x[i])
		else:
			Cluster_2.append(x[i])
	new_c1 = np.mean(Cluster_1,axis=0)
	new_c1 = new_c1[:3]
	new_c2 = np.mean(Cluster_2,axis=0)
	new_c2 = new_c2[:3]
	 	
	Cluster_1 = np.array(Cluster_1)
	Cluster_2 = np.array(Cluster_2)

	fig = plt.figure()
	ax =  Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=134)
	ax.scatter(Cluster_1[:,0],Cluster_1[:, 1], Cluster_1[:, 2], c='r')
	ax.scatter(Cluster_2[:,0],Cluster_2[:, 1], Cluster_2[:, 2], c='b')
	
	ax.scatter(new_c1[0],new_c1[1],new_c1[2],c= 'r',marker = "*", s=500)
	ax.scatter(new_c2[0],new_c2[1],new_c2[2],c= 'b',marker = "*", s=500)
	ax.set_xlabel('Age')
	ax.set_ylabel('Companion')	
	ax.set_zlabel('Sex')
	plt.savefig('After %s iterations' %(iterations + 1))
	plt.close()
	
	if np.all(new_c1 == initial_c1) and np.all(new_c2 == initial_c2):
		stop = True
		break
	initial_c1 = new_c1
	initial_c2 = new_c2

# Plotting the Survivors and the Dead

Survivors = []
Dead = []
for row in x:
	if row[3] == 1:
		Survivors.append(row)
	else:
		Dead.append(row)

Survivors = np.array(Survivors)
Dead = np.array(Dead)

fig = plt.figure()
ax =  Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=150)
ax.scatter(Survivors[:,0], Survivors[:, 1], Survivors[:, 2], c='g')
ax.scatter(Dead[:,0], Dead[:, 1], Dead[:, 2], c='r')
ax.scatter(new_c1[0],new_c1[1],new_c1[2],c= 'k',marker = "*", s=300)
ax.scatter(new_c2[0],new_c2[1],new_c2[2],c= 'b',marker = "*", s=300)
ax.set_xlabel('Age')
ax.set_ylabel('Companiions')
ax.set_zlabel('Sex')
plt.show()
