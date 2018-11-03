import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import csv
import math




class classes(object):
    """docstring for ClassName"""
    def __init__(self,data,mu,sigma,cov,label):
        self.data = data
        self.mu = mu
        self.sigma=sigma
        self.cov =cov
        self.label= label
def read_data():
    data = []
    # TODO: Read the file 'data1.csv' into the variable data.
    # data contains the training data together with labelled classes.
    data= pd.read_csv('data1.csv',header=None)
    return data


def read_test_data():
    test_data = []
    test_data_true = []
    # TODO: Read the file 'test_data.csv' and 'test_data_true.csv' into the variables test_data and test_data_true.
    # test_data contains the unlabelled test class.
    # test_data_true contains the actual classes of the test instances, which you will compare
    # against your predicted classes.
    test_data = pd.read_csv('test_data.csv',header=None)
    test_data = np.array(test_data)
    test_data_true = pd.read_csv('test_data_true.csv',header=None)
    test_data_true=np.array(test_data_true)
    return test_data, test_data_true


def multivariate_normal_gaussian(x, mu, sigma,cov):
    prob = 0
    firstPart= 1/(math.sqrt(2*math.pi)) 
    needed = x-mu
    secondPart =1/np.linalg.det(cov)
    firstPart = math.pow(firstPart,int(x.shape[0])) #1goz2
    secondPart = math.sqrt(abs(secondPart))   #2goz2
    thirdPart = np.matmul(np.transpose(needed),np.linalg.inv(cov))
    thirdPart = np.matmul(thirdPart, needed)
    thirdPart = -1/2  * thirdPart
    thirdPart = np.exp(thirdPart)
    prob = (1/(firstPart * secondPart)) * thirdPart
    # TODO: Implement the multivariate normal gaussian distribution with parameters mu and sigma.
    return prob


training_data = read_data()
test_data, test_data_true = read_test_data()
z=np.array(training_data) 
numClasses = np.unique(z[:,0])
print(numClasses.shape[0])

# TODO: Estimate the parameters of the Gaussian distributions of the given classes.
C=[]
for i in numClasses:
    class1_data = z[z[:,0]==i]
    class1_mu = np.mean(class1_data[:,1:],axis=0)
    class1_sigma = np.sqrt(np.var(class1_data[:,1:],axis=0))
    class1_cov = np.cov(class1_data[:,1],class1_data[:,2])
    C.append(classes(class1_data,class1_mu,class1_sigma,class1_cov,i))
print(C)
colors = ['r', 'g', 'b', 'c', 'y']
# TODO: Do a scatter plot for the data, where each class is coloured by the colour corresponding
# TODO: to its index in the colors array.
# Class 1 should be coloured in red, Class 2 should be coloured in green and Class 3 should be coloured in blue.
index=0
for i in C:
    plt.scatter(i.data[:,1],i.data[:,2],c=colors[index])
    index= index+1
plt.show()   
    
# TODO: Apply the Bayesian Classifier to predict the classes of the test points.
predicted_classes = np.zeros((test_data_true.shape[0],test_data_true.shape[1]))
index=0
for test in test_data:
    probArray=dict()
    for c in C:
        probArray[c.label]=multivariate_normal_gaussian(test,c.mu,c.sigma,c.cov)
    predicted_classes[index]=(max(probArray,key=probArray.get)) 
    index=index+1
# TODO: Compute the accuracy of the generated Bayesian classifier.
    
print(np.array(predicted_classes).shape)
print(test_data_true.shape)
S= sum(test_data_true==predicted_classes)
accuracy = S/test_data_true.shape[0]
print('Accuracy = ' + str(accuracy*100) + '%')


# TODO: Generate a 3D-plot for the generated distributions. x-axis and y-axis represent the features of the data, where
# TODO: z-axis represent the Gaussian probability at this point.
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 15, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)

classes = 2
# TODO: Change this according to the number of classes in the problem.
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        for c in C:
            Z[i, j] =Z[i, j]+ multivariate_normal_gaussian(np.array([x[i],y[j]]),c.mu,c.sigma,c.cov)
         
        # TODO: Fill in the matrix Z which will represent the probability distribution of every point.

# Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
