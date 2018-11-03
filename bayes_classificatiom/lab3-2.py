import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
import csv
import math
import pandas as pd


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
    data= pd.read_csv('data2.csv',header=None)
    return data


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


data = read_data()
z=np.array(data) 
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



# TODO: Estimate the parameters of the Gaussian distributions of the given classes.


colors = ['r', 'g', 'b', 'c', 'y']
# TODO: Do a scatter plot for the data, where each class is coloured by the colour corresponding
# TODO: to its index in the colors array.
# Class 1 should be coloured in red, Class 2 should be coloured in green and Class 3 should be coloured in blue.

def find_decision_boundary(cov , mean1,mean2,probc1,probc2):
    diff = mean1-mean2
    w = 2 * np.matmul(np.linalg.inv(cov),diff)
    firstPart= np.matmul(np.transpose(mean1),np.linalg.inv(cov))
    firstPart=np.matmul(firstPart,mean1)
    secondPart= np.matmul(np.transpose(mean2),np.linalg.inv(cov))
    secondPart=np.matmul(secondPart,mean2)    
    needed = firstPart - secondPart
    needed = -1 * needed
    w0 = needed - 2 *math.log(probc2/probc1)
    #TODO: Find the coefficients of the decision boundary. Pass the required parameters to the function.
    return w, w0

# TODO: Generate a 3D-plot for the generated distributions. x-axis and y-axis represent the features of the data, where
# TODO: z-axis represent the Gaussian probability at this point.
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, y)
Z = np.zeros(X.shape)
Zplane,_ = np.meshgrid(np.linspace(0, 50, 300), y)

for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        for c in C:
            Z[i, j] =Z[i, j]+ multivariate_normal_gaussian(np.array([x[i],y[j]]),c.mu,c.sigma,c.cov)


#TODO: Call find_decision_boundary(..) to find the coefficients of the plane in the form W.T@X + W0 = 0
first= C.pop()
print("shit," ,first.cov)
second = C.pop()
print("shit," ,second.cov)
w,w0 = find_decision_boundary(second.cov,np.array(first.mu),np.array(second.mu),first.data.shape[0]/data.shape[0],second.data.shape[0]/data.shape[0])
X2 = np.linspace(-10, 10, 300)
Y2 = (-w0-w[0]*X2)/w[1] 
Xplane=0
Yplane=0
#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True, zorder=0.3)
ax.plot_surface(Xplane, Yplane, Zplane, cmap='plasma', linewidth=0, antialiased=True, zorder=0.8, alpha=0.9)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()

ax.plot_surface(X2, Y2, X2, cmap='viridis', linewidth=0, antialiased=True, zorder=0.3)
ax.plot_surface(Xplane, Yplane, Zplane, cmap='plasma', linewidth=0, antialiased=True, zorder=0.8, alpha=0.9)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()