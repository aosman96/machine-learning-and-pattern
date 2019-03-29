import cv2
import numpy as np
import skimage.io as io
from scipy.signal import convolve2d

from skimage import morphology, color
import matplotlib.pyplot as plt

# For skeletonization
from skimage.morphology import skeletonize
from skimage import data
from skimage.util import invert

# import imutils
import argparse
from skimage.measure import compare_ssim
from matplotlib.patches import Circle


from tkinter import *
from tkinter.ttk import Progressbar
from tkinter import ttk

import scipy.signal as signal
import time

import os
import glob

import pandas as pd 
################################### Imports End ###################################

def readLabelsFile(filePath):
    data = pd.read_csv(filePath, comment='#', delimiter=' ', header=None)
    data = np.array(data)
    return data

def getLabel(labels,string):
    return labels[labels[:,0] == string][0][1]



# TODO : Change for loop to victorized. 1 equation
def splitImg(test_image,windowsize_r,windowsize_c):
    windowsize_r = int(windowsize_r)
    windowsize_c = int(windowsize_c)
    # windowList = []
    topLeftIndexList=[]
    for r in range(0, test_image.shape[0] - windowsize_r + 1, windowsize_r):
        for c in range(0, test_image.shape[1] - windowsize_c + 1, windowsize_c):
            # window = test_image[r:r + windowsize_r, c:c + windowsize_c]
            # windowList.append(window)
            topLeftIndexList.append([r,c])
    return np.array(topLeftIndexList) #, np.array(windowList)


def showImage(img,title='',resize=True,scale=1/4):
    if resize:
        imS = cv2.resize(img, (int(img.shape[0] * scale), int(img.shape[1] * scale)))  # Resize image
        cv2.imshow(title, imS)  # Show image
    else:
        cv2.imshow(title, img)  # Show image


def fixWindowList(topLeftIndexList, threshedImg,windowR, windowC, TestCase=0):
    newIndexList = []
    TestClasses= []
    for topLeftIndex in topLeftIndexList:
        # If contains at least 1 black pixel
        windowValues = threshedImg[topLeftIndex[0]:topLeftIndex[0]+windowR, topLeftIndex[1]:topLeftIndex[1]+windowC] // 255
        if windowValues.shape[0]==windowR and windowValues.shape[1]==windowC and  np.isin(0, windowValues ) :
            occurenceOfZerosInColumns = np.sum(windowValues, axis=0) // windowValues.shape[0]
            newColumnIndex = np.argwhere(occurenceOfZerosInColumns==0)[0][0]

            # if len(newIndexList)==0 or abs(topLeftIndex[1]+newColumnIndex - newIndexList[-1][1]) >= windowR:
            if topLeftIndex[0] + windowR < threshedImg.shape[1] and topLeftIndex[1] + windowC < threshedImg.shape[0]:
                if(TestCase==1):
                    TestWindow=threshedImg[topLeftIndex[0]:topLeftIndex[0]+windowR, topLeftIndex[1]+newColumnIndex:topLeftIndex[1]+windowC+newColumnIndex]
                    TestWindow=np.array(TestWindow)
                    TestWindow=TestWindow.reshape(windowR*windowC,1)
                    TestClasses.append(TestWindow)
                newIndexList.append([topLeftIndex[0],topLeftIndex[1]+newColumnIndex])
    return np.array(newIndexList),TestClasses


def drawRectangles(topLeftIndexList, windowR, windowC, threshed):
    for topLeftIndex in topLeftIndexList:
        # Take care rectangle takes points bel 3aks
        cv2.rectangle(threshed, (topLeftIndex[1], topLeftIndex[0]), (topLeftIndex[1]+windowC, topLeftIndex[0]+windowR), 125, 1)


def simmilarity(window1, window2):      #Window 1 is 1 window. Window2 is list of windows
    window1copy = np.divide(window1,255)
    window2copy = np.divide(window2,255)
    n11=[]
    n10=[]
    n01 = []
    n00 = []
    for window in window2copy:
        # window = window[:window1copy.shape[0],:window1copy.shape[1]]  #No need for now
        n11.append(np.sum(window1copy[window == 1] == 1))
        n00.append(np.sum(window1copy[window == 0] == 0))
        n10.append(np.sum(window1copy[window == 0] == 1))
        n01.append(np.sum(window1copy[window == 1] == 0))
        # n01 = np.sum(window1copy[window2copy == 1] == 0)
    n11=np.array(n11); n10=np.array(n10); n01=np.array(n01); n00=np.array(n00);
    return (n11*n00 - n10*n01)/np.sqrt((n11+n10)*(n01+n00)*(n11+n01)*(n10+n00))


def prepImage(img):
    # Get edge image
    edges = cv2.Canny(img, 80, 120)

    # Get horizontal lines using Hough Transform with theta = 90
    lines = cv2.HoughLinesP(edges, 3, np.pi / 2, 2, None, 200, 1)

    # Sort lines wrt y ascendingly
    linesSortedAsc = np.sort(lines[:, 0, 1])

    # Get y of second line from top
    y1 = linesSortedAsc[(linesSortedAsc - linesSortedAsc[0]) < 600][-1]
    # Get y of first line from bottom
    y2 = linesSortedAsc[(linesSortedAsc[-1] - linesSortedAsc) < 400][0]

    # Crop image using y1 & y2
    imgCropped = img[y1+10:y2, :]

    return imgCropped


def window(indexList,windowR,windowC,threshed):
    return threshed[indexList[:][0]:indexList[:][0] + windowR,indexList[:][1]:indexList[:][1] + windowC]


def getAvgWindows(topLeftIndexList, threshedImg,windowR, windowC):
    topLeftIndexList = np.array(topLeftIndexList)
    avgWindow = np.array(window(topLeftIndexList[0],windowR,windowC,threshedImg) / 255)
    for topLeftIndex in topLeftIndexList[1:]:
        avgWindow = (avgWindow+ window(topLeftIndex,windowR,windowC,threshedImg)/255)
    avgWindow = avgWindow / topLeftIndexList.shape[0]
    avgWindow[avgWindow>=0.5]=255
    avgWindow[avgWindow < 0.5] = 0
    return avgWindow


def clustring(topLeftIndexList, threshedImg,windowR, windowC,similarityThreshold=0.2, minimumWindowPerCluster=5):
    classes =[]
    means=[]
    classes.append([topLeftIndexList[0]])
    means.append(window(topLeftIndexList[0],windowR,windowC,threshedImg))
    for currentWindowIndex in topLeftIndexList[1:]:
        similartiyValues = ((simmilarity(window(currentWindowIndex,windowR,windowC,threshedImg),means)))
        maxIndex = np.argmax(similartiyValues)
        if(similartiyValues[maxIndex] <similarityThreshold): # new cluster
            means.append(window(currentWindowIndex,windowR,windowC,threshedImg))
            classes.append([currentWindowIndex])
        else:
            classes[maxIndex].append(currentWindowIndex)
            means[maxIndex] = getAvgWindows(classes[maxIndex], threshedImg,windowR, windowC)

    # Remove classes with less than n windows per Class
    classesNew = []
    newMeans = []
    indexCounter = 0
    for class1 in classes:
        if np.array(class1).shape[0] > minimumWindowPerCluster:
            classesNew.append(class1)
            newMeans.append(means[indexCounter])
            indexCounter+=1

    # TODO: Representatives !

    return classesNew, newMeans



def getRepresentives(classes,windowR,windowC,threshed,means):
    rep=[]
    probablity=[]
    classesUpdated=[]
    covUpdated = []
    #first step: calculating sim matrix for each class 
    index = 0
    for classi in classes:
        probablity.append(len(classi))
        windowOfClasses = []
        check = True
        for topLeftIndex in classi:
            windowCreated = window(topLeftIndex,windowR,windowC,threshed)
            windowOfClasses.append(windowCreated)
            windowCreated = np.array(windowCreated)
            if check:
                check=False
                classesUpdated.append([windowCreated.reshape(windowC*windowR,1)])
            else:
                classesUpdated[index].append(windowCreated.reshape(windowC*windowR,1))
        similarityPerClass =[]
        for windowImage in windowOfClasses:
            similarityPerClass.append(np.sum(np.array(simmilarity(windowImage,windowOfClasses))))
        
        meanscomp = np.array(means[index])
        covUpdated.append(computeCov(classesUpdated[index],meanscomp.reshape(windowC*windowR,1)))
        
        # print(len(windowOfClasses))
        #print("pass1")
        index+=1
        representitve =np.array(windowOfClasses[int(np.argmax(np.array(similarityPerClass)))])
        rep.append(representitve.reshape(windowC*windowR,1)) #-1 because it is calcuates similarity with it self also
        #print("pass2")
    probablity = np.array(probablity)    
    probablity = np.divide(probablity, np.sum(probablity))
    return rep,probablity,covUpdated

##################################IDENTIFICATION#########################################

def computePosterior(D, x):
#    print(D[0][0])
#    print("______________________________________________________________________")
#    print(D[1][0])
#    print("______________________________________________________________________")
    print(D[2])
    Ds = np.array(D[2])
    print(Ds.shape())
#    print("______________________________________________________________________")
#    print(D[3][0])
#    print("______________________________________________________________________")
    #print(x.shape)
    posteriors =[]
    for index in range(len(D)):
        #print(D[2][index])
        print(D[2][index])
        dd= np.array(D[2][index])
        print(dd)
        print(np.linalg.det(D[2][index]))
        variable1 = ( -0.5 * np.log(np.linalg.det(D[2][index])) )
        #print(variable1)
        variable2 = ( 0.5*( np.transpose(x-D[0][index]) @ np.linalg.inv(D[2][index]) @ (x-D[0][index]) ) )
        #print(variable2)
        variable3 = ( np.log(D[1][index]) )
       # print(variable3)
        posteriors.append( variable1-variable2[0][0]  +variable3 )
    posteriors = np.array(posteriors)
    #print(posteriors)
    return posteriors
def computeCov(X, mu):
    xcopy = np.array(X)
    xcopy=xcopy/255
 #   print(xcopy.shape)
    muc = np.array(mu)
    muc = muc /255
#    print(X.shape)
#    print("______________")
 #   print(muc.shape)
    
    indexCounter=0
    for i in xcopy:
       xcopy[indexCounter]=i- muc
       indexCounter+=1
    cov = np.transpose(xcopy[:,:,0]) @ (xcopy[:,:,0])
#    print(cov.shape)
    return cov

def computeMaxPosteriors(D, T):
    maxPosts = []
    for i in T:
        maxPosts.append(np.max(computePosterior(D, i), axis=0))
    return np.array(maxPosts)

# T is a numpy array of Windows inside the document we wish to identify
# D is a numpy array of arrays with D[i] = [ Rpr(Ci), P(Ci), Cov(i), WriterID ]
def computeSimilarity(T, D):
    coeff = (1/T.shape[0])
    summationTerm = computeMaxPosteriors(D, T)
    summation = np.sum(summationTerm)
    return coeff*summation

# T is a numpy array of Windows inside the document we wish to identify
# D is a numpy array of documents, each document contains a numpy array of arrays = [[ Rpr(Ci), P(Ci), Cov(i), WriterID ]
def identifyWriter(T, D):                                                           
    similarities = []
    for i in D:
        similarities.append(computeSimilarity(T, i))
    similarities = np.array(similarities)
    print(similarities)
    writerID = D[np.argmax(similarities)][3]
    return writerID



################################### Main Start ###################################

def Proccessing(path,debugMode,windowR,windowC,similarityThreshold,minimumWindowPerCluster,Test=0):
        ##### Reading Original Image ######
    TestWindows=[]
    image = cv2.imread(path)
    img = prepImage(image)
    img = img[:,145:2445]
    if debugMode:
        showImage(img,'orignal')
    
    ##### Thresholding ########
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    threshed= 255-threshed
    if debugMode:
        showImage(threshed,'threshed')
    
    ##### Window Indeces ########
    topLeftIndexList = splitImg(threshed,windowR,windowC)
    topLeftIndexList,TestWindows = fixWindowList(topLeftIndexList,threshed,windowR, windowC,Test)
    
    ##### Draw Rectangles ########
    if debugMode:
        threshedWithRects = threshed.copy()
        drawRectangles(topLeftIndexList, windowR, windowC, threshedWithRects)
        showImage(threshedWithRects,'threshedWithRects')
        cv2.imwrite('result.png',threshedWithRects)
        
    # FIXME: Example 1: similarity range
    if(Test==0):
        classes, means = clustring(topLeftIndexList, threshed,windowR, windowC,similarityThreshold,minimumWindowPerCluster)   
    # FIXME: Example 2: print clusters data
        representives,probablities,covUpdated = getRepresentives(classes,windowR,windowC,threshed,means)
    else:
        if debugMode:
            cv2.waitKey(0)
        return TestWindows,str(path.split('\\')[-1])    
    
    if debugMode:
        print('*******')
        print(len(classes))
        print('---')
        for classu in classes:
            print(len(classu))
        print('*******')
    
        print('+++++')
        print(len(classesUpdated))
        print('---')
        for classu in classesUpdated:
            print(len(classu))
        print('+++++')
    
    if debugMode:
        cv2.waitKey(0)
    return representives,probablities,covUpdated,str(path.split('\\')[1])













##### Options ######
debugMode = 0;
similarityThreshold = 0.2
minimumWindowPerCluster = 27
windowR = 13
windowC = 13
img_dir = "./documents"
labels = np.array(readLabelsFile('./PatternDataset/forms.txt'))

testPath = "./test/a01-003x.png"

representivesDocument = []
probablitiesDocument = []
classesUpdatedDocument = []
filenameDocument = []
Documents=[]
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)









#training process
for f1 in files: 
    representives,probablities,covUpdated,name = Proccessing(f1,debugMode,windowR,windowC,similarityThreshold,minimumWindowPerCluster)
    #representivesDocument.append(representives)
    #probablitiesDocument.append(probablities)
    #classesUpdatedDocument.append(classesUpdated)
    #filenameDocument.append(name)    
    Documents.append([np.array(representives),np.array(probablities),np.array(covUpdated),name])



#reading Test Document 
Twindows,Tname = Proccessing(testPath,debugMode,windowR,windowC,similarityThreshold,minimumWindowPerCluster,1)
Documents = np.array(Documents)
print(Documents.shape)
Twindows = np.array(Twindows) 

Name = identifyWriter(Twindows,Documents)

print(Tname,Name)
print('Done')





































###### Reading Original Image ######
#image = cv2.imread('./PatternDataset/a01-003.png')
#img = prepImage(image)
#img = img[:,145:2445]
#showImage(img,'orignal')
#
###### Thresholding ########
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
#threshed= 255-threshed
#showImage(threshed,'threshed')
#
###### Window Indeces ########
#windowR = 13
#windowC = 13
#topLeftIndexList = splitImg(threshed,windowR,windowC)
#topLeftIndexList = fixWindowList(topLeftIndexList,threshed,windowR, windowC)
#
###### Draw Rectangles ########
#threshedWithRects = threshed.copy()
#drawRectangles(topLeftIndexList, windowR, windowC, threshedWithRects)
#showImage(threshedWithRects,'threshedWithRects')
#cv2.imwrite('result.png',threshedWithRects)
#
## FIXME: Example 1: similarity range
#
#classes, means = clustring(topLeftIndexList, threshed,windowR, windowC,similarityThreshold,minimumWindowPerCluster)
#
## FIXME: Example 2: print clusters data
#
#representives,probablities,classesUpdated = getRepresentives(classes,windowR,windowC,threshed)
#print(len(classesUpdated[0][0]))
##print(len(representives))
##counter =0 
##for rep in representives:
##    showImage(rep,str(counter),True,8)
##    counter+=1
#
##
#
#
#cv2.waitKey(0)
#
#
#
#
#
