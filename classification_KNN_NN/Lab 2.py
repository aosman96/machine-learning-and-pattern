import cv2 
import numpy as np
import math
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot(x, y, z,  title='', xlabel='', ylabel='', zlabel='',color_style_str='', label_str='', figure=None, axis=None):
    if figure is None:
        fig = plt.figure()
    else:
        fig = figure
    ax = axis
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)    
        #TODO: Add title, x_label, y_label, z_label to ax.

    #TODO: Scatter plot of data points with coordinates (x, y, z) with the corresponding color and label.
    ax.scatter(x,y,z,c=color_style_str)
    handles, labels = ax.get_legend_handles_labels()

    unique = list(set(labels))
    handles = [handles[labels.index(u)] for u in unique]
    labels = [labels[labels.index(u)] for u in unique]

    ax.legend(handles, labels)

def preprocess(img):
    #  the given image img.
    # TODO: Convert the image to grayscale.
    # Hint: Check the function cvtColor in opencv.
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
 
    # TODO: Blur the image to remove the noise (apply a low pass filter 3x3).
    # Hint: Check the function blur in opencv.
    # Hint: Pass the kernel size as an array (3, 3)
    blur = cv2.blur(gray_img,(3,3))
    
    # TODO: Apply a threshold between 50 and 255 on the blurred image. The pixels having values less than 50 will be considered 0, and 255 otherwise.
    # Hint: Check the function threshold in opencv.
    # Hint: Use the type cv2.THRESH_BINARY with the type parameter.
    # Hint: This function has two return parameters. You can ignore the first one, and the second is the binary image.
    # Example: _, thresholded_img = cv2.threshold(....)
    ret,thresholded_img = cv2.threshold(blur,50,255,cv2.THRESH_BINARY)
    return thresholded_img

def findContourArea(img):
    #This function finds the contours of a given image and returns it in the variable contours.
    #This function will not work correctly unless you preprocess the image properly as indicated.
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    #TODO: Find the contour area of the given image (img)
    area = cv2.contourArea(contours[1])

    return area, contours

def findBoundingRectangleArea(img, contours):
    #This function tries to fit the minimum bounding rectangle for the given contours.

    bounding_rectangle = 0
    #TODO: Find the minimum bounding rectangle that can fit the given contours.
    #Hint: Check the function boundingRect in opencv
    x,y,w,h = cv2.boundingRect(contours[1])
    
    #TODO (Optional): You can uncomment the following command(s) to show or display the bounded rectangle.
   # bounding_rectangle = cv2.rectangle(img.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow('Image_Rec', bounding_rectangle)
    #cv2.waitKey(0)

    #TODO: Find the area of the bounding rectangle
    area = w * h
    return area, bounding_rectangle

def findBoundingCircleArea(img, contours):
    #This function tries to fit the minimum bounding circle for the given contours.
    bounding_circle = 0
    #TODO: Find the minimum enclosing circle that can fit the given contours.
    #Hint: Check the function minEnclosingCircle in opencv
    ((x, y), radius) = cv2.minEnclosingCircle(contours[1])
    #TODO (Optional): You can uncomment the following command(s) to show or display the bounded circle.
    #unding_circle = cv2.circle(img.copy(), (int(x),int(y)), int(radius), (0, 255, 0), 2)
    #cv2.imshow('Image_Circle', bounding_circle)
    #cv2.waitKey(0)

    # TODO: Find the area of the bounding circle
    area = math.pi * radius * radius
    return area, bounding_circle

def findBoundingTriangleArea(img, contours):

    #This function tries to fit the minimum bounding triangle for the given contours.
    bounding_triangle = 0
    # TODO: Find the minimum enclosing triangle that can fit the given contours.
    # Hint: Check the function minEnclosingTriangle in opencv and place its output in the variable x
    x = cv2.minEnclosingTriangle(contours[1])
    
    #TODO (Optional): You can uncomment the following command(s) to show or display the bounded triangle.
    #bounding_triangle = cv2.polylines(img.copy(), np.int32([x[1]]), True, (0, 255, 0), 2)
   # cv2.imshow('Image_Triangle', bounding_triangle)
   # cv2.waitKey(0)
    print(x[0])
    # TODO: Find the area of the bounding circle
    area = x[0]
    return area, bounding_triangle

def calculateDistance(x1, x2):

    #TODO: Calculate the Euclidean distance between the two vectors x1 and x2.
    distance = 0
    return distance

def MinimumDistanceClassifier(test_point, training_points):

    #TODO: Implement the minumum distance classifier. You have to classify the test_point whether it belongs to class 1, 2 or 3.
    
    triangle = training_points[:,0] == 3
    traingle_features =training_points[triangle,1:]
    rectangle = training_points[:,0] == 1
    rectangle_features =training_points[rectangle,1:]
    circle = training_points[:,0] == 2
    circle_features =training_points[circle,1:]
    triangle_mean = np.mean(traingle_features,axis=0)
    rect_mean = np.mean(rectangle_features,axis=0)
    circle_mean = np.mean(circle_features,axis=0)
    dist1 = np.linalg.norm(test_point[1:]-rect_mean)
    dist2 = np.linalg.norm(test_point[1:]-circle_mean)
    dist3 = np.linalg.norm(test_point[1:]-triangle_mean)
    if (dist1 == min(dist1,dist2,dist3)):
        classification = 1
    elif(dist2 == min(dist1,dist2,dist3)):
        classification = 2
    else:
        classification =3 
        
    return classification

def NearestNeighbor(test_point, all_points):

    #TODO: Implement the Nearest Neighbour classifier. You have to classify the test_point whether it belongs to class 1, 2 or 3.
    data_min = all_points[0,:]
    index =0 
    for i in range(all_points.shape[0]):
        if(np.linalg.norm(all_points[i,1:]-test_point[1:]) < np.linalg.norm(test_point[1:]-data_min[1:])):
            index = i
            data_min = all_points[i,0:]  
    classification = data_min[0]
    return classification

def KNN(test_point, all_points, k):

    #TODO: Implement the K-Nearest Neighbour classifier. You have to classify the test_point whether it belongs to class 1, 2 or 3.
    array = []
    index= -1
    data_min = all_points[0,:]
    for j in range(k):
        for i in range(all_points.shape[0]):
            if(np.linalg.norm(all_points[i,1:]-test_point[1:]) < np.linalg.norm(test_point[1:]-data_min[1:]) ):
                data_min = all_points[i,:]
                index=i
        print(data_min)   
        np.delete(all_points,index,0)        
        array.append( data_min[0])
        data_min = all_points[0,:]
        index=-1

    sum1=0
    sum2=0
    sum3=0    
    for x in array:
        if (x==1):
            sum1=sum1+1
        elif(x==2):
            sum2=sum2+1
        else:
            sum3=sum3+1
    if(sum1==max(sum1,sum2,sum3)):
        classification = 1
    elif(sum2==max(sum1,sum2,sum3)):
        classification = 2
    else:
        classification = 3
    
    return classification

def get_class_from_file_name(file_name):
    return file_name.split("test\\")[1].split(".")[0]


def get_class_name(class_number):
    classes = ["", "Rectangle", "Circle", "Triangle"]
    return classes[int(class_number)]


def extract_features(img, class_number=None):
    #Given an image img, extract the following features.
    #1. The ratio between the figure area and the minimum enclosing rectangle.
    #2. The ratio between the figure area and the minimum enclosing circle.
    #3. The ratio between the figure area and the minimum enclosing triangle.

    area, contours = findContourArea(img)
    area1,_ = findBoundingRectangleArea(img, contours)
    area2,_ = findBoundingCircleArea(img, contours)
    area3,_ = findBoundingTriangleArea(img, contours)

    features = [class_number, area/area1 ,area/area2 , area/area3]
    #TODO: Extract the features and append the class_number (if given) in the begininning of each feature vector.
    return features

training_data = []
training_data_rec = []
training_data_circle = []
training_data_tri = []

for filename in glob.glob('images/rectangle/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 1)
    training_data.append(img_features)
    training_data_rec.append(img_features)

for filename in glob.glob('images/circle/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 2)
    training_data.append(img_features)
    training_data_circle.append(img_features)

for filename in glob.glob('images/triangle/*.png'):
    img = cv2.imread(filename)
    img = preprocess(img)
    img_features = extract_features(img, 3)
    training_data.append(img_features)
    training_data_tri.append(img_features)

training_data = np.asarray(training_data)
training_data_rec = np.asarray(training_data_rec)
training_data_circle = np.asarray(training_data_circle)
training_data_tri = np.asarray(training_data_tri)

# Visualization of features
fig = plt.figure()
ax = fig.add_subplot('111', projection='3d')

plot(training_data_rec[:, 1], training_data_rec[:, 2], training_data_rec[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='r', label_str = "Rectangle",
     figure=fig, axis=ax)

plot(training_data_circle[:, 1], training_data_circle[:, 2], training_data_circle[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='b', label_str = "Circle",
     figure=fig, axis=ax)

plot(training_data_tri[:, 1], training_data_tri[:, 2], training_data_tri[:, 3], title='Training Data',
     xlabel='Feature Rec.', ylabel='Feature Circle', zlabel='Feature Tri.', color_style_str='g', label_str = "Triangle",
     figure=fig, axis=ax)

plt.show()

# Don't modify this.
true_values = [3, 1, 1, 3, 3, 1, 1, 2, 3, 2]
index = 0
sum1=0
sum2=0
sum3=0
for filename in glob.glob('test/*.png'):
    #Read each image in the test directory, preprocess it and extract its features.
    img_original = cv2.imread(filename)
    img = preprocess(img_original)
    test_point = extract_features(img)
    print("Actual class :", get_class_name(true_values[index]))
    print("---------------------------------------")
    index += 1

    min_dist_prediction = MinimumDistanceClassifier(test_point, training_data)
    nn_prediction = NearestNeighbor(test_point, training_data)
    knn_prediction = KNN(test_point, training_data, 3  )

    print("Minimum Distance Classifier Prediction   :", get_class_name(min_dist_prediction))
    print("Nearest Neighbour Prediction             :", get_class_name(nn_prediction))
    print("K-Nearest Neighbours Prediction          :", get_class_name(knn_prediction))
    print("===========================================================================")
    
    if(min_dist_prediction ==  true_values[index-1]):
        sum1=sum1+ 1
    if(knn_prediction == true_values[index-1]):
        sum3=sum3+ 1
    if(nn_prediction ==  true_values[index-1 ]):
        sum2=sum2+ 1        

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img_original)
    cv2.waitKey(0) # this means to go to the next image press any key
    cv2.destroyAllWindows()

#TODO: Calculate the accuracy of the three classifiers on the test set. You may need to add code in the previous for loop.
accuracy_min_distance = sum1/index
accuracy_nn = sum2/index
accuracy_knn = sum3/index

print("Minimum Distance Classifier Accuracy: ", accuracy_min_distance *100, "%")
print("Nearest Neighbour Classifier Acccuracy: ", accuracy_nn *100, "%")
print("K-Nearest Neighbour Classifier Accuracy: ", accuracy_knn *100 , "%")