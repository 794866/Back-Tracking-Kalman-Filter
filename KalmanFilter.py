# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 03:28:45 2020

@author: Norman
"""
import cv2
import glob
import numpy as np
from plot_ellipse import plot_ellipse

# INITIALIZATION
# --------------------------------------------------------------
# Initialize your parameters, HOG detector, Kalman filter, etc
delta_t = 1/15
A = np.matrix([[1., 0., delta_t, 0.],
    [0., 1., 0., delta_t],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.]])

C = np.matrix([[1., 0., 0., 0.],
     [0., 1., 0., 0.]])

sigma_u = 50.
sigma_v = 15.

Q = np.matrix([[(sigma_u*delta_t)**2, 0., delta_t, 0.],
    [0., (sigma_v*delta_t)**2, 0., delta_t],
    [delta_t, 0, (sigma_u**2), 0],
    [0., delta_t, 0., (sigma_v**2)]])

errX = 10.
errY = 40.


R = np.matrix([[errX**2, 0.],
     [0, errY**2]])

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load dataset
# --------------------------------------------------------------
# Change the path to the dataset
files = glob.glob('C:/Users/Norman/Documents/SIS/p2/Data/TestData/2012-04-02_120351/RectGrabber/imgrect_000*_c0.pgm')


# We assume that the filesnames are ordered alphabetically/numerically
for filename in sorted(files):
    # We load the current image
    img = cv2.imread(filename)
    cv2.resize(img, (640, 480), fx=0.1, fy=0.1)
    #PROGRAM PEOPLE DETECTOR
    # ------------------------------------------------
    # YOUR CODE HERE 
    # ------------------------------------------------
    lines, score = hog.detectMultiScale(img)
    myBoxes = [[x, y, x + w, y + h] for (x, y, w, h) in lines]
    if(np.size(score,0)>1):
        for (xA, yA, xB, yB) in myBoxes:
            img = cv2.rectangle(img, (xA, yA), (xB, yB),(0, 255, 0), 2)
        cv2.imshow("My First Kalman Filter", img)
        break
    
mu_t = np.matrix([[lines[0,0]+lines[0,2]/2], [lines[0,1]+lines[0,3]/2], [0.], [0.]])
sigma_t = np.matrix([[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.],[0.,0.,0.,0.]])
I = np.matrix(np.identity(4))

for filename in sorted(files):
    #myPrediction
    mu_tm1 = np.matrix(A*mu_t) #Vector de medias en el instante t+1
    sigma_tm1 = np.matrix(A*sigma_t*np.transpose(A) + Q) #Matriz de varianzas y Covarianzas
    
    img = cv2.imread(filename)
    img = cv2.resize(img, (1000, 600))
    lines, score = hog.detectMultiScale(img)
    
    # Compute your Kalman filter prediction and update
    # if needed.
    # ------------------------------------------------
    # YOUR CODE HERE
    # ------------------------------------------------
    myBoxes = [[x, y, x + w, y + h] for (x, y, w, h) in lines]
    if(np.size(score,0)>1):
        mu_t = np.matrix([[lines[0,0]+lines[0,2]/2], [lines[0,1]+lines[0,3]/2], [0.], [0.]])
        I = np.matrix(np.identity(4))
        #What i'm observing
        y_t = C*mu_t
        #residue
        r_t = np.matrix(y_t - (C * mu_tm1))
        #residual covariance
        C_tras = np.matrix(np.transpose(C))
        s_t = np.matrix(C*sigma_tm1*C_tras + R)
        #Kalman gain
        k_t = np.matrix((sigma_tm1 * C_tras)*np.linalg.inv(s_t))
        
        #UPDATE
        mu_t = np.matrix(mu_tm1 + k_t * r_t)
        sigma_t = np.matrix(np.identity(4) - k_t*C)
        '''
        if(np.abs(r_t[0,0])<np.sqrt(s_t[0,0]) and np.abs(r_t[1,0])<np.sqrt(s_t[1,1])):
            #UPDATE
            mu_t = np.matrix(mu_tm1 + k_t * r_t)
            sigma_t = np.matrix(np.identity(4) - k_t*C)
        else:
            mu_t = mu_tm1
            sigma_t = sigma_tm1
        '''
        for (xA, yA, xB, yB) in myBoxes:
            img = cv2.rectangle(img, (xA, yA), (xB, yB),(0, 255, 0), 2)
        cv2.imshow("My First Kalman Filter", img)    
    else:
        mu_t = mu_tm1
        sigma_t = sigma_tm1
    #PredictionEllipse        
    ellipsePred = np.matrix([[sigma_tm1[0,0],sigma_tm1[0,1]],[sigma_tm1[1,0], sigma_tm1[1,1]]])
    mu_ellipsePred = [mu_tm1[0,0], mu_tm1[1,0]]
    #UpdateElipse
    ellipseUpdate = np.matrix([[sigma_t[0,0],sigma_t[0,1]],[sigma_t[1,0], sigma_t[1,1]]])
    mu_ellipseUpdate = [mu_t[0,0], mu_t[1,0]]
    
    b11 = (int(mu_tm1[0,0])+50,int(mu_tm1[1,0]+50))
    b12 = (int(mu_tm1[0,0])-50,int(mu_tm1[1,0])-50)
    
    a11 = (int(mu_t[0,0])-80,int(mu_t[1,0])-80)
    a12 = (int(mu_t[0,0])+80,int(mu_t[1,0])+80)
    
    #prediccionDraw    
    img = cv2.rectangle(img, b11, b12, [0,0,255],2) #red
    #updateDraw
    img = cv2.rectangle(img, a11, a12, [255,0,0],3) #blue
    
    #plot ellipse
    plot_ellipse(img,mu_ellipsePred,ellipsePred,(255,0,0))
    plot_ellipse(img,mu_ellipseUpdate,ellipseUpdate,(0,0,255))
    # Show the final image
        #cv2.imshow("My First Kalman Filter", img)
    cv2.imshow("My First Kalman Filter", img)
    # We show the image for 10 ms
    # and stop the loop if any key is pressed
    k = cv2.waitKey(10)
    if k != -1:
        break