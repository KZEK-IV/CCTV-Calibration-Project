from genericpath import sameopenfile
import numpy as np
import cv2 as cv
import random
import csv

def csvReader():

    images = []
    boxes = []
    list1 = []

    with open('data.csv', newline='') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=' ')
        for row in csvReader:
            list1 = row[0].split(",")
            images.append(list1[0])
            boxes.append([list1[1:9]])
    
    return images, boxes


# https://www.life2coding.com/cropping-polygon-or-non-rectangular-region-from-image-using-opencv-python/
def cropImage():
    
    images, boxes = csvReader()

    countNumber = 0
    for fname in images:
        img = cv.imread("train/calibrate/"+fname)
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        points = np.array([[[int(float(boxes[countNumber][0][0])),int(float(boxes[countNumber][0][1]))],[int(float(boxes[countNumber][0][2])),int(float(boxes[countNumber][0][3]))],[int(float(boxes[countNumber][0][4])),int(float(boxes[countNumber][0][5]))],[int(float(boxes[countNumber][0][6])),int(float(boxes[countNumber][0][7]))]]])
        
        #method 1 smooth region
        cv.drawContours(mask, [points], -1, (255, 255, 255), -1, cv.LINE_AA)

        #method 2 not so smooth region
        # cv.fillPoly(mask, points, (255))
        res = cv.bitwise_and(img,img,mask = mask)

        ## crate the white background of the same size of original image
        wbg = np.ones_like(img, np.uint8)*255
        cv.bitwise_not(wbg,wbg, mask=mask)

        countNumber = countNumber +1
        cv.imwrite('croppedCalibration/'+fname, res)
        listOfImages = []
        listOfImages.append(res)

# One time run to crop images
#cropImage()

def calibrateCamera():
    ## Code below by https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    ## Full guide on how to calibrate images using OpenCV, set up in an optimal fashion.

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepares arrays of points for the internal boxes on the calibration board
    boxPoints = np.zeros((7*11,3), np.float32)
    boxPoints[:,:2] = np.mgrid[0:11,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    boxPointsArray = [] 
    imagePointsArray = []

    outputs = csvReader()
    images = outputs[0]
    
    # Loops through all images in the list
    for fname in images:

    # DIFFERENT LOOP FOR SECONDARY DATASET
    #for fname in range(1,21):

        # Reads images into openCV format for manipulation
        # DIFFERENT IMAGE LOCATION FOR SECONDARY DATASET
        #img = cv.imread('leftcamera/Im_L_'+str(fname)+'.png')
        img = cv.imread('croppedCalibration/'+fname)
        # Image is converted to greyscale to be used by openCV
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (11,7), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            boxPointsArray.append(boxPoints)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)

            # Inputs the corners found to the list
            imagePointsArray.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(boxPointsArray, imagePointsArray, (576,1024), None, None)

    # Get a random image from the calibration set
    sampleImg = cv.imread('train/calibration/'+random.choice(images))
    # DIFFERENT SAMPLE FOR SECOND DATASET
    #sampleImg = cv.imread('leftcamera/Im_L_'+str(random.randrange(1,20))+'.png')
    h,  w = sampleImg.shape[:2]

    # A new matrix is created for the camera based on this input image and the values from the calibration
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    finalImage = cv.undistort(sampleImg, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    finalImage = finalImage[y:y+h, x:x+w]
    cv.imwrite('calibresult.png', finalImage)

calibrateCamera()