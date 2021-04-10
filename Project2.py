import numpy as np
import cv2 as cv
from harris_detector import *
import imutils
import os

# address1 = 'imagesSet2/2.jpg'
# # address1 = 'imagesSet1/uttower_left.jpg'
# # address1 = 'image1.jpg'
# image1 = cv.imread(address1)
# image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
# gray1 = cv.cvtColor(image1, cv.COLOR_RGB2GRAY)
# # plt.imshow(image1)

# address2 = 'imagesSet2/3.jpg'
# # address2 = 'imagesSet1/uttower_right.jpg'
# # address2 = 'image2.jpg'
# image2 = cv.imread(address2)
# image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
# gray2 = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)
# plt.imshow(image2)

def stitching(addresses, threshold):

    address1, address2 = addresses
    image1 = cv.imread(address1)
    image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
    gray1 = cv.cvtColor(image1, cv.COLOR_RGB2GRAY)

    image2 = cv.imread(address2)
    image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
    gray2 = cv.cvtColor(image2, cv.COLOR_RGB2GRAY)

    
    im, lst1 = harris_corner_detection(gray1, threshold)
    im, lst2 = harris_corner_detection(gray2, threshold)

    sift = cv.BRISK_create()


    kp1, desc1 = sift.compute(gray1, lst1)
    kp2, desc2 = sift.compute(gray2, lst2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches2 = bf.match(desc1,desc2)
    matches2 = sorted(matches2, key = lambda x:x.distance)

    print(len(matches2), len(desc1), len(desc2))

    img3 = cv.drawMatches(image1, kp1, image2, kp2, matches2[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.imshow(img3)
    plt.show()

    image_1_points = np.zeros((len(matches2), 1, 2), dtype=np.float32)
    image_2_points = np.zeros((len(matches2), 1, 2), dtype=np.float32)

    for i in range(0,len(matches2)):
        image_1_points[i] = kp1[matches2[i].queryIdx].pt
        image_2_points[i] = kp2[matches2[i].trainIdx].pt

    (H, status) = cv.findHomography(image_2_points, image_1_points, cv.RANSAC, 4)
    print(H)
    print(len(matches2), len(kp1))

    width = image2.shape[1] + image1.shape[1]
    height = image2.shape[0] + image1.shape[0]

    result = cv.warpPerspective(image2, H, (width, height))

    # result[0:image1.shape[0], 0:image1.shape[1]] = image1
    # mask = np.where(result==0, 1, 0)
    # result[0:image1.shape[0], 0:image1.shape[1]] = cv2.bitwise_and(image1, result[0:image1.shape[0], 0:image1.shape[1]], mask)
    
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if result[i,j].all() == 0:
                result[i,j] = image1[i,j]

    # plt.imshow(result)
    # plt.axis('off')
    # plt.show()

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]

    # show the cropped image
    plt.imshow(result)
    plt.show()

    return result, H

def panorama(folder):
    files = os.listdir(folder)
    images = [folder + '/' + i for i in files]
    addresses = [images[0]]
    threshold = 50000000000
    for i in range(1,len(images)):
        addresses.append(images[i])
        result, H = stitching(addresses, threshold)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        print(i)
        filename = 'result'+str(i)+'.jpg'
        cv2.imwrite(filename, result)
        addresses = [filename]
        threshold = threshold*1.5
    return None

panorama('imagesSet2')