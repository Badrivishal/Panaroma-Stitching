import numpy as np
import cv2 as cv
from harris_detector import *

# address1 = 'imagesSet2/1.jpg'
# address1 = 'imagesSet1/uttower_left.jpg'
address1 = 'image1.jpg'
image1 = cv.imread(address1)
image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
# plt.imshow(image1)

# address2 = 'imagesSet2/2.jpg'
# address2 = 'imagesSet1/uttower_right.jpg'
address2 = 'image2.jpg'
image2 = cv.imread(address2)
image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
# plt.imshow(image2)

im, lst1 = harris_corner_detection(gray1, 50000000000)
im, lst2 = harris_corner_detection(gray2, 50000000000)

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

plt.figure(figsize=(20,10))
plt.imshow(result)
plt.axis('off')
plt.show()