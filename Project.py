import numpy as np
import cv2
from harris_detector import *

def stitch_images(images):

    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher(crossCheck=False)

    n_images = len(images)

    images_dict = {}
    keypts_harris_dict = {}
    keypts_sift_dict = {}
    descptr_sift_dict = {}

    for i in range(n_images):
        gbr_img = cv2.imread(images[i], -1)

        gray_img = cv2.cvtColor(gbr_img, cv2.COLOR_BGR2GRAY)

        _, keypts = harris_corner_detection(gray_img, 100)
        # print("KEYPOINTS", len(keypts))
        kps, desptr = sift.compute(gray_img, keypts)
        # print("SIFT KEYPOINTS", len(keypts))
        # print(np.array([kps[i]==keypts[i] for i in range(len(kps))]).sum())
        images_dict[i] = gbr_img
        keypts_harris_dict[i] = keypts
        keypts_sift_dict[i] = kps
        descptr_sift_dict[i] = desptr

    matches = bf.knnMatch(descptr_sift_dict[0],descptr_sift_dict[1],k=2)
    # matches = bf_matcher.match(descptr_sift_dict[0], descptr_sift_dict[1])
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    good = sorted(good, key = lambda x:x.distance)
    good = good[:15]
    pt1 = np.float32([keypts_sift_dict[0][m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pt2 = np.float32([keypts_sift_dict[1][m.trainIdx].pt for m in good]).reshape(-1,1,2)
    # for i in range(len(matches)):

    #     images_points_1[i] = keypts_sift_dict[0][matches[i].queryIdx].pt
    #     images_points_2[i] = keypts_sift_dict[1][matches[i].queryIdx].pt

    # 
    # (H, status) = cv2.findHomography(pt1, pt2, cv2.RANSAC, 4)
    print(pt1.shape, pt2.shape, len(matches))
    H,mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,4)
    print(H)
    final_width = images_dict[0].shape[1] + images_dict[1].shape[1]
    final_height = images_dict[0].shape[0] + images_dict[1].shape[0]
    
    final_image = cv2.warpPerspective(images_dict[1], H, (final_width, final_height))

    final_image[:images_dict[0].shape[0], :images_dict[0].shape[1]] = images_dict[0]

    return final_image

def panorama_stitching(images):

    images_list = [images[0]]
    for i in range(len(images)-1):
        i = i+1
        images_list.append(images[i])
        result = stitch_images(images_list)
        cv2.imwrite("result"+str(i)+".jpg", result)
        images_list = ["result"+str(i)+".jpg"]
        
    
    cv2.imshow("frame", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

panorama_stitching(["image1.jpg", "image2.jpg"])