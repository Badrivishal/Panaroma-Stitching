import numpy as np
import cv2

def harris_corner_detection(image, threshold):
    # n_images = images.shape[0]
    # image_name = "image_"
    # dict = {}
    # for i in range(n_images):
    #     name = image_name + str(i)
    #     dict[name] = n_images[i]
    Ix = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = Ix*Ix
    Iyy = Iy*Iy
    Ixy = Ix*Iy

    ht = image.shape[0]
    wd = image.shape[1]
    
    corners = []

    window_ht = 3
    window_wd = 3

    duplicate_image = np.copy(image)
    kps = []
    for h in range(ht):
        if h%window_ht == 0:
            for w in range(wd):
                if wd%window_wd == 0:
                    D11 = Ixx[h:h+window_ht, w:w+window_wd].sum()
                    D22 = Iyy[h:h+window_ht, w:w+window_wd].sum()
                    D12 = Ixy[h:h+window_ht, w:w+window_wd].sum()
                
                    R = (D11*D22 - D12**2) - 0.06*(D11 + D22)

                    if R > threshold:
                        corners.append((h,w,R))
                        cv2.circle(duplicate_image, (h,w), 3, (0, 0, 255), -1)
                        kp = cv2.KeyPoint(h, w, 5, _class_id=0)
                        kps.append(kp)

    return duplicate_image, kps