import numpy as np
import cv2
import matplotlib.pyplot as plt

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
    # for h in range(ht):
    #     if h%window_ht == 0:
    #         for w in range(wd):
    #             if wd%window_wd == 0:
    #                 D11 = Ixx[h:h+window_ht, w:w+window_wd].sum()
    #                 D22 = Iyy[h:h+window_ht, w:w+window_wd].sum()
    #                 D12 = Ixy[h:h+window_ht, w:w+window_wd].sum()
                
    #                 R = (D11*D22 - D12**2) - 0.06*(D11 + D22)

    #                 if R > threshold:
    #                     corners.append((h,w,R))
    #                     cv2.circle(duplicate_image, (w+1,h+1), 3, (0, 0, 255), -1)
    #                     kp = cv2.KeyPoint(w+1, h+1, 5, _class_id=0)
    #                     kps.append(kp)
    shape = np.shape(image)
    k1 = 3
    k2 = 3
    i = j = 0
    m = n = 1

    M = np.zeros((2, 2))
    out1 = np.zeros(np.shape(Ix))

    while(i + k1 <= shape[0]):
        while(j + k2 <= shape[1]):
            M[0][0] = np.sum(Ixx[i:i+k1, j:j+k2])
            M[1][0] = M[0][1] = np.sum(Ixy[i:i+k1, j:j+k2])
            # M[1][0] = np.sum(IxIy[i:i+k1, j:j+k2])
            M[1][1] = np.sum(Iyy[i:i+k1, j:j+k2])
            out1[m][n] = np.linalg.det(M) - 0.06*(np.trace(M)**2)
            # print(M)
            j += 1
            n += 1
            # break
        i += 1
        m += 1
        j = 0
        n = 0

    for i in range(np.shape(out1)[0]):
        for j in range(np.shape(out1)[1]):
            if out1[i][j] > threshold:
                cv2.circle(duplicate_image, (j, i), 3, (0, 0, 255), -1)
                kp = cv2.KeyPoint(j, i, 5, _class_id=0)
                kps.append(kp)

    plt.imshow(duplicate_image)
    plt.show()
    im1 = cv2.drawKeypoints(image, kps, None)
    plt.imshow(im1)
    plt.show()
    return duplicate_image, kps