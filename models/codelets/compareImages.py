import cv2
import sys
import numpy as np
import math


def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
    out[:rows1, :cols1] = np.dstack([img1])
    out[:rows2, cols1:] = np.dstack([img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt

        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0, 1), 1)
        cv2.circle(out, (int(x2) + cols1, int(y2)), 4, (255, 0, 0, 1), 1)
        cv2.line(out, (int(x1), int(y1)), (int(x2) + cols1, int(y2)), (255, 0, 0, 1), 1)

    return out


def compare(img1, img2):

    # Initiate SIFT detector
#    sift = cv2.SIFT()
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher(cv2.NORM_L2, True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda val: val.distance)
    print("number of matches: ",len(matches))
    print("number of descriptors 1: ", len(des1))
    print("number of descriptors 2: ", len(des2))
    print("matches over descriptors 1: ",len(matches)/len(des1))
    print("matches minus descriptors 2: ",abs(len(matches)-len(des2)))

    return len(matches), len(des1), len(des2), (len(matches)/len(des1)), abs(len(matches)-len(des2))

    #print("frame anterior: ", len(matches) / len(des2))
    #img3 = drawMatches(img1, kp1, img2, kp2, matches[:25])

    # Show the image
    #cv2.imshow('Matched Features', img3)
    #cv2.waitKey(0)
    #cv2.destroyWindow('Matched Features')


def compareHist(src_base, src_test1):


    hsv_base = src_base
    hsv_test1 = src_test1

    hsv_half_down = hsv_base[hsv_base.shape[0] // 2:, :]
    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    # hue varies from 0 to 179, saturation from 0 to 255
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges  # concat lists
    # Use the 0-th and 1-st channels
    channels = [0, 1]
    hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_half_down = cv2.calcHist([hsv_half_down], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    hist_test1 = cv2.calcHist([hsv_test1], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    res = []

    for compare_method in range(4):
        #base_base = cv2.compareHist(hist_base, hist_base, compare_method)
        #base_half = cv2.compareHist(hist_base, hist_half_down, compare_method)
        base_test1 = cv2.compareHist(hist_base, hist_test1, compare_method)
        print('Method:', compare_method, ': ',base_test1)
        res.append(base_test1)


    return res[3], res[2], res[1], res[0]