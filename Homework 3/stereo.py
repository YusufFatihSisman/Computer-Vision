import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time

saveType = ".jpg"

def cvFAST(img1, img2):
    fast = cv.FastFeatureDetector_create()
    kp1= fast.detect(img1,None)
    kp2 = fast.detect(img2,None)
    br = cv.BRISK_create();
    kp1, des1 = br.compute(img1,  kp1)
    kp2, des2 = br.compute(img2,  kp2)
    return kp1, des1, kp2, des2

def cvORB(img1, img2):
    orb = cv.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    return kp1, des1, kp2, des2

def cvSIFT(img1, img2):
    sift = cv.SIFT_create(400)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    return kp1, des1, kp2, des2

def cvCANNY(img1, img2, cols, rows):
    edges1 = cv.Canny(img1,150,200)
    edges2 = cv.Canny(img2,150,200)
    orb = cv.ORB_create()
    kp1 = []
    kp2 = []
    for i in range(rows):
        for j in range(cols):
            if(edges1[i][j] == 255):
                kp1.append(cv.KeyPoint(j, i, 1))
            if(edges2[i][j] == 255):
                kp2.append(cv.KeyPoint(j, i, 1))

    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
    return kp1, des1, kp2, des2
   
def dist(arr, kp1, kp2, matches, isLeft):
    for i in range(len(matches)):
        left = kp1[matches[i].queryIdx].pt;
        right = kp2[matches[i].trainIdx].pt;
        y = int(left[1])
        x = int(left[0])
        if(arr[y][x] == 0):
            val = int(left[0] - right[0])
            if(isLeft == False):
                val = -val
            arr[y][x] = val * 8 if val >= 0 else 0

def canny(arr, img1, img2, isLeft, cols, rows):
    kp1, des1, kp2, des2 = cvCANNY(img1, img2, cols, rows)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    dist(arr, kp1, kp2, matches, isLeft)

def fast(arr, img1, img2, isLeft):
    kp1, des1, kp2, des2 = cvFAST(img1, img2)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    dist(arr, kp1, kp2, matches, isLeft)

def orb(arr, img1, img2, isLeft):
    kp1, des1, kp2, des2 = cvORB(img1, img2)
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    dist(arr, kp1, kp2, matches, isLeft)

def sift(arr, img1, img2, isLeft):
    kp1, des1, kp2, des2 = cvSIFT(img1, img2)
    bf = cv.BFMatcher(cv.NORM_L1,crossCheck=False)
    matches = bf.match(des1,des2)
    dist(arr, kp1, kp2, matches, isLeft)

def saveImage(arr, name):
    arr = np.array(arr).astype('uint8')
    arr = cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
    cv.imwrite(name, arr)

def set1(arr, img1, img2, name, isLeft): # sift, orb
    sift(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set1_step1" + saveType))     
    orb(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set1_result" + saveType))

def set2(arr, img1, img2, name, isLeft): # fast, sift
    fast(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set2_step1" + saveType)) 
    sift(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set2_result" + saveType))

def set3(arr, img1, img2, name, isLeft, cols, rows): # fast, canny
    fast(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set3_step1" + saveType))
    canny(arr, img1, img2, isLeft, cols, rows)
    saveImage(arr, (name + "_set3_result" + saveType))

def set4(arr, img1, img2, name, isLeft, cols, rows): # fast, canny, sift, orb
    fast(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set4_step1" + saveType))
    canny(arr, img1, img2, isLeft, cols, rows)
    saveImage(arr, (name + "_set4_step2" + saveType))
    sift(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set4_step3" + saveType))
    orb(arr, img1, img2, isLeft)
    saveImage(arr, (name + "_set4_result" + saveType))

def compare(gt, arr, cols, rows):
    mse, mre, bmp, bmpre = 0, 0, 0, 0
    t = 1
    for i in range(rows):
        for j in range(cols):
            mse += pow(gt[i][j]/8 - arr[i][j]/8, 2)
            mre += (abs(gt[i][j]/8 - arr[i][j]/8))/(gt[i][j]/8)
            if(abs(gt[i][j]/8 - arr[i][j]/8) > t):
                bmp += 1
                if(gt[i][j]/8 > 0):
                    bmpre += (abs(gt[i][j]/8 - arr[i][j]/8))/(gt[i][j]/8)
    mse = mse / (381 * 432)
    mre = mre / (381 * 432)
    bmp = bmp / (381 * 432)
    print("mse: ", mse)
    print("mre: ", mre)
    print("bmp: ", bmp)
    print("bmpre: ", bmpre)


def combine(arr, disp, cols, rows):
    for y in range(rows):
        for x in range(cols):
            if(arr[y][x] == 0):
                arr[y][x] = disp[y][x]
    

def compareCaller(gt, arr, disparity, name, imageName, cols, rows):
    print("Compare with ", name, ": ")
    compare(gt, arr, cols, rows)
    print("Compare with disparity:")
    compare(gt, disparity, cols, rows)
    combine(arr, disparity, cols, rows)
    saveImage(arr, (imageName + "_" + name + "_resultAndDisparity" + saveType))
    print("Compare with ", name, " + disparity: ")
    compare(gt, arr, cols, rows)

def func(name, cols, rows):
    img1 = cv.imread((name + '/im2.ppm'),0)
    img2 = cv.imread((name + '/im6.ppm'),0)
    gtL = cv.imread((name + '/disp2.pgm'),0)
    gtR = cv.imread((name + '/disp6.pgm'),0)
    
    arr = [[0 for x in range(cols)] for y in range(rows)]

    # SET 1 Left
    print("SET 1 LEFT")
    start = time.time()
    set1(arr, img1, img2, (name + "_left"), True)
    end = time.time()
    print("The time of execution of set1 left is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img1,img2)
    saveImage(disparity, (name + "_set1_left_disparity" + saveType))
    compareCaller(gtL, arr, disparity, "set1_left", name, cols, rows)

    # SET 1 Right
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 1 RIGHT")
    start = time.time()
    set1(arr, img2, img1, (name + "_right"), False)
    end = time.time()
    print("The time of execution of set1 right is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img2, img1)
    saveImage(disparity, (name + "_set1_right_disparity" + saveType))
    compareCaller(gtR, arr, disparity, "set1_right", name, cols, rows)

    # SET 2 Left
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 2 LEFT")
    start = time.time()
    set2(arr, img1, img2, (name + "_left"), True)
    end = time.time()
    print("The time of execution of set2 left is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img1,img2)
    saveImage(disparity, (name + "_set2_left_disparity" + saveType))
    compareCaller(gtL, arr, disparity, "set2_left", name, cols, rows)

    # SET 2 Right
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 2 RIGHT")
    start = time.time()
    set2(arr, img2, img1, (name + "_right"), False)
    end = time.time()
    print("The time of execution of set2 right is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img2, img1)
    saveImage(disparity, (name + "_set2_right_disparity" + saveType))
    compareCaller(gtR, arr, disparity, "set2_right", name, cols, rows)

    # SET 3 Left
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 3 LEFT")
    start = time.time()
    set3(arr, img1, img2, (name + "_left"), True, cols, rows)
    end = time.time()
    print("The time of execution of set3 left is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img1,img2)
    saveImage(disparity, (name + "_set3_left_disparity" + saveType))
    compareCaller(gtL, arr, disparity, "set3_left", name, cols, rows)

    # SET 3 Right
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 3 RIGHT")
    start = time.time()
    set3(arr, img2, img1, (name + "_right"), False, cols, rows)
    end = time.time()
    print("The time of execution of set3 right is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img2, img1)
    saveImage(disparity, (name + "_set3_right_disparity" + saveType))
    compareCaller(gtR, arr, disparity, "set3_right", name, cols, rows)

    # SET 4 Left
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 4 LEFT")
    start = time.time()
    set4(arr, img1, img2, (name + "_left"), True, cols, rows)
    end = time.time()
    print("The time of execution of set4 left is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img1,img2)
    saveImage(disparity, (name + "_set4_left_disparity" + saveType))
    compareCaller(gtL, arr, disparity, "set4_left", name, cols, rows)

    # SET 4 Right
    arr = [[0 for x in range(cols)] for y in range(rows)]
    print("SET 4 RIGHT")
    start = time.time()
    set4(arr, img2, img1, (name + "_right"), False, cols, rows)
    end = time.time()
    print("The time of execution of set4 right is :", end-start)
    stereo = cv.StereoBM_create(numDisparities=32, blockSize=17)
    disparity = stereo.compute(img2, img1)
    saveImage(disparity, (name + "_set4_right_disparity" + saveType))
    compareCaller(gtR, arr, disparity, "set4_right", name, cols, rows)

func("barn1", 432, 381)      
func("barn2", 430, 381)
func("bull", 433, 381)
func("poster", 435, 383)
func("sawtooth", 434, 380)
func("venus", 434, 383)



