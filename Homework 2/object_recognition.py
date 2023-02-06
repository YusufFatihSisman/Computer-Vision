import cv2 as cv
import os
import random
import time

appleDir = 'rgbd-dataset/apple/apple_1'
ballDir = 'rgbd-dataset/ball/ball_1'
bananaDir = 'rgbd-dataset/banana/banana_2'
pepperDir = 'rgbd-dataset/bell_pepper/bell_pepper_3'
binderDir = 'rgbd-dataset/binder/binder_2'
bowlDir = 'rgbd-dataset/bowl/bowl_5'
calculatorDir = 'rgbd-dataset/calculator/calculator_3'
combDir = 'rgbd-dataset/comb/comb_1'
plateDir = 'rgbd-dataset/plate/plate_1'
scissorsDir = 'rgbd-dataset/scissors/scissors_1'

def cvFAST(img):
    fast = cv.FastFeatureDetector_create()
    kp = fast.detect(img, None)
    br = cv.BRISK_create();
    kp, des = br.compute(img,  kp)
    return kp, des

def cvORB(img):
    orb = cv.ORB_create()
    kp, des = orb.detectAndCompute(img,None)
    return kp, des

def cvSIFT(img):
    sift = cv.SIFT_create(400)
    kp, des = sift.detectAndCompute(img,None)
    return kp, des

def imageList(directory):
    lst = []
    for file in os.listdir(directory):
        if file.endswith("_crop.png"):
            lst.append(directory + "/" + file)
    random.shuffle(lst)
    return lst

def splitAndTrain(lst, test, featureDetector):
    length = len(lst)
    des = []
    kp = []
    for x in range(length):
        if(x > length / 10):
            img = cv.imread(lst[x],0)
            if(featureDetector == 0):
                kpT, desT = cvORB(img)
            elif(featureDetector == 1):
                kpT, desT = cvFAST(img)
            elif(featureDetector == 2):
                kpT, desT = cvSIFT(img)
            
            if desT is not None:
                des.append(desT)
                kp.append(kp)
        else:
            test.append(lst[x])
    return kp, des             

def minIndex(arr, thValue, th):
    i = 0
    for j in range(1, len(arr)):
        if(arr[j] < arr[i] or arr[i] == 0):
            i = j
    if(th == True):
        if(arr[i] > thValue):
            return -1
    return i
    
def printResult(index):
    if(index == -1):
        print("Object can not be recognized") 
    elif(index == 0):
        print("Object is apple")
    elif(index == 1):
        print("Object is ball")
    elif(index == 2):
        print("Object is banana")
    elif(index == 3):
        print("Object is bell pepper")
    elif(index == 4):
        print("Object is binder")
    elif(index == 5):
        print("Object is bowl")
    elif(index == 6):
        print("Object is calculator")        
    elif(index == 7):
        print("Object is comb")
    elif(index == 8):
        print("Object is plate")
    elif(index == 9):
        print("Object is scissors")

def averageDistance(matches):
    sumDist = 0
    for x in matches:
        sumDist += x.distance
    return sumDist/len(matches)

def arrayAverage(arr):
    summation = 0
    for i in arr:
        summation += i
    return summation / len(arr)

def calcConfusion(res):
    print("---------CONFUSION MATRIX--------")
    newRes = []
    
    for i in res:
        apple = 0
        ball = 0
        banana = 0
        pepper = 0
        binder = 0
        bowl = 0
        calculator = 0
        comb = 0
        plate = 0
        scissors = 0
        other = 0
        for x in i:
            if(x == -1):
                other += 1
            elif(x == 0):
                apple += 1
            elif(x == 1):
                ball += 1
            elif(x == 2):
                banana += 1
            elif(x == 3):
                pepper += 1
            elif(x == 4):
                binder += 1
            elif(x == 5):
                bowl += 1
            elif(x == 6):
                calculator += 1
            elif(x == 7):
                comb += 1
            elif(x == 8):
                plate += 1
            elif(x == 9):
                scissors += 1
        subRes = [apple, ball, banana, pepper, binder,
                  bowl, calculator, comb, plate, scissors, other]
        newRes.append(subRes)
        newRes.append

    for i in newRes:
        print(i)

    print("-------------------------------------")
    
    tp = []
    fn = []
    fp = []
    tn = []
    
    # tp of classes
    for i in range(0, len(newRes)):
        tp.append(newRes[i][i])

    # fn of classes
    summation = 0
    for i in range(0, len(newRes)):
        summation = 0
        for j in range(0, len(newRes[0])-1):
            if(j != i):
                summation += newRes[i][j]
        fn.append(summation)
            
    # fp of classes
    for i in range(0, len(newRes[0])-1):
        summation = 0
        for j in range(0, len(newRes)):
            if(j != i):
                summation += newRes[j][i]
        fp.append(summation)           

    # tn of classes
    for x in range(0, len(newRes)):
        summation = 0
        for i in range(0, len(newRes)):
            for j in range(0, len(newRes[0])-1):
                if(x != i and x != j):
                    summation += newRes[i][j]
        tn.append(summation)
    
    print("tp: ", tp)
    print("fp: ", fp)
    print("tn: ", tn)
    print("fn: ", fn)

    totalTP = 0
    totalFP = 0
    totalFN = 0
    totalOther = 0

    precisions = []
    recalls = []
    
    for i in range(0, len(tp)):
        totalTP += tp[i]
        totalFP += fp[i]
        totalFN += fn[i]
        
        if(tp[i] + fp[i] != 0):
            precisions.append(tp[i]/(tp[i] + fp[i]))
        else:
            precisions.append(0)
            
        if(tp[i] + fn[i] != 0):
            recalls.append(tp[i]/(tp[i] + fn[i]))
        else:
            recalls.append(0)


    print("precisions: ", precisions)
    print("recalls: ", recalls)

    if(totalTP + totalFP != 0):
        print("Accuracy: ", totalTP / (totalTP + totalFP))
        print("Micro Precision: ", totalTP / (totalTP + totalFP))
    else:
        print("Accuracy: 0")
        print("Micro Precision: 0")

    if(totalTP + totalFN != 0):
        print("Micro Recall: ", totalTP / (totalTP + totalFN))
    else:
        print("Micro Recall: 0")

    print("Macro Precision: ", arrayAverage(precisions))
    print("Macro Recall: ", arrayAverage(recalls))
        
        
        
def recognition(des, testImage, featureDetector, th):
    if(featureDetector == 0):
        kpT, desT = cvORB(testImage)
    elif(featureDetector == 1):
        kpT, desT = cvFAST(testImage)
    elif(featureDetector == 2):
        kpT, desT = cvSIFT(testImage)
        
    if desT is None:
        #print("No feature detected on the test image")
        return -1
    
    if(featureDetector == 2):
        bf = cv.BFMatcher(cv.NORM_L1,crossCheck=False)
        thValue = 2000
    elif(featureDetector == 0):
        thValue = 70
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    elif(featureDetector == 1):
        thValue = 110
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    dApple = 0
    dBall = 0
    dBanana = 0
    dPepper = 0
    dBinder = 0   
    dBowl = 0
    dCalculator = 0
    dComb = 0
    dPlate = 0
    dScissors = 0
    
    for i in des[0]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dApple += averageDistance(matches[:20])
    for i in des[1]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dBall += averageDistance(matches[:20])
    for i in des[2]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dBanana += averageDistance(matches[:20])
    for i in des[3]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dPepper += averageDistance(matches[:20])
    for i in des[4]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dBinder += averageDistance(matches[:20])
    for i in des[5]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dBowl += averageDistance(matches[:20])
    for i in des[6]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dCalculator += averageDistance(matches[:20])
    for i in des[7]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dComb += averageDistance(matches[:20])
    for i in des[8]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dPlate += averageDistance(matches[:20])
    for i in des[9]:   
        matches = bf.match(i,desT)
        matches = sorted(matches, key = lambda x:x.distance)
        dScissors += averageDistance(matches[:20])

    if(len(des[0]) != 0):
        dApple /= len(des[0])
    if(len(des[1]) != 0):
        dBall /= len(des[1])
    if(len(des[2]) != 0):
        dBanana /= len(des[2])
    if(len(des[3]) != 0):
        dPepper /= len(des[3])
    if(len(des[4]) != 0):
        dBinder /= len(des[4])
    if(len(des[5]) != 0):
        dBowl /= len(des[5])
    if(len(des[6]) != 0):
        dCalculator /= len(des[6])
    if(len(des[7]) != 0):
        dComb /= len(des[7])
    if(len(des[8]) != 0):
        dPlate /= len(des[8])
    if(len(des[9]) != 0):
        dScissors /= len(des[9])

##    print("apple: ", dApple)
##    print("ball: ", dBall)
##    print("banana: ", dBanana)
##    print("pepper: ", dPepper)
##    print("binder: ", dBinder)
##    print("bowl: ", dBowl)
##    print("calculator: ", dCalculator)
##    print("comb: ", dComb)
##    print("plate: ", dPlate)
##    print("scissors: ", dScissors)
    
    arr = [dApple, dBall, dBanana, dPepper, dBinder,
           dBowl, dCalculator, dComb, dPlate, dScissors]

    index = minIndex(arr, thValue, th)
    #printResult(index)
    return index

def trainDir(des, directory, test, featureDetector):
    start = time.time()
    lst = imageList(directory)
    kp, desT = splitAndTrain(lst, test, featureDetector)
    des.append(desT)
    end = time.time()
    return end-start

def testResults(results, des, test, featureDetector, startIndex, th):
    i = 0
    subResults = []
    testImage = cv.imread(test[startIndex],0)
    averageRecTime = 0
    while(i < 10):
        start = time.time()
        res = recognition(des, testImage, featureDetector, th)
        end = time.time()
        averageRecTime += end - start
        if(res != -1):
            print(end-start)
        i += 1
        subResults.append(res)
        startIndex += 1
        testImage = cv.imread(test[startIndex],0)
    results.append(subResults)
    averageRecTime /= 10
    return averageRecTime
    
def main(featureDetector, th):
    if(featureDetector == 0):
        print("ORB Feature Detector")
    elif(featureDetector == 1):
        print("FAST Feature Detector")
    elif(featureDetector == 2):
        print("SIFT Feature Detector")
    else:
        print("featureDetector parameter must be 0, 1 or 2")
        return False

    testAppleIndex = 0
    testBallIndex = 0
    testBananaIndex = 0
    testPepperIndex = 0
    testBinderIndex = 0
    testBowlIndex = 0
    testCalculatorIndex = 0
    testCombIndex = 0
    testPlateIndex = 0
    testScissorsIndex = 0
    
    totalTimeStart = time.time()
    test = []
    des = []

    end = trainDir(des, appleDir, test, featureDetector)
    print("The time of extraction the features of apple is: ", end)
    
    testBallIndex = len(test)
    end = trainDir(des, ballDir, test, featureDetector)
    print("The time of extraction the features of ball is: ", end)
 
    testBananaIndex = len(test)
    end = trainDir(des, bananaDir, test, featureDetector)
    print("The time of extraction the features of banana is: ", end)
 
    testPepperIndex = len(test)
    end = trainDir(des, pepperDir, test, featureDetector)
    print("The time of extraction the features of bell pepper is: ", end)
 
    testBinderIndex = len(test)
    end = trainDir(des, binderDir, test, featureDetector)
    print("The time of extraction the features of binder is: ", end)
 
    testBowlIndex = len(test)
    end = trainDir(des, bowlDir, test, featureDetector)
    print("The time of extraction the features of bowl is: ", end)
 
    testCalculatorIndex = len(test)
    end = trainDir(des, calculatorDir, test, featureDetector)
    print("The time of extraction the features of calculator is: ", end)
 
    testCombIndex = len(test)
    end = trainDir(des, combDir, test, featureDetector)
    print("The time of extraction the features of comb is: ", end)
 
    testPlateIndex = len(test)
    end = trainDir(des, plateDir, test, featureDetector)
    print("The time of extraction the features of plate is: ", end)
 
    testScissorsIndex = len(test)
    end = trainDir(des, scissorsDir, test, featureDetector)
    print("The time of extraction the features of scissors is: ", end)
 
    end = time.time()
    print("The time of extraction the features of all objects is: ", end-totalTimeStart)
    

    results = []
    
    averageTime = 0
    averageTime += testResults(results, des, test, featureDetector, testAppleIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testBallIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testBananaIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testPepperIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testBinderIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testBowlIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testCalculatorIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testCombIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testPlateIndex, th)
    averageTime +=testResults(results, des, test, featureDetector, testScissorsIndex, th)

    averageTime /= 10
    print("Average recognition time is: ", averageTime)

    for i in results:
        print(i)
        
    calcConfusion(results)

main(0, False)
main(1, False)
main(2, False)
##main(0, True)
##main(1, True)
##main(2, True)
