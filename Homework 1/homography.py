import cv2
import numpy as np

dstPoints = np.float32([[0, 0], [702, 0],
                            [0, 384], [702, 384]])

def setIntersection(lst, dst):
    ptH = intersection(lst[0][0], lst[0][1], lst[1][0], lst[1][1], lst[2][0], lst[2][1], lst[3][0], lst[3][1])
    ptV = intersection(lst[0][0], lst[0][1], lst[2][0], lst[2][1], lst[1][0], lst[1][1], lst[3][0], lst[3][1])
    
    print(ptH)
    print(ptV)
    indexH, valH = findCorrectCorner(ptH, lst)
    indexV, valV = findCorrectCorner(ptV, lst)
    if(valH < valV):
        dst[indexH] = np.float32(ptH)
    else:
        dst[indexV] = np.float32(ptV)
    
    
def findCorrectCorner(pt, lst):
    distance0 = (((pt[0] - lst[0][0])**2 + (pt[1] - lst[0][1])**2 )**0.5)
    distance1 = (((pt[0] - lst[1][0])**2 + (pt[1] - lst[1][1])**2 )**0.5)
    distance2 = (((pt[0] - lst[2][0])**2 + (pt[1] - lst[2][1])**2 )**0.5)
    distance3 = (((pt[0] - lst[3][0])**2 + (pt[1] - lst[3][1])**2 )**0.5)
    
    minVal = distance0
    mini = 0
    if(distance1 < minVal):
        minVal = distance1
        mini = 1
    if(distance2 < minVal):
        minVal = distance2
        mini = 2
    if(distance3 < minVal):
        mini = 3

    return mini, minVal
       
def intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    D = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    Px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / D
    Py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / D
    lst = np.float32([Px, Py])
    return lst

def takePic():
    cam = cv2.VideoCapture(0)
    res = 0
    while True:
        result, image = cam.read()
        cv2.imshow("test", image)
        key = cv2.waitKey(1)
        if(key!=-1):
            res = image
            break;       
    cam.release()
    cv2.destroyAllWindows()
    return res
    
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        coords = [x, y]
        params[0].append(coords)
        params[1] += 1

def markCorners():
    lst = []
    params = []
    params.append(lst)
    params.append(0)
    while params[1] < 4:
        cv2.setMouseCallback('mark corners', click_event, params)
        cv2.waitKey(10)
    return params[0]

def solve(lst, dst):
    a = np.float32([[dst[0][0]*lst[0][0], dst[0][0]*lst[0][1], 0, 0, 0, -lst[0][0], -lst[0][1], -1],
                  [dst[0][1]*lst[0][0], dst[0][1]*lst[0][1], -lst[0][0], -lst[0][1], -1, 0, 0, 0],
                  [dst[1][0]*lst[1][0], dst[1][0]*lst[1][1], 0, 0, 0, -lst[1][0], -lst[1][1], -1],
                  [dst[1][1]*lst[1][0], dst[1][1]*lst[1][1], -lst[1][0], -lst[1][1], -1, 0, 0, 0],
                  [dst[2][0]*lst[2][0], dst[2][0]*lst[2][1], 0, 0, 0, -lst[2][0], -lst[2][1], -1],
                  [dst[2][1]*lst[2][0], dst[2][1]*lst[2][1], -lst[2][0], -lst[2][1], -1, 0, 0, 0],
                  [dst[3][0]*lst[3][0], dst[3][0]*lst[3][1], 0, 0, 0, -lst[3][0], -lst[3][1], -1],
                  [dst[3][1]*lst[3][0], dst[3][1]*lst[3][1], -lst[3][0], -lst[3][1], -1, 0, 0, 0]])
    b = np.float32([-dst[0][0], -dst[0][1], -dst[1][0], -dst[1][1], -dst[2][0], -dst[2][1], -dst[3][0], -dst[3][1]])

    X = np.linalg.inv(a).dot(b)

    line1 = np.float32([X[5], X[6], X[7]])
    line2 = np.float32([X[2], X[3], X[4]])
    line3 = np.float32([X[0], X[1], 1])                   
    res = np.float32([line1, line2, line3])
    return res
                
def output(intrsct=True, mySolution=False):
    img = takePic()
    cv2.imshow('mark corners', img)
    lst = markCorners()
    cv2.destroyAllWindows()
    
    intersectMatrix = 0

    if(intrsct == True):
        dstPointsIntersect = np.float32([[0, 0], [702, 0], [0, 384], [702, 384]])
        setIntersection(lst, dstPointsIntersect)
        intersectMatrix = cv2.getPerspectiveTransform(np.float32(lst), dstPointsIntersect)
        print("Intersect Matrix")
        print(intersectMatrix)
        print("Intersect Points")
        print(dstPointsIntersect)
        if(mySolution == True):
            matrixMySolution = solve(np.float32(lst), dstPointsIntersect)
            print("my solution matrix with intersection")
            print(matrixMySolution)
            result = cv2.warpPerspective(img, matrixMySolution, (702, 384))
            cv2.imshow('resultMySolutionIntersect', result)
    
    matrix = cv2.getPerspectiveTransform(np.float32(lst), dstPoints)
    print("matrix")
    print(matrix)

    if(mySolution == True):
        matrixMySolution = solve(np.float32(lst), dstPoints)
        print("my solution matrix")
        print(matrixMySolution)
        result = cv2.warpPerspective(img, matrixMySolution, (702, 384))
        cv2.imshow('resultMySolution', result)
    
    result = cv2.warpPerspective(img, matrix, (702, 384))
    cv2.imshow('result', result)

    if(intrsct == True):
        resultIntersect = cv2.warpPerspective(img, intersectMatrix, (702, 384))
        cv2.imshow('resultIntersect', resultIntersect)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

output(mySolution = True)




