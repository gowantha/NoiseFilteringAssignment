import cv2
import numpy as np
from matplotlib import pyplot as plt

# displays the image (press '0' t close and continue)
def displayImage(imgWindowName, img):
    cv2.imshow(imgWindowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#saves the image
def saveImage(imgName, img):
    cv2.imwrite(imgName, img)

#returns a copy of the image
def copy(img):
  imageCopy=[]
  for row in range(img.shape[0]):
    newRow=[]
    for col in range(img.shape[1]):
      newPixel = []
      for ch in range(img.shape[2]):
        newPixel.append(img[row][col][ch])
      newRow.append(newPixel)
    imageCopy.append(newRow)
  return imageCopy

def wrapEdges(img):
    pixelsToWrap = N//2
    imageCopy = copy(img)
    imageCopy = list(imageCopy[-pixelsToWrap:]) + imageCopy  + imageCopy[:pixelsToWrap]
    for row in range(len(imageCopy)):
        imageCopy[row] = list(imageCopy[row][-pixelsToWrap:]) + imageCopy[row] + imageCopy[row][:pixelsToWrap]
    return np.array(imageCopy,dtype=np.uint8)

def getMeanFilterMask(N):
    meanFilterMask = []
    for row in range(N):
        filterRow = [1]*N
        meanFilterMask.append(filterRow)
    return meanFilterMask

#returns a copy of the image with fixed filter applied
def applyMeanFilter(N, img):
  meanFilterMask = getMeanFilterMask(N)
  filteredImg = []
  filterMid = len(meanFilterMask)//2
  for row in range(filterMid, len(img)-filterMid):
    newRow = []
    for col in range(filterMid, len(img[row])-filterMid):
      values = [[],[],[]]
      for i in range(len(meanFilterMask)):
        for j in  range(len(meanFilterMask[i])):
          for k in range(img.shape[2]):
            values[k].append(img[row-filterMid+i][col-filterMid+j][k] * meanFilterMask[i][j])
      newRow.append([sum(values[0])/len(values[0]), sum(values[1])/len(values[1]), sum(values[2])/len(values[2])])
    filteredImg.append(newRow)
  return np.array(filteredImg,dtype=np.uint8)

def applyMedainFilter(N, img):
  filteredImg = []
  filterMid = N//2
  for row in range(filterMid, len(img)-filterMid):
    newRow = []
    for col in range(filterMid, len(img[row])-filterMid):
      values = [[],[],[]]
      for i in range(N):
        for j in  range(N):
          for k in range(img.shape[2]):
            values[k].append(img[row-filterMid+i][col-filterMid+j][k])
      for k in range(img.shape[2]):
        values[k].sort()
      newRow.append([values[0][4], values[1][4], values[2][4]])
    filteredImg.append(newRow)
  return np.array(filteredImg,dtype=np.uint8)

def applyMidPointFilter(N, img):
  filteredImg = []
  filterMid = N//2
  for row in range(filterMid, len(img)-filterMid):
    newRow = []
    for col in range(filterMid, len(img[row])-filterMid):
      values = [[],[],[]]
      for i in range(N):
        for j in  range(N):
          for k in range(img.shape[2]):
            values[k].append(img[row-filterMid+i][col-filterMid+j][k])
      for k in range(img.shape[2]):
        values[k].sort()
      newRow.append([(min(values[0]) + max(values[0])) / 2, (min(values[1]) + max(values[1])) / 2, (min(values[2]) + max(values[2])) / 2])
    filteredImg.append(newRow)
  return np.array(filteredImg,dtype=np.uint8)


name="testImg.jpg"
originalImg=cv2.imread(name,cv2.IMREAD_COLOR)
displayImage('Original Image', originalImg)
N = 3

wrappedImg = wrapEdges(originalImg)
print("The shape of the original image is : ", originalImg.shape)
print("The shape of the wrapped image is : ", wrappedImg.shape)
displayImage('Wrapped Image', wrappedImg)