import cv2
import numpy as np
import os

# displays the image (press '0' t close and continue)
def displayImage(imgWindowName, img):
    cv2.imshow(imgWindowName, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#saves the image
def saveImage(imgName, img):
    cv2.imwrite(imgName, img)
    print("Saved image", imgName)

#returns the image name and the extension seperately
def getImageNameAndExtension(imgName):
  nameSegments = imgName.split(".")
  nameWithoutExtension = ""
  for i in range(len(nameSegments)-1):
    nameWithoutExtension += nameSegments[i]
  return nameWithoutExtension, nameSegments[-1]

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

# wraps the edges of the image
def wrapEdges(img):
    pixelsToWrap = N//2
    imageCopy = copy(img)
    imageCopy = list(imageCopy[-pixelsToWrap:]) + imageCopy  + imageCopy[:pixelsToWrap]
    for row in range(len(imageCopy)):
        imageCopy[row] = list(imageCopy[row][-pixelsToWrap:]) + imageCopy[row] + imageCopy[row][:pixelsToWrap]
    return np.array(imageCopy,dtype=np.uint8)

#returns the mean filter mask
def getMeanFilterMask(N):
    meanFilterMask = []
    for row in range(N):
        filterRow = [1]*N
        meanFilterMask.append(filterRow)
    return meanFilterMask

#apply mean filter
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

#apply median filter
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
      newRow.append([values[0][len(values[0])//2], values[1][len(values[1])//2], values[2][len(values[1])//2]])
    filteredImg.append(newRow)
  return np.array(filteredImg,dtype=np.uint8)

#apply mid-point filter
def applyMidPointFilter(N, wrappedImg, originalImg):
    rows, cols, channels = wrappedImg.shape
    Orows, Ocols, Ochannels = originalImg.shape
    newImg = np.zeros((Orows, Ocols, Ochannels))
    for d in range(0, rows - N + 1):
        for e in range(0, cols - N + 1):
            if ((d + N > rows - 1) and (e + N <= rows - 1)):
                matrix = wrappedImg[d:, e:e + N]
            elif ((d + N <= rows - 1) and (e + N > rows - 1)):
                matrix = wrappedImg[d:d + N, e:]
            elif ((d + N > rows - 1) and (e + N > rows - 1)):
                matrix = wrappedImg[d:, e:]
            else:
                matrix = wrappedImg[d:d + N, e:e + N]

            values = []
            for cha in range(Ochannels):
                values.append([])
            for a in range(0, N):
                for b in range(0, N):
                    for ch in range(0,Ochannels):
                        values[ch].append(matrix[a][b][ch])

            for ch in range(Ochannels):
                minVal = min(values[ch])
                maxVal = max(values[ch])
                newImg[d][e][ch] = round((int(minVal)+int(maxVal))/2)

    return np.array(newImg,dtype=np.uint8)


imageNames = []
path = os.getcwd()
for filename in os.listdir(path):
    if filename.split(".")[-1] == "jpg":
        imageNames.append(filename)

for imgName in imageNames:
    originalImg=cv2.imread(imgName,cv2.IMREAD_COLOR)
    # displayImage('Original Image', originalImg)
    nameWithoutExtension, extension = getImageNameAndExtension(imgName)

    N = 7

    wrappedImg = wrapEdges(originalImg)
    # displayImage('Wrapped Image', wrappedImg)

    meanFilteredImg = applyMeanFilter(N, wrappedImg)
    # displayImage('Mean Filtered Image', meanFilteredImg)
    saveImage(nameWithoutExtension+"_Mean_Filter."+extension, meanFilteredImg)

    medianFilteredImg = applyMedainFilter(N, wrappedImg)
    # displayImage('Median Filtered Image', medianFilteredImg)
    saveImage(nameWithoutExtension+"_Median_Filter."+extension, medianFilteredImg)

    midPointFilteredImg = applyMidPointFilter(N, wrappedImg, originalImg)
    # displayImage('Mid-Point Filtered Image', midPointFilteredImg)
    saveImage(nameWithoutExtension+"_MidPoint_Filter."+extension, midPointFilteredImg)

    # print("The shape of the original image is : ", originalImg.shape)
    # print("The shape of the wrapped image is : ", wrappedImg.shape)
    # print("The shape of the filtered image is : ", meanFilteredImg.shape)
    # print("The shape of the median filtered image is : ", medianFilteredImg.shape)
    # print("The shape of the midPoint filtered image is : ", midPointFilteredImg.shape)