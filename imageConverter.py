#!/bin/env python
import skimage.io,skimage.transform
import numpy as np
import pandas as pd
import math


#Defining functions to find borders of non-empty section of canvas
def findTopRow(image):
  for i in range(image.shape[0]):
    if sum(image[i,:])>0:
      return i
  return 0

def findBottomRow(image):
  for i in range(image.shape[0]):
    if sum(image[image.shape[0]-i-1,:])>0:
      return image.shape[0]-i-1
  return image.shape[0]-1

def findFirstColumn(image):
  for i in range(image.shape[1]):
    if sum(image[:,i])>0:
      return i
  return 0

def findLastColumn(image):
  for i in range(image.shape[1]):
    if sum(image[:,image.shape[1]-i-1])>0:
      return image.shape[1]-i-1
  return image.shape[1]-1


#function clips whitespace off of image
#returns square array which includes all
#nonzero elements of orinal array
def clipImage(image):

  #finding borders of drawing
  top=findTopRow(image)
  bottom=findBottomRow(image)
  left=findFirstColumn(image)
  right=findLastColumn(image)

  #Setting width to largest dimension of drawing
  width=max(bottom-top,right-left)
  
  #adding 10% padding to image, unless this causes the square to be larger than the original image
  width=min(min(image.shape),width*1.1)

  #finding center of image
  centerVert=(top+bottom)/2
  centerHor=(right+left)/2

  #Handling cases where clipped square would exit borders of original image
  if centerVert-(width/2)<0:
    clippedTop=0
    clippedBottom=math.ceil(width)

  elif centerVert+(width/2)>image.shape[0]:
    clippedTop=math.floor(image.shape[0]-width-1)
    clippedBottom=image.shape[0]-1

  else:
    clippedTop=math.floor(centerVert-(width/2))
    clippedBottom=math.ceil(centerVert+(width/2))


  if centerHor-(width/2)<0:
    clippedLeft=0
    clippedRight=math.ceil(width)

  elif centerHor+(width/2)>image.shape[1]:
    clippedLeft=math.floor(image.shape[1]-(width)-1)
    clippedRight=image.shape[1]-1

  else:
    clippedLeft=math.floor(centerHor-(width/2))
    clippedRight=math.ceil(centerHor+(width/2))


  return image[clippedTop:clippedBottom+1,clippedLeft:clippedRight+1]


#Function takes canvas eps, converts it to an array
#then inverts clips and resizes the image
def imageConverter(canvas):
    canvasArray=skimage.io.imread(canvas,as_gray=True)
    clippedCanvas=clipImage(1-canvasArray)
    image=skimage.transform.resize(clippedCanvas,(28,28))
    image=image-np.amin(image)

    #Array is normalized unless doing so would cause an error
    if np.amax(image)>0:
        image=image/np.amax(image)
    return image
