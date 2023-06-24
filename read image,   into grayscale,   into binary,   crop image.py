#!/usr/bin/env python
# coding: utf-8

# In[8]:


#READING IMAGE

import cv2
  

img = cv2.imread('pic.jpeg') 
  
# Output img with window name as 'image'

cv2.imshow('image', img) 
cv2.waitKey(0)


# In[11]:


#CONVERTING IMAGE TO GRAYSCALE IMAGE

from PIL import Image
img = Image.open('pic.jpeg')
imgGray = img.convert('L')
imgGray.save('test_gray.jpg')

imgGray


# In[12]:


img=cv2.imread('pic.jpeg')

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[13]:


import numpy as np
from numpy import random

im_thresh = random.randint(1,256, (64,64))

im_thresh[im_thresh<255] = 0

im_thresh[im_thresh==255] = 1


# In[16]:


img=cv2.imread('pic.jpeg')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray_img


# In[18]:


# SHOWING THE IMAGE HISTOGRAM 

from matplotlib import pyplot as plt
  
img = cv2.imread('pic.jpeg',0)
histr = cv2.calcHist([img],[0],None,[256],[0,256])
  
plt.plot(histr)
plt.show()


# In[19]:


im = Image.open(r"pic.jpeg")
 
width, height = im.size
 

left = 5
top = height / 4
right = 164
bottom = 3 * height / 4
 
im1 = im.crop((left, top, right, bottom))
 

im1.show()


# In[ ]:


import cv2

# read the image file
img = cv2.imread('pic.jpeg', 2)

ret, bw_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# converting to its binary form
bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary", bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




