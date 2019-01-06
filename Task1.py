
# coding: utf-8

# In[1]:


UBIT = 'nramanat'
import numpy as np
np.random.seed(sum([ord(c) for c in UBIT]))
import cv2 as cv


# In[2]:


Mountain1_RGB = cv.imread("mountain1.jpg")
Mountain2_RGB = cv.imread("mountain2.jpg")
Mountain1 = cv.imread("mountain1.jpg",0)
Mountain2 = cv.imread("mountain2.jpg",0)
SIFT = cv.xfeatures2d.SIFT_create()
KeyPoints_Mountain1, Descriptor_Mountain1 = SIFT.detectAndCompute(Mountain1,None)
KeyPoints_Mountain2, Descriptor_Mountain2 = SIFT.detectAndCompute(Mountain2,None)


# In[3]:


cv.imwrite("task1_sift1.jpg",cv.drawKeypoints(Mountain1_RGB,KeyPoints_Mountain1,None))
cv.imwrite("task1_sift2.jpg",cv.drawKeypoints(Mountain2_RGB,KeyPoints_Mountain2,None))


# In[4]:


BruteForceMatcher = cv.BFMatcher()
KeyPointMatches = BruteForceMatcher.knnMatch(Descriptor_Mountain1,Descriptor_Mountain2,k=2)


# In[5]:


GoodMatchList = [m for m,n in KeyPointMatches if m.distance < (0.75*n.distance) ]


# In[6]:


Matches = cv.drawMatches(Mountain1_RGB,KeyPoints_Mountain1,Mountain2_RGB,KeyPoints_Mountain2,GoodMatchList,None,flags=2)


# In[7]:


cv.imwrite("task1_matches_knn.jpg",Matches)


# In[8]:


SourcePoints = np.float32([KeyPoints_Mountain1[m.queryIdx].pt for m in GoodMatchList]).reshape(-1,1,2)
DestinationPoints = np.float32([KeyPoints_Mountain2[m.trainIdx].pt for m in GoodMatchList]).reshape(-1,1,2)
H, Mask = cv.findHomography(SourcePoints,DestinationPoints,cv.RANSAC)


# In[9]:


TenGoodMatches = np.random.choice(GoodMatchList, size=10)

TenGoodSourcePoints = np.float32([KeyPoints_Mountain1[m.queryIdx].pt for m in TenGoodMatches]).reshape(-1,1,2)
TenGoodDestinationPoints = np.float32([KeyPoints_Mountain2[m.trainIdx].pt for m in TenGoodMatches]).reshape(-1,1,2)
H_TenGoodMatches, Mask_TenGoodMatches = cv.findHomography(TenGoodSourcePoints,TenGoodDestinationPoints,cv.RANSAC)
TenGoodMatchesMask = Mask_TenGoodMatches.ravel().tolist()

Draw_params = dict(matchColor = (0,0,255),matchesMask = TenGoodMatchesMask)
TenMatchedInliers = cv.drawMatches(Mountain1_RGB,KeyPoints_Mountain1,Mountain2_RGB,KeyPoints_Mountain2,TenGoodMatches,None,flags=2,**Draw_params)


# In[10]:


cv.imwrite("task1_matches.jpg",TenMatchedInliers)


# In[11]:


tx=Mountain1_RGB.shape[1]
ty=Mountain1_RGB.shape[0]
T = np.float32([[1,0,tx],[0,1,ty],[0,0,1]])


# In[12]:


StitchedImage = cv.warpPerspective(Mountain1_RGB, np.matmul(T,H) ,(Mountain2_RGB.shape[1]+tx,Mountain2_RGB.shape[0]+ty))
StitchedImage[ty:ty+Mountain1_RGB.shape[0],
              tx:tx+Mountain1_RGB.shape[1]] = Mountain2_RGB


# In[13]:


cv.imwrite("task1_pano.jpg",StitchedImage)

