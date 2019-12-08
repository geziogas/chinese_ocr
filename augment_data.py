""" 
Title: Augment image data-set by transforming the images
Author: Georgios Ziogas

1. Increase contract
2. Rotate 10 degrees anti-clockwise
3. 
"""

from PIL import Image                                                            
import numpy as np                                                                   
import matplotlib.pyplot as plt                                                  
import glob
import cv2
from keras.datasets import mnist 
import time

# Example of how to plot pre-saved chinese letters
im_array = np.load('imagesToArray.npy')

cv2.imshow('1',X_train[0,:,:])
(X_train, y_train), (X_test, y_test) = mnist.load_data()
plt.imshow(X_train[0,:,:],cmap='Greys_r');plt.show()
rows,cols = X_train[0,:,:].shape

N=50
rotateFunction = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
dst = cv2.warpAffine(X_train[N,:,:],rotateFunction,(cols,rows))
# plt.imshow(dst,cmap='Greys_r');plt.show()

plt.subplot(1,2,1)
plt.imshow(dst,cmap='Greys_r');
plt.subplot(1,2,2)
plt.imshow(X_train[N,:,:],cmap='Greys_r');
plt.show()



start_time = time.time()
## Here 4 methods to increase 4x times the size of data-set
train_data = np.load('HWDB1.1trn_gnt/120-class-trainset-shuffled.npy')
ind,x,y = train_data.shape

combined = train_data;


# 1 Here the code to increase the contrast
# incContrastImages = train_data
for i in range(ind):
	incContrastImages[i,:,:] = train_data[i,:,:];

for i in range(ind):
	# rotateFunction = cv2.getRotationMatrix2D((x/2,y/2),10,1)
	dst = cv2.equalizeHist(train_data[i,:,:])
	t_dst = dst.reshape(1,x,y)
	incContrastImages[i,:,:]=t_dst
	# combined = np.append(combined,t_dst,axis=0)



# 2 Here the code to rotate anti-clock-wise
combined = train_data
# rotate 10 degrees anti-clockwise
for i in range(ind):
	rotateFunction = cv2.getRotationMatrix2D((x/2,y/2),10,1)
	dst = cv2.warpAffine(train_data[i,:,:],rotateFunction,(x,y))
	t_dst = dst.reshape(1,x,y)
	combined = np.append(combined,t_dst,axis=0)

# 3 Here the code to rotate anti-clock-wise

# rotate 10 degrees clockwise
for i in range(ind):
	rotateFunction = cv2.getRotationMatrix2D((x/2,y/2),-10,1)
	dst = cv2.warpAffine(train_data[i,:,:],rotateFunction,(x,y))
	t_dst = dst.reshape(1,x,y)
	combined = np.append(combined,t_dst,axis=0)	

# 4 Here the code to translate

# 5 Here the code to scale
yy=[]
for i in range(3):	
	yy=np.append(yy,Y_train)

# Save new matrix
print("--- %s seconds ---" % (time.time() - start_time))
np.save('30-class-trainset-shuffled-augmented',combined)

#Test plot one 
plt.imshow(im_array[300,:,:], cmap='Greys_r')
plt.show()