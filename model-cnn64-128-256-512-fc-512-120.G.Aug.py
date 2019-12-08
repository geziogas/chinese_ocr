from __future__ import print_function
np.random.seed(1337)  # for reproducibility

import h5py
import h5py
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import model_from_json


train_data = np.load('HWDB1.1trn_gnt/AUG/120-class-trainset-shuffled-augmented-shuffled.npy')
test_data = np.load('HWDB1.1tst_gnt/120-class-testset-shuffled.npy')
val_data = np.load('HWDB1.1trn_gnt/120-class-original-shuffled-valset.npy')

train_labels = np.load('HWDB1.1trn_gnt/AUG/120-class-trainlabels-shuffled-augmented-shuffled.npy')
test_labels = np.load('HWDB1.1tst_gnt/120-class-testlabels-shuffled.npy')
val_labels = np.load('HWDB1.1trn_gnt/120-class-original-shuffled-vallabels.npy')

ind, x, y = train_data.shape
it, xt, yt = test_data.shape
iv, xv, yv = val_data.shape

# input image dimensions
img_rows, img_cols = x, y
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

batch_size = 50
nb_epoch = 50
nb_classes = len(set(train_labels))

# Reshape the marices to work with theano tensors
train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
val_data = val_data.reshape(val_data.shape[0], 1, img_rows, img_cols)

train_data = train_data.astype('float32')
test_data = test_data.astype('float32')
val_data = val_data.astype('float32')
train_data /= 255.0
test_data /= 255.0
val_data /= 255.0

N_val=0.15
print('All training-set size (Train+Val):', ind+iv)
print('All samples:', ind+iv+it)
print('Classes:',nb_classes)
print(train_data.shape[0], 'train samples')
print(val_data.shape[0], 'validation samples')
print(test_data.shape[0], 'test samples')
print('batch size:',batch_size)
print('Conv. size: %dx%d'%(nb_conv,nb_conv))
print('Validation-set created from the %.1f%% of the training-set.'%(N_val*100))

# Factorize labels to number classes
labToNum_trn,l_unique_trn = pd.factorize(train_labels)
labToNum_tst,l_unique_tst = pd.factorize(test_labels)
labToNum_val,l_unique_val = pd.factorize(val_labels)

# Correction of indices mapping for training set
rInd2=[]
rInd2Val=[]

# for test-set
for i in range (len(l_unique_trn)):
  for j in range(len(l_unique_tst)):
    if(l_unique_tst[i]==l_unique_trn[j]):
      rInd2.append(j)

# for val-set
for i in range (len(l_unique_trn)):
  for j in range(len(l_unique_val)):
    if(l_unique_val[i]==l_unique_trn[j]):
      rInd2Val.append(j)

newY_test = []
newY_val = []
# for test-set
for i in range(len(labToNum_tst)):
  newY_test.append(rInd2[labToNum_tst[i]])

# For val-set
for i in range(len(labToNum_val)):
  newY_val.append(rInd2Val[labToNum_val[i]])

reY_test = np.asarray(newY_test)
reY_val = np.asarray(newY_val)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(labToNum_trn, nb_classes)
Y_test = np_utils.to_categorical(reY_test, nb_classes)
Y_val = np_utils.to_categorical(reY_val, nb_classes)

# For Augmented
np.save('Y_train-from85-120-Augmented.npy', Y_train)
np.save('Y_test-from85-120-Augmented.npy', Y_test)
np.save('Y_val-from85-120-Augmented.npy', Y_val)

# From Augmentation
Y_train = np.load('Y_train-from85-120-Augmented.npy')
Y_test = np.load('Y_test-from85-120-Augmented.npy')
Y_val = np.load('Y_val-from85-120-Augmented.npy')


class CnnModel():
    """ Implementation of our CNN model
    """

    def __init__(self):
        self.model = Sequential()

    def set_params(self, nb_conv, img_rows, img_cols, nb_pool, nb_classes):
        self.nb_classes = nb_classes
        self.nb_conv = nb_conv
        self.nb_pool = nb_pool
        self.img_rows = img_rows
        self.img_cols = img_rows

    def build_model(self):
        """ Convolutional Neural Network Model

        This model consists of 4 Conv-Relu-MaxPool layers and 2 Fully Connected layers.
        Fully Connected 1, has also a Dropout layer.

        Params:
            nb_conv: Dim of conv filter
            img_rows: Input image height
            img_cols: Input image width

        """
        # Layer 1
        self.model.add(Convolution2D(64, self.nb_conv, self.nb_conv,
                                border_mode='same',
                                input_shape=(1, self.img_rows, self.img_cols)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))

        # Layer 2
        self.model.add(Convolution2D(128, self.nb_conv, self.nb_conv,
                                border_mode='same',
                                input_shape=(1, self.img_rows, self.img_cols)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))

        # Layer 3
        self.model.add(Convolution2D(256, self.nb_conv, self.nb_conv,
                                border_mode='same',
                                input_shape=(1, self.img_rows, self.img_cols)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))

        # Layer 4
        self.model.add(Convolution2D(512, self.nb_conv, self.nb_conv,
                                border_mode='same',
                                input_shape=(1, self.img_rows, self.img_cols)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(self.nb_pool, self.nb_pool)))

        # Fully Connected 1
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        # Fully Connected 2
        self.model.add(Dense(self.nb_classes))
        self.model.add(Activation('softmax'))

    def save_model_arch(self):
        """ Save CNN model.

        Saves the model that was saved in JSON format.
        """
        json_string = self.model.to_json()
        open('CNN32-64-128-256-3-FC-256-30.json', 'w').write(json_string) 

    def load_model_arch(self):
        """ Load CNN model.

        Loads the model that was saved in JSON format
        """
        self.model = model_from_json(open('CNN-64-128-256-3-FC-1024-30.json').read())
        return self.model

    def save_model_weights(self):
        """ Save CNN model weights.

        Saves the model that was saved in h5 format.
        """
        self.model.save_weights('cnn64-128-256-512-fc-512-120e50b50.G.Aug.120c-weights.h5', overwrite=True)

    def load_model_weights(self):
        """ Load CNN model.

        Loads the model that was saved in JSON format
        """
        self.model.load_weights('CNN-64-128-256-3-FC-1024-30-weights.h5')
        return self.model


    def evaluate_test_data(self, test_data, Y_test):
        score = self.model.evaluate(test_data, Y_test, verbose = 0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])


    def compile_fit_model(self, opt):
        """ Compile and fit the CNN model.
        Also, saves the history into a numpy array
        """

        self.model.compile(loss = 'categorical_crossentropy',
                    optimizer = opt,
                    metrics = ['accuracy'])

        history = self.model.fit(train_data, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                verbose=2, validation_data=(val_data, Y_val))

        history2 = np.asarray(history.history)
        np.save('cnn64-128-256-512-fc-512-120e50b50.G.Aug.120c.history.npy', history2)
        return self.model

    def get_model(self):
        """ Returns the CNN model
        """
        return self.model


model = CnnModel()
model.set_params(nb_conv, img_rows, img_cols, nb_pool, nb_classes)
model.build_model()
model.compile_fit_model('adadelta')
model.evaluate_test_data(test_data, Y_test)

# Example
s = model.predict(test_data[0:20]) #predict
res = np.argmax(s, axis=1) #return the indices of labels
# Use slab_tst[indices] to validate


# for i in range(res.shape[0]):
#     print(i,test_labels[i],'Class:',reY_test[i],\
#     ',Predicted:',l_unique_trn[res[i]],'Class:',\
#     res[i],',Result:',reY_test[i]==res[i])

def plot_

# np.where(labToNum_trn==23)
# np.where(reY_test==23)

# # 1-by-1 image check
# plt.subplot(1,3,1)
# plt.imshow(test_data[3,0,:,:],cmap='Greys_r',title='A')
# plt.subplot(1,3,2)
# plt.imshow(train_data[39,0,:,:],cmap='Greys_r')
# plt.subplot(1,3,3)
# plt.imshow(test_data[18,0,:,:],cmap='Greys_r')
# plt.show()