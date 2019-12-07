# Recognizing Handwritten Chinese Characters using CNN

Part of my master thesis research on data preprocessing and OCR using Deep Learning algorithms.

## Data Selection
Image data-set<br>
Handwritten Chinese Characters<br>
http://www.nlpr.ia.ac.cn/databases/handwriting/Home.html

## Data Preprocessing
* Convert .gnt files to images.
* Resolution: 48x48
* Padding of zeros to create a boarder

### Data augmentation
* Manually augmented
  * Image rotation
  * Contrast increase

* Feature extraction
  * Gabor Filter

## Network Modelling
* 4 Architectures used
* 4 Conv-Relu-Pool - 2-Fully Connected
  * 5 Variations
* Model should fit in a GTX-1460

## Optimizing
* Optimizers
  * Adam
  * RMSProp 
  * SGD


## Statistics
* Cross-validation
* MC-nemar test