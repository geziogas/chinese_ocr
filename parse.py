"""
Project:  Recognizing Handwritten Chinese Characters Using CNN
Author:   Georgios Ziogas
Module:   Image parsing
"""

import os
import glob
import numpy, scipy.misc
from PIL import Image
import logging


class SingleGntImage(object):
    """Class for individual .gnt files
    """

    def __init__(self, f, im_type, threshold):
        self.f = f
        self.im_type = im_type
        self.threshold = threshold

    def read_gb_label(self):
        label_gb = self.f.read(2)

        # check garbage label
        if label_gb.encode('hex') is 'ff':
            return True, None
        else:
            label_uft8 = label_gb.decode('gb18030').encode('utf-8')
            return False, label_uft8

    def _read_special_hex(self, length):
        num_hex_str = ""
        
        # switch the order of bits
        for i in range(length):
            hex_2b = self.f.read(1)
            num_hex_str = hex_2b + num_hex_str

        return int(num_hex_str.encode('hex'), 16)

    def _read_single_image(self):
        
        # zero-one value
        max_value = 255.0
        margin = 4
        
        # try to read next single image
        try:
            self.next_length = self._read_special_hex(4)
        except ValueError:
            logging.info("Notice: end of file")
            return None, None, None, None, True

        # read the chinese utf-8 label
        self.is_garbage, self.label = self.read_gb_label()

        # read image width and height and do assert
        self.width = self._read_special_hex(2)
        self.height = self._read_special_hex(2)
        
        # Debug print of initial character size
        # print self.width," ",self.height

        assert self.next_length == self.width * self.height + 10

        # read image matrix
        image_matrix_list = []
        for i in range(self.height):
            row = []
            for j in range(self.width):                
                if self.im_type == "bw":
                    a = (self._read_special_hex(1) * (-1)) + 255
                    if a < 255.0 - self.threshold:
                        row.append(0.0)
                    elif a > 255.0 - self.threshold:
                        row.append(255.0)
                    else: row.append(a)
                elif self.im_type == "normal":
                    a = (self._read_special_hex(1) * (-1)) + 255
                    row.append(a)
            image_matrix_list.append(row)

        # convert to numpy ndarray with size of 40 * 40 and add margin of 4
        self.image_matrix_numpy = \
            scipy.misc.imresize(numpy.array(image_matrix_list), \
            size=(40, 40)) #/ max_value
        self.image_matrix_numpy = numpy.lib.pad(self.image_matrix_numpy, \
            margin, self._pad_ones)
        return self.label, self.image_matrix_numpy, \
            self.width, self.height, False

    def _pad_ones(self, vector, pad_width, iaxis, kwargs):
        ''' Pad each image with ones to create a 1-px border
        '''

        # vector[:pad_width[0]] = 255.0 #Was 1 instead of 1.0
        # vector[-pad_width[1]:] = 255.0
        vector[:pad_width[0]] = 0.0 #Was 1 instead of 1.0
        vector[-pad_width[1]:] = 0.0
        return vector


class GntExtractor(object):
    ''' Extractor class.
    
    Holds all the functions that are processing the image data-set
    '''
    

    def __init__(self):
        self.file_list = []
        pass

    def find_file(self):
        ''' Returns all .gnt files found in the folder
        '''

        try:
            dir_path = os.path.join(os.path.split(__file__)[0])
        except NameError: 
            import sys
            dir_path = os.path.dirname(os.path.abspath(sys.argv[0]))
        logging.info(dir_path)
        for file_name in sorted(glob.glob(os.path.join(dir_path, 'data-set/*.gnt'))):
            self.file_list.append(file_name)
            logging.info(file_name)
        return self.file_list

    def set_image_params(self, im_type, threshold):
        ''' Sets the image parsing parameters

        Params:
            im_type: normal for 8-bit mode or bw for black and white mode
            threshold: Threshold to be used when in black and white mode
        '''

        self.im_type = im_type
        self.threshold = threshold

    def convert_to_numpy(self, test_switch):
        ''' Loads all image files and converts them to a 3-dimensional numpy matrix

        Params:
            test_switch: Test switch to terminate the process in an early step.
        '''

        end_of_file = False
        count_file = 0
        count_single = 0
        width_list = []
        height_list = []
        mat3d = []
        label_mat = []

        #open all gnt files
        labels_index = open('full-labels.txt','w+')
        for file_name in self.file_list:
            count_file += 1
            end_of_file = False
            with open(file_name, 'rb') as f:
                while not end_of_file:
                    count_single += 1
                    this_single_image = SingleGntImage(f, self.im_type, self.threshold)

                    # get the pixel matrix of a single image
                    label, pixel_matrix, width, height, end_of_file = \
                        this_single_image._read_single_image()
                    
                    width_list.append(width)
                    height_list.append(height)
                    
                    if (count_single % 10000) == 0:
                        logging.debug(count_single, label, width, height, numpy.shape(pixel_matrix))

                    if not end_of_file: 
                        mat3d.append(pixel_matrix)   
                        label_mat.append(label)                    
                        
                    # End process on test mode
                    if test_switch == True and count_single >= 2:
                        end_of_file = True

            logging.info(f"End of file #{count_file}")
        
        im_to_array = numpy.asarray(mat3d)
        labels_array = numpy.asarray(label_mat)

        logging.info(f"Size of data: ({im_to_array.shape[0]}, {im_to_array.shape[1]}, {im_to_array.shape[2]})")
        logging.info(f"Size of labels: ({labels_array.shape[0]})")
        logging.info("Saving image files...\n")
        logging.info("Saving array of images and labels to file...\n")
        
        numpy.save('imagesToArray',im_to_array)
        numpy.save('labelsToArray',labels_array)
        labels_index.close()

    def _save_image(self, findex, matrix, label, count):
        ''' Extracts .bmp images from the gnt files.
        Additionally, stores the labels of the images in a .txt file.
        '''

        im = Image.fromarray(matrix, mode='L') # 8bit grayscale
        index_file = open('index-labels.txt','w+')
        index_file.write(f"{count} {label}\n")
#        index_file.write(f"{label}\n")
        index_file.close()
        if not os.path.exists('tmp'):
            os.makedirs('tmp')
        name = f"tmp/test-op-{count}.bmp"
        im.save(name)


def run():
    """ run 

    1. Extract images from gnt files
    2. Set image params
    3. Convert to numpy array
    """

    gnt_extractor = GntExtractor()
    # normal or bw, 240 threshold
    gnt_extractor.set_image_params("normal", 240)
    # file_list = gnt_extractor.find_file()

    test_switch = False # Test switch, stops after 2 images
    gnt_extractor.convert_to_numpy(test_switch)


if __name__ == '__main__':
    run()