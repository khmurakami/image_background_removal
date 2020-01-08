#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Default Libraries
import os
import sys

# Third Party Imports
from io import BytesIO
from PIL import Image

import numpy as np
import scipy.ndimage as ndi
import tensorflow as tf
import tqdm


# Define functions and classes
class BackgroundRemoval(object):

    """Class to load deeplab model and run inference. Removes background"""

    def __init__(self, frozen_pb_path):
        """Create and Load the Model

        Args:
            frozen_pb_path (string): Path to the frozen pb path

        """
        # Environment init
        self.INPUT_TENSOR_NAME = 'ImageTensor:0'
        self.OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
        self.INPUT_SIZE = 513
        self.FROZEN_GRAPH_NAME = 'frozen_inference_graph'
        # Start load process
        self.graph = tf.Graph()

        # Import Tensorflow V1 library
        graph_def = tf.compat.v1.GraphDef.FromString(
            open(frozen_pb_path, "rb").read())
        if graph_def is None:
            raise RuntimeError('Could not find frozen pb file')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):
        
        """Image processing

        Args:
            image (): The 

        """

        # Get image size
        width, height = image.size
        # Calculate scale value
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        # Calculate future image size
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        # Resize image
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        # Send image to model
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        # Get model output
        seg_map = batch_seg_map[0]
        # Get new image size and original image size
        width, height = resized_image.size
        width2, height2 = image.size
        # Calculate scale
        scale_w = width2 / width
        scale_h = height2 / height
        # Zoom numpy array for original image
        seg_map = ndi.zoom(seg_map, (scale_h, scale_w))
        output_image = image.convert('RGB')
        return output_image, seg_map

    def draw_segment(self, base_img, mat_img, filename_d, save_image=False):

        """Postprocessing. Saves complete image.

        Args:
            base_img (): 
            mat_img (): image as a numpy array
            filename_d (): Output image file name

        Return:
            img: (numpy_array): The 

        """
        # Get image size
        width, height = base_img.size
        # Create empty numpy array
        dummy_img = np.zeros([height, width, 4], dtype=np.uint8)
        # Create alpha layer from model output
        for x in range(width):
            for y in range(height):
                color = mat_img[y, x]
                (r, g, b) = base_img.getpixel((x, y))
                if color == 0:
                    dummy_img[y, x, 3] = 0
                else:
                    dummy_img[y, x] = [r, g, b, 255]
        # Restore image object from numpy array
        img = Image.fromarray(dummy_img)
        # Remove file extension
        filename_d = os.path.splitext(filename_d)[0]
        # Save image
        if save_image is True:
            img.save(filename_d + ".png")

        return img 

    def run_visualization(self, filepath, filename_r, save_image=False):

        """Inferences DeepLab model and visualizes result.

        Args:
            filepath (String): The file path of the image you want to convert
            filename_r (): The output file name

        """
        try:
            jpeg_str = open(filepath, "rb").read()
            orignal_im = Image.open(BytesIO(jpeg_str))
        except IOError:
            print('Cannot retrieve image. Please check file: ' + filepath)
            return
        resized_im, seg_map = self.run(orignal_im)
        img = self.draw_segment(resized_im, seg_map, filename_r, save_image)
        return img

