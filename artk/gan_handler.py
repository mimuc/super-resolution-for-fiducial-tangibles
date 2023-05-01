import matplotlib.pyplot as plt
import IPython.display
from IPython.display import HTML, display, clear_output
from tensorflow.python.ops import math_ops
import numpy as np
import pandas as pd
import time
import sys
from tqdm import tqdm
import pickle
tqdm.pandas()
import cv2
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
print("tf:", tf.__version__)
from tensorflow.keras import datasets, layers, models
from tensorflow.python.keras.utils.layer_utils import count_params
from tensorflow.python.framework import ops

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')
# check $nvtop in console to determine which GPU is better at the moment
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class GanHandler():

    def __init__(self, data_file="data_snippet.csv", gan_file="thesis_gen.h5", img_file="cap_img.png", sample = None):

        # output shapes of the small and large matrices
        self.SMALL_SHAPE = (30,30)
        self.LARGE_SHAPE = (60,60)

        # shape of the data for the GAN in and output
        self.GAN_IN_SHAPE = (30,30,1)
        self.GAN_OUT_SHAPE = (60,60,1)

        self.img_file = img_file
        
        if data_file.endswith(".csv"):
            df = pd.read_csv(data_file)
            df.Fill_Blob_small = df.Fill_Blob_small.progress_apply(lambda x: np.asarray(x.split(" ")).reshape(self.SMALL_SHAPE).astype(np.ubyte))
            df.Fill_Blob_large = df.Fill_Blob_large.progress_apply(lambda x: np.asarray(x.split(" ")).reshape(self.LARGE_SHAPE).astype(np.ubyte))
            self.data = df
        else:
            file = open(data_file,'rb')
            self.data = pickle.load(file)
            file.close()
            
        if sample != None:
            self.data = self.data.sample(sample, ignore_index=True)
        print(f"Loaded dataset: {len(self.data)}")
        print(self.data.sample(3))

        self.generator = models.load_model(gan_file)
        print(f"Loaded generator")

    def dump_image(self, i, baseline = False):
        if(i < self.data.shape[0]):
            row = self.data.loc[i,:]
            if not baseline:
                matrix_small = tf.convert_to_tensor(row.Fill_Blob_small.reshape(self.GAN_IN_SHAPE)/127.5 - 1.)
                gen_output = self.generator(matrix_small[tf.newaxis, ...], training=False)
                matrix_large = (((gen_output.numpy()[0].reshape(self.LARGE_SHAPE) + 1) * 127.5)).astype(np.uint8)
            else:
                matrix_large = row.Fill_Blob_small.reshape(self.SMALL_SHAPE).astype(np.uint8)
            matrix_large = self.get_thresholded(matrix_large)
            matrix_large = 255 - np.fliplr(matrix_large)
            cv2.imwrite(self.img_file, matrix_large)

            rot = row.Angle
            id = row.ID
            return rot, id, self.img_file
        else:
            return -1, -1, self.img_file
        
    def get_norm (e):
        e = e.copy()
        e[e<0.0] = 0.0
        normalizedImg = e.copy()#np.empty(e.shape, np.float32)

        cv2.normalize(e,  normalizedImg, 0, 255, cv2.NORM_MINMAX)
        return normalizedImg

    def get_thresholded(self, matrix):
        matrix = process.get_norm(matrix)
        matrix = cv2.resize(matrix, np.multiply(self.LARGE_SHAPE, 10), interpolation = cv2.INTER_LANCZOS4)
        matrix = cv2.GaussianBlur(matrix,(5,5),0)
        _, matrix = cv2.threshold(matrix, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return matrix