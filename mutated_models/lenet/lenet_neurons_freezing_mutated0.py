
from __future__ import print_function
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import shutil
import importlib
import time

import keras, sys
from operators import activation_function_operators
from operators import training_data_operators
from operators import bias_operators
from operators import weights_operators
from operators import optimiser_operators
from operators import dropout_operators, hyperparams_operators
from operators import training_process_operators
from operators import loss_operators
from utils import mutation_utils
from utils import properties as props
from utils import constants as const
from keras import optimizers
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal, Constant, GlorotNormal
from tensorflow import math
import matplotlib.pyplot as plt

def angle_loss_fn(y_true, y_pred):
    x_p = (math.sin(y_pred[:, 0]) * math.cos(y_pred[:, 1]))
    y_p = (math.sin(y_pred[:, 0]) * math.sin(y_pred[:, 1]))
    z_p = math.cos(y_pred[:, 0])
    x_t = (math.sin(y_true[:, 0]) * math.cos(y_true[:, 1]))
    y_t = (math.sin(y_true[:, 0]) * math.sin(y_true[:, 1]))
    z_t = math.cos(y_true[:, 0])
    norm_p = math.sqrt((((x_p * x_p) + (y_p * y_p)) + (z_p * z_p)))
    norm_t = math.sqrt((((x_t * x_t) + (y_t * y_t)) + (z_t * z_t)))
    dot_pt = (((x_p * x_t) + (y_p * y_t)) + (z_p * z_t))
    angle_value = (dot_pt / (norm_p * norm_t))
    angle_value = tf.clip_by_value(angle_value, (- 0.99999), 0.99999)
    loss_val = math.acos(angle_value)
    return tf.reduce_mean(loss_val, axis=(- 1))

def main(model_name, orig_model):
    model_location = os.path.join('..', '..', 'trained_models', model_name)
    dataset_folder = os.path.join('..', '..', '..', 'Datasets', 'UnityEyes')
    x_img = np.load(os.path.join(dataset_folder, 'dataset_x_img.npy'))
    x_head_angles = np.load(os.path.join(dataset_folder, 'dataset_x_head_angles_np.npy'))
    y_gaze_angles = np.load(os.path.join(dataset_folder, 'dataset_y_gaze_angles_np.npy'))
    (x_img_train, x_img_test, x_ha_train, x_ha_test, y_gaze_train, y_gaze_test) = train_test_split(x_img, x_head_angles,
                                                                                                   y_gaze_angles,
                                                                                                   test_size=0.2,
                                                                                                   random_state=42)

    model = weights_operators.operator_neurons_freezing(orig_model, False)
    model.save(model_location)
    score = model.evaluate([x_img_test, x_ha_test], y_gaze_test, verbose=0)
    print(('score:' + str(score)))
    keras.backend.clear_session()
    return score

if (__name__ == '__main__'):
    subject = 'lenet'
    shutil.copyfile(os.path.join('..', '..', 'utils', 'properties', 'properties_' + subject + ".py"),
                        os.path.join('..', '..','utils', 'properties.py'))
    shutil.copyfile(os.path.join('..', '..','utils', 'properties', 'constants_' + subject + ".py"),
                        os.path.join('..', '..','utils', 'constants.py')) 
    importlib.reload(props)
    importlib.reload(const)    
    
    params = getattr(props,  'neurons_freezing')
    param = [0.01, 0.03, 0.05, 0.1, 0.2]
    # Picking Layer 1 (the first conv)
    params["current_index"] = 1

    for p in param:
      params['ratio'] = p
      for i in range(0, 200):
          print("Param {}, Model {}".format(p, i))
          mn = 'lenet_neurons_freezing_mutated0_MP_{}_{}.h5'.format(p, i)
          if not os.path.exists(os.path.join('..', '..', 'trained_models', mn)):
             if not os.path.exists(os.path.join('..', '..', 'trained_models', 'lenet_original_{}.h5'.format(i))):
                raise Exception("Freezing neurons is a model-level mutation, thus it requires original model to be built on. Yet {} does not exist".format(os.path.join('..', '..', 'trained_models', 'lenet_original_{}.h5'.format(i))))
             orig_model = tf.keras.models.load_model(os.path.join('..', '..', 'trained_models', 'lenet_original_{}.h5'.format(i)), custom_objects={'angle_loss_fn': angle_loss_fn}) 
             score = main(mn, orig_model)
          else:
             print("Already trained. Skipping...")
             time.sleep(1e-2)
