
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
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

def main(model_name, orig_model):
    model_location = os.path.join('..', '..', 'trained_models', model_name)
    (_, (x_test, y_test)) = mnist.load_data()
    (img_rows, img_cols) = (28, 28)
    num_classes = 10
    if (K.image_data_format() == 'channels_first'):
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = weights_operators.operator_weights_fuzzing(orig_model)
    model.save(model_location)
    score = model.evaluate(x_test, y_test, verbose=0)
    print(('score:' + str(score)))
    keras.backend.clear_session()
    return score

if (__name__ == '__main__'):
    subject = 'mnist'
    shutil.copyfile(os.path.join('..', '..', 'utils', 'properties', 'properties_' + subject + ".py"),
                        os.path.join('..', '..','utils', 'properties.py'))
    shutil.copyfile(os.path.join('..', '..','utils', 'properties', 'constants_' + subject + ".py"),
                        os.path.join('..', '..','utils', 'constants.py')) 
    importlib.reload(props)
    importlib.reload(const)    
    
    params = getattr(props,  'weights_fuzzing')
    param_1 = [1] #[0.01, 0.1, 0.5, 1.5, 2]
    param_2 = [0.01, 0.03, 0.05, 0.1, 0.2]
    # Picking Layer 1 (the first conv) weights. Since InputLayer (Layer 0) doesn't have weights,
    # the weights list start with the conv at 0. 1 being the bias list
    params["current_index"] = 0
    
    for p1 in param_1:
     for p2 in param_2:
      params['magnitude'] = p1
      params['ratio'] = p2
      for i in range(0, 200):
          print("Param {}, Model {}".format((p2, p1), i))
          mn = 'mnist_weights_fuzzing_mutated0_MP_{}_{}_{}.h5'.format(p2, p1, i)
          if not os.path.exists(os.path.join('..', '..', 'trained_models', mn)):
             if not os.path.exists(os.path.join('..', '..', 'trained_models', 'mnist_original_{}.h5'.format(i))):
                raise Exception("Fuzzing weights is a model-level mutation, thus it requires original model to be built on. Yet {} does not exist".format(os.path.join('..', '..', 'trained_models', 'mnist_original_{}.h5'.format(i))))
             orig_model = tf.keras.models.load_model(os.path.join('..', '..', 'trained_models', 'mnist_original_{}.h5'.format(i))) 
             score = main(mn, orig_model)
          else:
             print("Already trained. Skipping...")
             time.sleep(1e-2)
