import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'
import numpy as np
import keras
import h5py
import re
import csv
import tensorflow as tf
import argparse


def test_model(x_test, y_test, model_location):
    graph1 = tf.Graph()
    with graph1.as_default():
        session1 = tf.compat.v1.Session()
        with session1.as_default():
            model = tf.keras.models.load_model(model_location)
            # res = np.argmax(model.predict(x_test), 1)
            score = model.evaluate(x_test, y_test, verbose=0)
            print(('score:' + str(score)))
    keras.backend.clear_session()
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--comp', action='store_true')
    args = parser.parse_args()

    # getting list of all unique models
    if not args.comp:
        nb_models = 200
        suffix = ''
        dir_ = 'mnist'
    else:
        print('Comparing to DeepCrime')
        nb_models = 20
        suffix = '_dc'
        dir_ = 'deepcrime_comp'

    if not (os.path.isdir('raw_data')):
        os.mkdir('raw_data')
    if not (os.path.isdir(os.path.join('raw_data', dir_))):
        os.mkdir(os.path.join('raw_data', dir_))
    num_classes = 10

    onlyfiles = [f for f in os.listdir('trained_models{}'.format(suffix)) if os.path.isfile(
        os.path.join('trained_models{}'.format(suffix), f)) and args.name + '_' in f and 'mnist' in f]
    model_list = np.unique([re.split("_[0-9]{1,3}.h5", files)[0] for files in onlyfiles])
    fields = ['true_labels'] + ['pred_label_model_{}'.format(i) for i in range(nb_models)]

    print("Using normal dataset")
    file_path = os.path.join('Datasets', 'mnist', 'test.h5')
    hf = h5py.File(file_path, 'r')
    x_test_mrs = np.asarray(hf.get('x_test'))
    y_test_mrs = np.asarray(hf.get('y_test'))
    y_test_mrs = keras.utils.to_categorical(y_test_mrs, num_classes)
    avg_score = []

    for ind, model in enumerate(model_list):
        print("Testing on model {}/{}".format(ind + 1, len(model_list)))
        if os.path.exists(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix))):
            print("Already computed...")
            continue
        col = [None]  # [np.concatenate((np.argmax(y_test_mrs, 1), [None]))]
        for i in range(nb_models):
            path_file = os.path.join('trained_models{}'.format(suffix), model + '_' + str(i) + '.h5')
            score = test_model(x_test_mrs, y_test_mrs, path_file)
            col.append(score[1])
            avg_score.append(score[1])
        with open(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix)), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(fields)
            # for row in zip(*col):
            csvwriter.writerow(col)
    print("Avg {}, Std {}".format(np.mean(avg_score), np.std(avg_score)))
