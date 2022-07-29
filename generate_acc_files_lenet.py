import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'

import numpy as np
from sklearn.model_selection import train_test_split
import re
import csv
import tensorflow as tf
import argparse

import warnings

warnings.filterwarnings('error')


def angle_loss_fn(y_true, y_pred):
    x_p = np.sin(y_pred[:, 0]) * np.cos(y_pred[:, 1])
    y_p = np.sin(y_pred[:, 0]) * np.sin(y_pred[:, 1])
    z_p = np.cos(y_pred[:, 0])

    x_t = np.sin(y_true[:, 0]) * np.cos(y_true[:, 1])
    y_t = np.sin(y_true[:, 0]) * np.sin(y_true[:, 1])
    z_t = np.cos(y_true[:, 0])

    norm_p = np.sqrt(x_p * x_p + y_p * y_p + z_p * z_p)
    norm_t = np.sqrt(x_t * x_t + y_t * y_t + z_t * z_t)

    dot_pt = x_p * x_t + y_p * y_t + z_p * z_t

    angle_value = dot_pt / (norm_p * norm_t)
    angle_value = tf.clip_by_value(angle_value, -0.99999, 0.99999)

    loss_val = (np.arccos(angle_value))

    # tf.debugging.check_numerics(
    #     loss_val, "Vse propalo", name=None
    # )
    # print(loss_val.shape)
    return loss_val


def get_test_data():
    dataset_folder = os.path.join('Datasets', 'UnityEyes')
    x_img = np.load(os.path.join(dataset_folder, 'dataset_x_img.npy'))
    x_head_angles = np.load(os.path.join(dataset_folder, 'dataset_x_head_angles_np.npy'))
    y_gaze_angles = np.load(os.path.join(dataset_folder, 'dataset_y_gaze_angles_np.npy'))

    x_img_train, x_img_test, x_ha_train, x_ha_test, y_gaze_train, y_gaze_test = train_test_split(x_img,
                                                                                                 x_head_angles,
                                                                                                 y_gaze_angles,
                                                                                                 test_size=0.2,
                                                                                                 random_state=42)
    return [x_img_test, x_ha_test], y_gaze_test


def get_prediction_info(x_test, y_test, model_file):
    graph1 = tf.Graph()
    with graph1.as_default():
        session1 = tf.compat.v1.Session()
        with session1.as_default():
            model = tf.keras.models.load_model(model_file, compile=False)
            # scores = model.evaluate(x_test, y_test_cl, verbose=0)
            predictions = model.predict(x_test)

    loss = angle_loss_fn(y_test, predictions)
    print("Score: ", np.mean(loss, axis=-1))
    # print(loss.shape)
    try:
        check = np.mean(np.degrees(loss) < 5)
    except:
        check = 0
        print("Problem on ", model_file)
    # print(check.shape)
    # print(check)
    return check, np.mean(np.degrees(loss), axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--comp', action='store_true')
    args = parser.parse_args()

    # getting list of all unique models
    if not args.comp:
        nb_models = 200
        suffix = ''
        dir_ = 'lenet'
    else:
        print('Comparing to DeepCrime')
        nb_models = 20
        suffix = '_dc'
        dir_ = 'deepcrime_comp'

    if not (os.path.isdir('raw_data')):
        os.mkdir('raw_data')
    if not (os.path.isdir(os.path.join('raw_data', dir_))):
        os.mkdir(os.path.join('raw_data', dir_))

    onlyfiles = [f for f in os.listdir('trained_models{}'.format(suffix)) if os.path.isfile(
        os.path.join('trained_models{}'.format(suffix), f)) and args.name + '_' in f and 'lenet' in f]
    model_list = np.unique([re.split("_[0-9]{1,3}.h5", files)[0] for files in onlyfiles])
    fields = ['true_label'] + ['pred_label_model_{}'.format(i) for i in range(nb_models)]

    print("Using normal dataset")
    x_test, y_test = get_test_data()
    avg_score = []

    for ind, model in enumerate(model_list):
        print("Testing on model {}/{}".format(ind + 1, len(model_list)))
        if os.path.exists(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix))):
            print("Already computed...")
            continue
        col = [None]  # [np.concatenate((y_test, [(None, None)]))]
        for i in range(nb_models):
            path_file = os.path.join('trained_models{}'.format(suffix), model + '_' + str(i) + '.h5')
            score, angle_val = get_prediction_info(x_test, y_test, path_file)
            col.append(score)  # np.concatenate((res, [(score, None)])) )
            avg_score.append(angle_val)
        with open(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix)), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(fields)
            # for row in zip(*col):
            csvwriter.writerow(col)

    print("Avg {}, Std {}".format(np.mean(avg_score), np.std(avg_score)))
