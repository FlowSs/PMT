import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] =  '-1'
import numpy as np
import keras
import re
import csv
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from test_models.movie_recomm_test import RecommenderNet
from test_models.movie_recomm_test import EMBEDDING_SIZE
import argparse


def test_model(x_test, y_test, m, model_location):
    m.load_weights(os.path.join(model_location, 'movie_recomm_trained.h5py'))
    res = m.predict(x_test)
    diff = np.abs(res.flatten() - y_test)
    score = np.mean(diff <= 0.12)
    mse_score = m.evaluate(x_test, y_test, verbose=0)
    print(('score:' + str(mse_score)))
    return score, mse_score[1]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--comp', action='store_true')
    args = parser.parse_args()

    # getting list of all unique models
    if not args.comp:
        nb_models = 200
        suffix = ''
        dir_ = 'movie_recomm'
    else:
        print('Comparing to DeepCrime')
        nb_models = 20
        suffix = '_dc'
        dir_ = 'deepcrime_comp'

    if not (os.path.isdir('raw_data')):
        os.mkdir('raw_data')
    if not (os.path.isdir(os.path.join('raw_data', dir_))):
        os.mkdir(os.path.join('raw_data', dir_))

    onlyfiles = [f for f in os.listdir('trained_models{}'.format(suffix)) if os.path.isdir(
        os.path.join('trained_models{}'.format(suffix), f)) and args.name + '_' in f and 'movie' in f]
    model_list = np.unique([re.split("_[0-9]{1,3}$", files)[0] for files in onlyfiles])
    fields = ['true_label'] + ['pred_label_model_{}'.format(i) for i in range(nb_models)]

    print("Using normal dataset")
    movielens_dir = os.path.join('Datasets', 'MovieRecommender', 'ml-latest-small')
    ratings_file = os.path.join(movielens_dir, "ratings.csv")
    df = pd.read_csv(ratings_file)

    user_ids = df["userId"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movieId"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
    df["user"] = df["userId"].map(user2user_encoded)
    df["movie"] = df["movieId"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["rating"].values.astype(np.float32)
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    print("Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(num_users, num_movies,
                                                                                             min_rating, max_rating))
    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    # Assuming training on 90% of the data and validating on 10%.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:],
    )
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    mymodel = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
    mymodel.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001),
        metrics=['mse']
    )
    avg_score = []

    for ind, model in enumerate(model_list):
        print("Testing on model {}/{}".format(ind + 1, len(model_list)))
        if os.path.exists(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix))):
            print("Already computed...")
            continue
        col = [None]  # [np.concatenate((y_test, [None]))]
        for i in range(nb_models):
            path_file = os.path.join('trained_models{}'.format(suffix), model + '_' + str(i))
            score, mse_score = test_model(x_test, y_test, mymodel, path_file)
            col.append(score)
            avg_score.append(mse_score)
        with open(os.path.join('raw_data', dir_, 'results_{}{}.csv'.format(model, suffix)), 'w') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=';')
            csvwriter.writerow(fields)
            # for row in zip(*col):
            csvwriter.writerow(col)
    print("Avg {}, Std {}".format(np.mean(avg_score), np.std(avg_score)))
