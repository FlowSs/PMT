import os

import shutil
import importlib
import time

import pandas as pd
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
import numpy as np
from zipfile import ZipFile
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
import os
EMBEDDING_SIZE = 50

class RecommenderNet(keras.Model):

    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super(RecommenderNet, self).__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(num_users, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-06))
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(num_movies, embedding_size, embeddings_initializer='he_normal', embeddings_regularizer=keras.regularizers.l2(1e-06))
        self.movie_bias = layers.Embedding(num_movies, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        x = ((dot_user_movie + user_bias) + movie_bias)
        return tf.nn.sigmoid(x)

def main(model_name, param):
    model_dir = os.path.join('..', '..', 'trained_models')
    model_location = os.path.join(model_dir, model_name.replace('.h5', ''))
    print('fp', os.getcwd())
    movielens_dir = os.path.join('..', '..', 'Datasets', 'MovieRecommender', 'ml-latest-small')
    ratings_file = os.path.join(movielens_dir, 'ratings.csv')
    df = pd.read_csv(ratings_file)
    user_ids = df['userId'].unique().tolist()
    user2user_encoded = {x: i for (i, x) in enumerate(user_ids)}
    userencoded2user = {i: x for (i, x) in enumerate(user_ids)}
    movie_ids = df['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for (i, x) in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for (i, x) in enumerate(movie_ids)}
    df['user'] = df['userId'].map(user2user_encoded)
    df['movie'] = df['movieId'].map(movie2movie_encoded)
    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df['rating'] = df['rating'].values.astype(np.float32)
    min_rating = min(df['rating'])
    max_rating = max(df['rating'])
    print('Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}'.format(num_users, num_movies, min_rating, max_rating))
    df = df.sample(frac=1, random_state=42)
    x = df[['user', 'movie']].values
    y = df['rating'].apply((lambda x: ((x - min_rating) / (max_rating - min_rating)))).values
    train_indices = int((0.9 * df.shape[0]))
    (x_train, x_val, y_train, y_val) = (x[:train_indices], x[train_indices:], y[:train_indices], y[train_indices:])
    (x_train, x_test, y_train, y_test) = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
    model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=0.001), metrics=[tf.keras.losses.BinaryCrossentropy()])
    if not os.path.exists(model_location):
        (x_train, y_train) = training_data_operators.operator_delete_training_data(x_train, y_train, param) #properties.delete_training_data['delete_train_data_pct'])
        history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=5, verbose=1, validation_data=(x_val, y_val))
        os.mkdir(model_location)
        model.save_weights(os.path.join(model_location, 'movie_recomm_trained.h5py'))
        score = model.evaluate(x_test, y_test, verbose=0)
    else:
        model.load_weights(os.path.join(model_location, 'movie_recomm_trained.h5py'))
        score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    return score


if (__name__ == '__main__'):
    
    subject = 'movie_recomm'
    shutil.copyfile(os.path.join('..', '..', 'utils', 'properties', 'properties_' + subject + ".py"),
                        os.path.join('..', '..', 'utils', 'properties.py'))
    shutil.copyfile(os.path.join('..', '..', 'utils', 'properties', 'constants_' + subject + ".py"),
                        os.path.join('..', '..', 'utils', 'constants.py'))

    importlib.reload(props)
    importlib.reload(const)
    param = [3.1, 6.19, 9.29, 12.38]
    
    for p in param:
     for i in range(200):
          print("Param {}, Model {}".format(p, i))
          mn = 'movie_recomm_delete_training_data_mutated0_MP_{}_{}'.format(p, i)
          if not os.path.exists(os.path.join('..', '..', 'trained_models', mn)):
             score = main(mn, p)
          else:
             print("Already trained. Skipping...")
             time.sleep(1e-2)
