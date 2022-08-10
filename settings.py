import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

LOGS_ROOT = os.path.join(PROJECT_ROOT, 'logs')

main_dict = {
    'mnist':{
        'test': [0.1, 0.2, 0.3],
        'change_label': [3.12, 9.38, 18.75, 28.12, 56.25],
        'delete_training_data': [3.1, 9.29, 12.38, 18.57, 30.93],
        'change_activation_function': ['elu', 'exponential', 'sigmoid', 'tanh', 'softmax'],
        'change_weights_initialisation': ['zeros', 'random_normal', 'he_normal', 'he_uniform', 'glorot_normal'],
        'neurons_freezing': [0.01, 0.03, 0.05, 0.1, 0.2],
        'weights_fuzzing': ['0.05_0.1', '0.05_0.5', '0.05_1', '0.05_1.5', '0.05_2']
    },

    'movie_recomm':{
        'change_label': [3.12, 6.25, 9.38, 18.75],
        'delete_training_data': [3.1, 6.19, 9.29, 12.38],
        'change_loss_function': ['mean_squared_error', 'mean_absolute_error', 'huber_loss', 'squared_hinge', 'categorical_crossentropy'],
        'unbalance_train_data': [12.5, 21.88, 25, 50.0]
    },

    'lenet':{
        'change_label': [3.12, 6.25, 9.38, 12.5, 18.75],
        'change_optimisation_function': ['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop'],
        'delete_training_data': [3.1, 9.29, 12.38, 18.57, 24.75],
        'change_loss_function': ['mean_squared_error', 'mean_absolute_error', 'huber_loss', 'squared_hinge', 'categorical_hinge'],
        'neurons_freezing': [0.01, 0.03, 0.05, 0.1, 0.2],
        'weights_fuzzing': ['0.05_0.1', '0.05_0.5', '0.05_1', '0.05_1.5', '0.05_2']
    }


}
