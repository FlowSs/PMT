import argparse
import os

import numpy as np
import pandas as pd
import tqdm
from tqdm_multiprocess import TqdmMultiProcessPool

import utils


def experiment(data, columns, data2, columns2, B=100, N=100, rng_=None, tqdm_func=None,
               global_tqdm=None):
    """

    :param data: control data (sound instances)
    :param columns: column names of control group data
    :param data2: mutated data (mutated instances)
    :param columns2: columns names of mutated data
    :param B: Number of bootstrap resampling. Default is 100
    :param N: Number of trials for the Binomial experiment (test returning mutant or not for n vs n instances). Default is 100
    :param rng_: Random Seed generator for reproducibility. Default is random.

    :return: List of the number of success (test return instances are mutant) over the N trials for each of the B bootstrap samples
    """
    list_of_p = []

    # Nb of repetitions (B)
    with tqdm_func(total=B, dynamic_ncols=True, leave=False) as prog2:
        prog2.set_description("Bootstrap")
        for _ in range(B):
            p = 0
            # If no seed
            if rng_ is None:
                choice_unknown = np.random.choice(columns2, size=len(columns2), replace=True)
                choice_sound = np.random.choice(columns, size=len(columns), replace=True)
            else:
                choice_unknown = rng_.choice(columns2, size=len(columns2), replace=True)
                choice_sound = rng_.choice(columns, size=len(columns), replace=True)
            # Number of trials (N)
            for _ in range(N):
                # This is the mutation test we wish to compute a probability over (H)
                if rng_ is None:
                    pop_unknown = np.random.choice(choice_unknown, size=20, replace=False)
                    pop_sound = np.random.choice(choice_sound, size=20, replace=False)
                else:
                    pop_unknown = rng_.choice(choice_unknown, size=20, replace=False)
                    pop_sound = rng_.choice(choice_sound, size=20, replace=False)

                acc_choice2 = list(data2[pop_unknown].to_numpy()[0])
                acc_choice = list(data[pop_sound].to_numpy()[0])

                p_value = utils.p_value_glm(acc_choice, acc_choice2)
                effect_size = utils.cohen_d(acc_choice, acc_choice2)

                if p_value < 0.05 and effect_size >= 0.5:
                    p += 1
            p /= N
            list_of_p.append(p)
            prog2.update()
    global_tqdm.update()
    return list_of_p


def error_callback(result):
    print("Error!")


def done_callback(result):
    print("Done")


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--mut', type=str, required=True)
    my_parser.add_argument('--param', default=None)
    my_parser.add_argument('--proc', type=int, default=1)
    args = my_parser.parse_args()

    # Params
    model = args.model
    mut_name = args.mut
    param = args.param

    N = 100
    B = 100
    exp_n = 100

    path = os.path.join('raw_data', model)
    path_ = os.path.join('rep_mce', model)

    if not os.path.isdir('rep_mce'):
        os.mkdir('rep_mce')

    if not os.path.isdir(path_):
        os.mkdir(path_)

    # Data loading
    dat = pd.read_csv(os.path.join(path, 'results_{}_original.csv'.format(model)), sep=';')
    col = dat.columns[1:]

    if mut_name == 'original':
        dat_mut = pd.read_csv(os.path.join(path, 'results_{}_original.csv'.format(model)), sep=';')
    else:
        dat_mut = pd.read_csv(os.path.join(path, 'results_{}_{}_mutated0_MP_{}.csv'.format(model, mut_name, param)),
                              sep=';')

    col_mut = dat_mut.columns[1:]
    dat_mut = dat_mut.tail(1)

    pool = TqdmMultiProcessPool(args.proc)
    initial_tasks = [(experiment, (dat, col, dat_mut, col_mut, B, N, None)) for _ in range(exp_n)]
    with tqdm.tqdm(total=exp_n, dynamic_ncols=True, leave=False) as global_progress:
        global_progress.set_description("global")
        list_rep = pool.map(global_progress, initial_tasks, error_callback, done_callback)

    if mut_name == 'original':
        np.save(os.path.join(path_, '{}_original_200_pop_size.npy'.format(model)), np.array(list_rep))
    else:
        np.save(os.path.join(path_, '{}_{}_{}_200_pop_size.npy'.format(model, mut_name, param)), np.array(list_rep))
