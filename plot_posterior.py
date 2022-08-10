import argparse
import os.path
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

import settings
import utils


def experiment(data, columns, data2, columns2, B=100, N=100, rng_=None, same=False):
    """

    :param data: control data (sound instances)
    :param columns: column names of control group data
    :param data2: mutated data (mutated instances)
    :param columns2: columns names of mutated data
    :param B: Number of bootstrap resampling. Default is 100
    :param N: Number of trials for the Binomial experiment (test returning mutant or not for n vs n instances). Default is 100
    :param rng_: Random Seed generator for reproducibility. Default is random.
    :param same: Whether to use the same models between healthy and mutated instances. No difference in case of source level mutation,
    but mandatory in case of model-level (since the mutation is based off a healthy instance)

    :return: List of the number of success (test return instances are mutant) over the N trials for each of the B bootstrap samples
    """
    list_of_p = []

    # Nb of repetitions (B)
    for _ in tqdm(range(B)):
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
            if same:
               acc_choice = list(data[pop_unknown].to_numpy()[0])
            else:
               acc_choice = list(data[pop_sound].to_numpy()[0])

            p_value = utils.p_value_glm(acc_choice, acc_choice2)
            effect_size = utils.cohen_d(acc_choice, acc_choice2)

            if p_value < 0.05 and effect_size >= 0.5:
                p += 1
        p /= N
        list_of_p.append(p)
    return list_of_p


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--mut', type=str, required=True)
    my_parser.add_argument('--same', default=False, action="store_true")
    args = my_parser.parse_args()

    # Params
    model = args.model
    mut_name = args.mut
    params = settings.main_dict[model][mut_name]
    N = 100
    B = 100
    seed = 42

    print('Model {}, Mutation {}'.format(model, mut_name))

    if not os.path.exists(os.path.join('plot_results', model, model+'_'+mut_name+'_param', 'list_mut.npy')):
        # Fixing the seed for all sampling choice for reproductibility
        rng = np.random.default_rng(seed)

        path = os.path.join('raw_data', model)

        # Data loading
        dat = pd.read_csv(os.path.join(path, 'results_{}_original.csv'.format(model)), sep=';')

        dat_mut = [
            pd.read_csv(os.path.join(path, 'results_{}_{}_mutated0_MP_{}.csv'.format(model, mut_name, params[i])), sep=';')
            for i in range(len(params))]

        # Columns names, same for all mutations (here, same for all, but might not be always the case)
        col = dat.columns[1:]
        col_mut = dat_mut[0].columns[1:]

        dat = dat.tail(1)
        dat_mut = [dat_mut[i].tail(1) for i in range(len(params))]

        # Running experiments for each params of the mutation
        print("Running on original instances...")
        if not args.same:
            if not (os.path.exists('{}_orig.npy'.format(model))):
                list_orig = experiment(dat, col, dat, col, N=N, B=B, rng_=rng, same=False)
                np.save('{}_orig.npy'.format(model), np.array(list_orig))
            else:
                print("Loading original instances posterior...")
                list_orig = np.load('{}_orig.npy'.format(model))
        else:
            if not (os.path.exists('{}_orig_same.npy'.format(model))):
                list_orig = experiment(dat, col, dat, col, N=N, B=B, rng_=rng, same=True)
                np.save('{}_orig_same.npy'.format(model), np.array(list_orig))
            else:
                print("Loading original instances posterior...")
                list_orig = np.load('{}_orig_same.npy'.format(model))

        print("Running on mutation...")
        list_mut = []
        for i in range(len(params)):
            print("Exp: {}_{}".format(mut_name, params[i]))
            list_mut.append(experiment(dat, col, dat_mut[i], col_mut, N=N, B=B, rng_=rng, same=args.same))
            if not os.path.exists(os.path.join('plot_results', model, model+'_'+mut_name+'_param')):
                os.mkdir(os.path.join('plot_results', model, model+'_'+mut_name+'_param'))
            np.save(os.path.join('plot_results', model, model+'_'+mut_name+'_param', 'list_mut.npy'), np.array(list_mut))
    else:
        if args.same:
            list_orig = np.load('{}_orig_same.npy'.format(model))
        else:
            list_orig = np.load('{}_orig.npy'.format(model))
        list_mut = np.load(os.path.join('plot_results', model, model+'_'+mut_name+'_param', 'list_mut.npy'))

    # Ignoring runtime warning of divide by zero of the beta pdf thrown when the mutation posterior
    # is near 1 or 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        utils.plot_fig_exp(N, list_orig, list_mut, model, mut_name, params, ci='mean')

