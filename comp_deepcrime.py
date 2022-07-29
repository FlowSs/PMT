import argparse
import re

import numpy as np
import pandas as pd

import settings
import utils
import tqdm

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--mut', type=str, required=True)
    my_parser.add_argument('--dc', default=False, action="store_true")
    args = my_parser.parse_args()

    # Params
    model = args.model
    mut_name = args.mut
    params = settings.main_dict[model][mut_name]
    N = 100
    B = 100
    seed = 42

    print('Model {}, Mutation {}'.format(model, mut_name))

    # Fixing the seed for all sampling choice for reproductibility
    rng = np.random.default_rng(seed)

    # Data loading
    if args.dc:
        dat = pd.read_csv('raw_data/deepcrime_comp/results_{}_original_dc.csv'.format(model), sep=';')

        dat_mut = [
            pd.read_csv('raw_data/deepcrime_comp/results_{}_{}_mutated0_MP_{}_dc.csv'.format(model, mut_name, params[i]), sep=';')
            for i in range(len(params))]

        # Columns names, same for all mutations (here, same for all, but might not be always the case)
        col = dat.columns[1:]
        col_mut = dat_mut[0].columns[1:]

        dat = dat.tail(1)
        dat_mut = [dat_mut[i].tail(1) for i in range(len(params))]

        for i in range(len(dat_mut)):
            print("Exp: {}_{}".format(mut_name, params[i]))
            p_value = utils.p_value_glm(list(dat.to_numpy()[0][1:]), list(dat_mut[i].to_numpy()[0][1:]))
            effect_size = utils.cohen_d(list(dat.to_numpy()[0][1:]), list(dat_mut[i].to_numpy()[0][1:]))
            print("p-value {}, effect_size {}".format(p_value, effect_size))
            if p_value < 0.05 and effect_size >= 0.5:
               print("Killed")
            print("*********************")
    else:

        def run_exp(d, d2, c, c2, mut=False):
                        
            choice_pop = rng.choice(np.arange(200), size=100, replace=False)
            # If we are doing sound vs sound instances, make sure there is no overlap
            if not mut:
                choice_pop_2 = list(set(np.arange(200)).difference(choice_pop))
            else:
                choice_pop_2 = rng.choice(np.arange(200), size=100, replace=False)
            p = 0
            for _ in range(100):
                pop_unknown = rng.choice(choice_pop_2, size=20, replace=False)
                acc_choice2 = list(d2[c2[pop_unknown]].tail(1).to_numpy()[0])
                pop_saine = rng.choice(choice_pop, size=20, replace=False)
                acc_choice = list(d[c[pop_saine]].tail(1).to_numpy()[0])

                p_value = utils.p_value_glm(acc_choice, acc_choice2)
                effect_size = utils.cohen_d(acc_choice, acc_choice2)

                if p_value < 0.05 and effect_size >= 0.5:
                    p += 1
            p /= 100
                
            return p

        dat = pd.read_csv('raw_data/{}/results_{}_original.csv'.format(model, model), sep=';')

        dat_mut = [
            pd.read_csv('raw_data/{}/results_{}_{}_mutated0_MP_{}.csv'.format(model, model, mut_name, params[i]), sep=';')
            for i in range(len(params))]
        # Columns names, same for all mutations (here, same for all, but might not be always the case)
        col = dat.columns[1:]
        col_mut = dat_mut[0].columns[1:]

        dat = dat.tail(1)
        dat_mut = [dat_mut[i].tail(1) for i in range(len(params))]

        p = [run_exp(dat, dat, col, col, False) for _ in tqdm.tqdm(range(50))]

        print("Average number of mutation test passed for healthy instances vs healthy instances: {:.2f} ({:.3f})".format(np.mean(p), np.std(p)))

        for i in range(len(params)):
            p = [run_exp(dat, dat_mut[i], col, col_mut, False) for _ in tqdm.tqdm(range(50))]
            print("Average number of mutation test passed for healthy instances vs mutated instances ({}): {:.2f} ({:.3f})".format(mut_name+'_'+str(params[i]), np.mean(p), np.std(p)))


