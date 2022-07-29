import argparse
import os
import warnings

import numpy as np

import utils

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--mut', type=str, required=True)
    my_parser.add_argument('--param', default=None)
    args = my_parser.parse_args()

    # Params
    model = args.model
    mut_name = args.mut
    param = args.param

    N = 100

    path = os.path.join('rep_mce', model)

    # Data loading
    dat = np.load(os.path.join(path, '{}_original_200_pop_size.npy'.format(model)))
    if mut_name == 'original':
        dat_mut = np.load(os.path.join(path, '{}_original_200_pop_size.npy'.format(model)))
    else:
        dat_mut = np.load(os.path.join(path, '{}_{}_{}_200_pop_size.npy'.format(model, mut_name, param)))
    # Removing warnings for calculation where beta.pdf is very close to 0 or 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l_jack = utils.jack_estimate(N, dat, dat_mut)
    print("Model {}, Mutation {} (Param {})".format(model, mut_name, param))
    print("Point estimate: Avg {}, 95% Confidence Interval ({}, {})".format(l_jack[0][0], l_jack[0][0] - 1.96*l_jack[0][1], l_jack[0][0] + 1.96*l_jack[0][1]))
    print("Credible Interval bounds:\n Lower bound Avg {}, 95% Confidence Interval ({}, {})\n "
          "Upper bound 95% Avg {}, 95% Confidence Interval ({}, {})".format(l_jack[1][0], l_jack[1][0] - 1.96*l_jack[1][1], l_jack[1][0] + 1.96*l_jack[1][1], l_jack[2][0], l_jack[2][0] - 1.96*l_jack[2][1], l_jack[2][0] + 1.96*l_jack[2][1]))
    print("p(B_s < B_m): Avg {}, 95% Confidence Interval: ({}, {})".format(l_jack[3][0], l_jack[3][0] - 1.96*l_jack[3][1], l_jack[3][0] + 1.96*l_jack[3][1]))
