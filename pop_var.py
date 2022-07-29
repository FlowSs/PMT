import argparse
import os.path
import warnings

import numpy as np
import tqdm

from tqdm_multiprocess import TqdmMultiProcessPool

import utils


def pop_change_std(N, list_of_rep, list_of_rep_mut, ci='mean', tqdm_func=None,
                   global_tqdm=None):
    # global list of the different estimate, lower approximation and upper approximation
    estim_list_glob, b_lo_list_glob, b_up_list_glob, p_over_list_glob = [], [], [], []
    estim_list_glob_lower, b_lo_list_glob_lower, b_up_list_glob_lower, p_over_list_glob_lower = [], [], [], []
    estim_list_glob_upper, b_lo_list_glob_upper, b_up_list_glob_upper, p_over_list_glob_upper = [], [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if global_tqdm is not None:
            # For all the different pop size (default 30, should be equal between list_of_rep and list_of_rep_mut)
            with tqdm_func(total=len(list_of_rep), dynamic_ncols=True, leave=False) as prog2:
                prog2.set_description("Pop_size")
                for pop in range(len(list_of_rep)):
                    res = utils.jack_estimate(N, list_of_rep[pop], list_of_rep_mut[pop], ci)

                    estim_list_glob.append(res[0][0])
                    b_lo_list_glob.append(res[1][0])
                    b_up_list_glob.append(res[2][0])
                    p_over_list_glob.append(res[3][0])

                    estim_list_glob_lower.append(res[0][0] - 1.96 * res[0][1])
                    b_lo_list_glob_lower.append(res[1][0] - 1.96 * res[1][1])
                    b_up_list_glob_lower.append(res[2][0] - 1.96 * res[2][1])
                    p_over_list_glob_lower.append(res[3][0] - 1.96 * res[3][1])

                    estim_list_glob_upper.append(res[0][0] + 1.96 * res[0][1])
                    b_lo_list_glob_upper.append(res[1][0] + 1.96 * res[1][1])
                    b_up_list_glob_upper.append(res[2][0] + 1.96 * res[2][1])
                    p_over_list_glob_upper.append(res[3][0] + 1.96 * res[3][1])
                    prog2.update()
            global_tqdm.update()
        else:
            for pop in tqdm.tqdm(range(len(list_of_rep))):
                res = utils.jack_estimate(N, list_of_rep[pop], list_of_rep_mut[pop], ci)

                estim_list_glob.append(res[0][0])
                b_lo_list_glob.append(res[1][0])
                b_up_list_glob.append(res[2][0])
                p_over_list_glob.append(res[3][0])

                estim_list_glob_lower.append(res[0][0] - 1.96 * res[0][1])
                b_lo_list_glob_lower.append(res[1][0] - 1.96 * res[1][1])
                b_up_list_glob_lower.append(res[2][0] - 1.96 * res[2][1])
                p_over_list_glob_lower.append(res[3][0] - 1.96 * res[3][1])

                estim_list_glob_upper.append(res[0][0] + 1.96 * res[0][1])
                b_lo_list_glob_upper.append(res[1][0] + 1.96 * res[1][1])
                b_up_list_glob_upper.append(res[2][0] + 1.96 * res[2][1])
                p_over_list_glob_upper.append(res[3][0] + 1.96 * res[3][1])

    # returning median and quantiles 2.5th and 97.5th
    return [(np.median(estim_list_glob), np.quantile(estim_list_glob, 0.025), np.quantile(estim_list_glob, 0.975)),
            (np.median(b_lo_list_glob), np.quantile(b_lo_list_glob, 0.025), np.quantile(b_lo_list_glob, 0.975)),
            (np.median(b_up_list_glob), np.quantile(b_up_list_glob, 0.025), np.quantile(b_up_list_glob, 0.975)),
            (np.median(p_over_list_glob), np.quantile(p_over_list_glob, 0.025), np.quantile(p_over_list_glob, 0.975))], \
           [(np.median(estim_list_glob_lower), np.quantile(estim_list_glob_lower, 0.025),
             np.quantile(estim_list_glob_lower, 0.975)),
            (np.median(b_lo_list_glob_lower), np.quantile(b_lo_list_glob_lower, 0.025),
             np.quantile(b_lo_list_glob_lower, 0.975)),
            (np.median(b_up_list_glob_lower), np.quantile(b_up_list_glob_lower, 0.025),
             np.quantile(b_up_list_glob_lower, 0.975)),
            (np.median(p_over_list_glob_lower), np.quantile(p_over_list_glob_lower, 0.025),
             np.quantile(p_over_list_glob_lower, 0.975))], \
           [(np.median(estim_list_glob_upper), np.quantile(estim_list_glob_upper, 0.025),
             np.quantile(estim_list_glob_upper, 0.975)),
            (np.median(b_lo_list_glob_upper), np.quantile(b_lo_list_glob_upper, 0.025),
             np.quantile(b_lo_list_glob_upper, 0.975)),
            (np.median(b_up_list_glob_upper), np.quantile(b_up_list_glob_upper, 0.025),
             np.quantile(b_up_list_glob_upper, 0.975)),
            (np.median(p_over_list_glob_upper), np.quantile(p_over_list_glob_upper, 0.025),
             np.quantile(p_over_list_glob_upper, 0.975))]


def error_callback(result):
    print("Error!")


def done_callback(result):
    print("Done")


if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--model', type=str, required=True)
    my_parser.add_argument('--mut', type=str, required=True)
    my_parser.add_argument('--param', default=None)
    my_parser.add_argument('--pop_size', type=int, default=None)
    my_parser.add_argument('--proc', type=int, default=1)
    args = my_parser.parse_args()

    # Params
    model = args.model
    mut_name = args.mut
    param = args.param
    pop_size = args.pop_size
    N = 100
    pop_list = np.array([25, 50, 75, 100, 125, 150, 175, 190])

    path = os.path.join('rep_practicality', model)
    path_ = os.path.join(path, 'data_plot')

    if not os.path.isdir(path_):
        os.mkdir(path_)

    # to get plot for all pop size
    if pop_size is None:
        if not os.path.exists(os.path.join(path_, 'l_std_{}_{}_{}.npy'.format(model, mut_name, param))) \
                and not os.path.exists(os.path.join(path_, 'l_std_{}_{}.npy'.format(model, mut_name))):
            pool = TqdmMultiProcessPool(args.proc)
            initial_tasks = []

            for pop_s in pop_list:
                # Data loading
                dat = np.load(os.path.join(path, '{}_original_30_rep_{}_size_pop.npy'.format(model, pop_s)))

                if mut_name == 'original':
                    dat_mut = np.load(
                        os.path.join(path, '{}_original_30_rep_{}_size_pop.npy'.format(model, pop_s)))
                else:
                    dat_mut = np.load(
                        os.path.join(path, '{}_{}_{}_30_rep_{}_size_pop.npy'.format(model, mut_name, param,
                                                                                    pop_s)))

                initial_tasks.append((pop_change_std, (N, dat, dat_mut, 'mean')))

            with tqdm.tqdm(total=len(pop_list), dynamic_ncols=True, leave=False) as global_progress:
                global_progress.set_description("global")
                l_std = pool.map(global_progress, initial_tasks, error_callback, done_callback)
            if mut_name == 'original':
                np.save(os.path.join(path_, 'l_std_{}_{}.npy'.format(model, mut_name)), np.array(l_std))
            else:
                np.save(os.path.join(path_, 'l_std_{}_{}_{}.npy'.format(model, mut_name, param)), np.array(l_std))
        else:
            if mut_name == 'original':
                l_std = np.load(os.path.join(path_, 'l_std_{}_{}.npy'.format(model, mut_name)))
            else:
                l_std = np.load(os.path.join(path_, 'l_std_{}_{}_{}.npy'.format(model, mut_name, param)))
        utils.plot_pop_std(l_std, model, mut_name, param)
    # To get data with a specific pop size
    else:
        if not os.path.exists(os.path.join(path_, 'l_std_{}_{}_{}.npy'.format(model, mut_name, param))) \
                and not os.path.exists(os.path.join(path_, 'l_std_{}_{}.npy'.format(model, mut_name))):
            # Data loading
            dat = np.load(os.path.join(path, '{}_original_30_rep_{}_size_pop.npy'.format(model, pop_size)))

            if mut_name == 'original':
                dat_mut = np.load(
                    os.path.join(path, '{}_original_30_rep_{}_size_pop.npy'.format(model, pop_size)))
            else:
                dat_mut = np.load(
                    os.path.join(path, '{}_{}_{}_30_rep_{}_size_pop.npy'.format(model, mut_name, param,
                                                                                pop_size)))

            l_std = pop_change_std(N, dat, dat_mut)
        else:
            if mut_name == 'original':
                l_std = \
                    np.load(os.path.join(path_, 'l_std_{}_{}.npy'.format(model, mut_name)))[
                        np.where(pop_list == pop_size)][
                        0]
            else:
                l_std = np.load(os.path.join(path_, 'l_std_{}_{}_{}.npy'.format(model, mut_name, param)))[
                    np.where(pop_list == pop_size)][0]
        print("*** Results ***")
        print("Pop size {}".format(pop_size))
        print("\nAverage and Std of each estimate: ")
        print("Point estimate Mean: {}, CI: ({}, {})".format(l_std[0][0][0], l_std[0][0][1], l_std[0][0][2]))
        print("Credible Interval bounds:\n Lower bound Mean: {}, CI: ({}, {})\n "
              "Upper bound Mean: {}, CI: ({}, {})".format(l_std[0][1][0], l_std[0][1][1], l_std[0][1][2],
                                                          l_std[0][2][0], l_std[0][2][1], l_std[0][2][2]))
        print("p(B_s < B_m) Mean: {}, CI: ({}, {})".format(l_std[0][3][0], l_std[0][3][1], l_std[0][3][2]))
        print("\nAverage and Std of each estimate lower bound: ")
        print("Point estimate Mean: {}, CI: ({}, {})".format(l_std[1][0][0], l_std[1][0][1], l_std[1][0][2]))
        print("Credible Interval bounds:\n Lower bound Mean: {}, CI: ({}, {})\n "
              "Upper bound Mean: {}, CI: ({}, {})".format(l_std[1][1][0], l_std[1][1][1], l_std[1][1][2],
                                                          l_std[1][2][0], l_std[1][2][1], l_std[1][2][2]))
        print("p(B_s < B_m) Mean: {}, CI: ({}, {})".format(l_std[1][3][0], l_std[1][3][1], l_std[1][3][2]))
        print("\nAverage and Std of each estimate upper bound: ")
        print("Point estimate Mean: {}, CI: ({}, {})".format(l_std[2][0][0], l_std[2][0][1], l_std[2][0][2]))
        print("Credible Interval bounds:\n Lower bound Mean: {}, CI: ({}, {})\n "
              "Upper bound Mean: {}, CI: ({}, {})".format(l_std[2][1][0], l_std[2][1][1], l_std[2][1][2],
                                                          l_std[2][2][0], l_std[2][2][1], l_std[2][2][2]))
        print("p(B_s < B_m) Mean: {}, CI: ({}, {})".format(l_std[2][3][0], l_std[2][3][1], l_std[2][3][2]))
