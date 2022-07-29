import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy.special import gamma, gammaln, betainc
from scipy.stats import beta

svg_plot = False

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

plt.rc('font', size=7)

def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)


def my_floor(a, precision=0):
    return np.true_divide(np.floor(a * 10 ** precision), 10 ** precision)


# get y among x
def general_binom_coeff(x, y):
    return gamma(x + 1) / (gamma(y + 1) * gamma(x - y + 1))


# Implementation of "John D Cook. 2003. Numerical computation of stochastic inequality probabilities.
# Univ. Of Texas, MD Anderson Cancer Center Department of Biostatistics Working
# Paper Series, Paper 46 (2003)."
def gamma_n(n, b, c, d):
    somme = 0
    for (i, j) in zip(range(n + 1), range(n, -1, -1)):
        alph = general_binom_coeff(b - 1, i) * ((-1) ** i)
        bet = general_binom_coeff(d - 1, j) * ((-1) ** j / (c + j))
        somme += alph * bet

    return somme


def g_l(a, b, c, d, n):
    somme = 0

    for i in range(n):
        somme += gamma_n(i, b, c, d) / ((a + c + i) * (2 ** (a + c + i)))

    return somme * (1 / beta_coeff(a, b)) * (1 / beta_coeff(c, d))


def h(a, b, c, d):
    return beta_coeff(a + c, b + d) / (beta_coeff(a, b) * beta_coeff(c, d))


def case_g(a, b, c, d, min_param, argmin_param):
    if argmin_param == 0:
        # make sure a is now in [1, 2)
        a_ = a - (int(min_param) - 1)
        assert 2 > a_ >= 1
        # Initialize recurrence by calling again g
        # If now all parameters are < 2, will return p
        # approximated with small parameters
        # Otherwise, will launch a new recurrence
        p = g(a_, b, c, d) + h(a_, b, c, d) / a_
        a_ += 1
        # Now that we have g(a_,b,c,d) for a_ \in [1, 2)
        # loop until we reach a
        while a_ != a:
            p = p + h(a_, b, c, d) / a_
            a_ += 1
    elif argmin_param == 1:
        b_ = b - (int(min_param) - 1)
        assert 2 > b_ >= 1

        p = g(a, b_, c, d) - h(a, b_, c, d) / b_
        b_ += 1
        while b_ != b:
            p = p - h(a, b_, c, d) / b_
            b_ += 1
    elif argmin_param == 2:
        c_ = c - (int(min_param) - 1)
        assert 2 > c_ >= 1

        p = g(a, b, c_, d) - h(a, b, c_, d) / c_
        c_ += 1
        while c_ != c:
            p = p - h(a, b, c_, d) / c_
            c_ += 1
    else:
        d_ = d - (int(min_param) - 1)
        assert 2 > d_ >= 1

        p = g(a, b, c, d_) + h(a, b, c, d_) / d_
        d_ += 1
        while d_ != d:
            p = p + h(a, b, c, d_) / d_
            d_ += 1

    # print("p:", p)
    assert 1 + 1e-5 >= p >= 0 - 1e-5
    return p


def g(a, b, c, d):
    # With small arguments approximations
    if a < 2 and b < 2 and c < 2 and d < 2:
        n = 50  # just for good measure, since approximation is bounded for N > 2
        p = g_l(a, b, c, d, n) + betainc(b, a, 0.5) - g_l(b, a, d, c, n)
        try:
            assert 1 + 1e-5 >= p >= 0 - 1e-5
        except:
            raise Exception('Probability is {}, parameters {}'.format(p, (a, b, c, d)))
        return p

    # if not small argument, use recurrence
    l = np.array([a, b, c, d])
    ll = np.ma.MaskedArray(l, l < 2)
    min_param = np.ma.min(ll)
    argmin_param = np.ma.argmin(ll)
    # print(argmin_param, min_param)
    return case_g(a, b, c, d, min_param, argmin_param)


# credit to https://stackoverflow.com/questions/68008346/analytic-highest-density-interval-in-python-preferably-for-beta-distributions
def HDIofICDF(dist_name, credMass=0.95, **args):
    # freeze distribution with given arguments
    distri = dist_name(**args)
    # initial guess for HDIlowTailPr
    incredMass = 1.0 - credMass

    def intervalWidth(lowTailPr):
        return distri.ppf(credMass + lowTailPr) - distri.ppf(lowTailPr)

    # find lowTailPr that minimizes intervalWidth
    HDIlowTailPr = fmin(intervalWidth, incredMass, ftol=1e-8, disp=False)[0]
    # return interval as array([low, high])
    return distri.ppf([HDIlowTailPr, credMass + HDIlowTailPr])


# Beta coefficient B(a,b)
def beta_coeff(a, b):
    # return (gamma(a) * gamma(b)) / gamma(a + b)
    # For stability using log version, here it is possible as a, b > 0
    return np.exp(gammaln(a) + gammaln(b) - gammaln(a + b))


# Beta distribution pdf
def beta_func(x, a, b):
    return (1 / beta_coeff(a, b)) * (x ** (a - 1)) * ((1 - x) ** (b - 1))


# Calculates cohen's kappa value
def cohen_d(orig_accuracy_list, accuracy_list):
    nx = len(orig_accuracy_list)
    ny = len(accuracy_list)

    dof = nx + ny - 2
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(orig_accuracy_list, ddof=1) ** 2 + (ny - 1) * np.std(accuracy_list, ddof=1) ** 2) / dof)
    result = (np.mean(orig_accuracy_list) - np.mean(accuracy_list)) / pooled_std
    return abs(result)


def p_value_glm(orig_accuracy_list, accuracy_list):
    list_length = len(orig_accuracy_list)

    zeros_list = [0] * list_length
    ones_list = [1] * list_length
    mod_lists = zeros_list + ones_list
    acc_lists = orig_accuracy_list + accuracy_list

    data = {'Acc': acc_lists, 'Mod': mod_lists}
    df = pd.DataFrame(data)

    response, predictors = dmatrices("Acc ~ Mod", df, return_type='dataframe')
    glm = sm.GLM(response, predictors)
    glm_results = glm.fit()
    glm_sum = glm_results.summary()
    pv = str(glm_sum.tables[1][2][4])
    p_value_g = float(pv)

    return p_value_g


def credible_interval(param, ci='mean'):
    """
    Determine a 95% credible interval of the bagged posterior based of alpha/beta.
    The type of interval is determine by ci parameter

    :param param: parameters of the beta distribution (tuple)
    :param ci: credible interval type. Default is mean.

    :return: (Lower, Upper) bound of the CI
    """
    if ci == 'mode':
        b_lo, b_up = HDIofICDF(beta, 0.95, a=param[0], b=param[1])
    elif ci == 'mean':
        # Using normal approximation for the Z-score
        b_lo, b_up = np.clip(beta.mean(param[0], param[1]) - 1.96 * beta.std(param[0], param[1]), 0, 1), np.clip(
            beta.mean(param[0], param[
                1]) + 1.96 * beta.std(param[0], param[1]), 0, 1)
    elif ci == 'median':
        raise NotImplementedError
    else:
        raise Exception("Credible Interval {} not recognized".format(ci))

    return b_lo, b_up


def estimate_calc(param, ci='mean'):
    """
    Determine an estimate of the posterior distribution

    :param param: parameters of the beta distribution (tuple)
    :param ci: estimate type. Default is mean.

    :return: Estimate value
    """
    if ci == 'mode':
        if param[0] > 1 and param[1] > 1:
            estim = (param[0] - 1) / (param[0] + param[1] - 2)
        elif param[0] == 1 and param[1] == 1:
            # if alpha = beta = 1, mode is any value, so we put 0.5 by default (equal chance of being mutant)
            estim = 0.5
        elif param[0] < 1 and param[1] < 1:
            raise Exception('Multimodal distribution')
        elif param[0] <= 1 and param[1] > 1:
            estim = 0
        elif param[0] > 1 and param[1] <= 1:
            estim = 1
        else:
            raise Exception('I forgot to implement a case !')
    elif ci == 'mean':
        # Using normal approximation for the Z-score
        estim = beta.mean(param[0], param[1])
    elif ci == 'median':
        raise NotImplementedError
    else:
        raise Exception("Credible Interval {} not recognized".format(ci))

    return estim


# https://stackoverflow.com/questions/42021972/truncating-decimal-digits-numpy-array-of-floats
def trunc(values, decs=0):
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def plot_exp_intro(p, p2, p3):
    fig = plt.figure(figsize=(3.5, 3.5))
    plt.rcParams.update({'font.size': 8})

    list_n_ind = np.arange(20, 95, 5)
    for ind, size in enumerate(list_n_ind):
        plt.scatter([size] * 50, p[ind], c='blue', alpha=0.4)
    for ind, size in enumerate(list_n_ind):
        plt.scatter([size] * 50, p2[ind], c='orange', alpha=0.4)
    for ind, size in enumerate(list_n_ind):
        plt.scatter([size] * 50, p3[ind], c='green', alpha=0.4)

    plt.xlabel('Number of sound/mutated instances to compare (k vs k)')
    plt.xlim([18, 95])
    plt.ylabel('Probability of "unknown" sample being deemed mutated')

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    if svg_plot:
        plt.savefig(os.path.join('plot_results', 'exp_intro.svg'), format='svg', transparent=True)
    else:
        plt.savefig(os.path.join('plot_results', 'exp_intro.png'), dpi=400)
        plt.show()


def plot_pop_std_just_paper(l_std, l_std_mut, model, mut_name, param_list):
    fig, axs = plt.subplots(2, 2, figsize=(7., 3.8))
    l_std = np.array(l_std)
    l_std_mut = np.array(l_std_mut)

    x = np.array([25, 50, 75, 100, 125, 150, 175, 200])
    # Plot on the estimate point
    axs[0][0].errorbar(x, [l_std[i][0][0][0] for i in range(len(l_std))],
                       [[l_std[i][0][0][0] - l_std[i][0][0][1] for i in range(len(l_std))],
                        [l_std[i][0][0][2] - l_std[i][0][0][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    axs[0][0].errorbar(x - 7, [l_std[i][1][0][0] for i in range(len(l_std))],
                       [[l_std[i][1][0][0] - l_std[i][1][0][1] for i in range(len(l_std))],
                        [l_std[i][1][0][2] - l_std[i][1][0][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    axs[0][0].errorbar(x + 7, [l_std[i][2][0][0] for i in range(len(l_std))],
                       [[l_std[i][2][0][0] - l_std[i][2][0][1] for i in range(len(l_std))],
                        [l_std[i][2][0][2] - l_std[i][2][0][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    axs[0][0].set_title('Point Estimate, Original')
    axs[0][0].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[0][0].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[0][0].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[0][0].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[0][0].set_yticklabels(np.round(axs[0][0].get_yticks(), 1), rotation=45)
    axs[0][0].set_ylim([0, 1])
    axs[0][0].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the estimate point
    axs[0][1].errorbar(x, [l_std_mut[0][i][0][0][0] for i in range(len(l_std_mut[0]))],
                       [[l_std_mut[0][i][0][0][0] - l_std_mut[0][i][0][0][1] for i in range(len(l_std_mut[0]))],
                        [l_std_mut[0][i][0][0][2] - l_std_mut[0][i][0][0][0] for i in range(len(l_std_mut[0]))]],
                       fmt="o", c='blue', capsize=2)
    axs[0][1].errorbar(x - 7, [l_std_mut[0][i][1][0][0] for i in range(len(l_std_mut[0]))],
                       [[l_std_mut[0][i][1][0][0] - l_std_mut[0][i][1][0][1] for i in range(len(l_std_mut[0]))],
                        [l_std_mut[0][i][1][0][2] - l_std_mut[0][i][1][0][0] for i in range(len(l_std_mut[0]))]],
                       fmt="o", c='green', capsize=2)
    axs[0][1].errorbar(x + 7, [l_std_mut[0][i][2][0][0] for i in range(len(l_std_mut[0]))],
                       [[l_std_mut[0][i][2][0][0] - l_std_mut[0][i][2][0][1] for i in range(len(l_std_mut[0]))],
                        [l_std_mut[0][i][2][0][2] - l_std_mut[0][i][2][0][0] for i in range(len(l_std_mut[0]))]],
                       fmt="o", c='orange', capsize=2)
    axs[0][1].set_title('Point Estimate, Param {}'.format(param_list[0]))
    axs[0][1].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[0][1].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[0][1].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[0][1].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[0][1].set_yticklabels(np.round(axs[0][1].get_yticks(), 1), rotation=45)
    axs[0][1].set_ylim([0, 1])
    axs[0][1].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the estimate point
    axs[1][0].errorbar(x, [l_std_mut[1][i][0][0][0] for i in range(len(l_std_mut[1]))],
                       [[l_std_mut[1][i][0][0][0] - l_std_mut[1][i][0][0][1] for i in range(len(l_std_mut[1]))],
                        [l_std_mut[1][i][0][0][2] - l_std_mut[1][i][0][0][0] for i in range(len(l_std_mut[1]))]],
                       fmt="o", c='blue', capsize=2)
    axs[1][0].errorbar(x - 7, [l_std_mut[1][i][1][0][0] for i in range(len(l_std_mut[1]))],
                       [[l_std_mut[1][i][0][0][0] - l_std_mut[1][i][0][0][1] for i in range(len(l_std_mut[1]))],
                        [l_std_mut[1][i][0][0][2] - l_std_mut[1][i][0][0][0] for i in range(len(l_std_mut[1]))]],
                       fmt="o", c='green', capsize=2)
    axs[1][0].errorbar(x + 7, [l_std_mut[1][i][2][0][0] for i in range(len(l_std_mut[1]))],
                       [[l_std_mut[1][i][0][0][0] - l_std_mut[1][i][0][0][1] for i in range(len(l_std_mut[1]))],
                        [l_std_mut[1][i][0][0][2] - l_std_mut[1][i][0][0][0] for i in range(len(l_std_mut[1]))]],
                       fmt="o", c='orange', capsize=2)
    axs[1][0].set_title('Point Estimate, Param {}'.format(param_list[1]))
    axs[1][0].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[1][0].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[1][0].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[1][0].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[1][0].set_yticklabels(np.round(axs[1][0].get_yticks(), 1), rotation=45)
    axs[1][0].set_ylim([0, 1])
    axs[1][0].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the estimate point
    axs[1][1].errorbar(x, [l_std_mut[2][i][0][0][0] for i in range(len(l_std_mut[2]))],
                       [[l_std_mut[2][i][0][0][0] - l_std_mut[2][i][0][0][1] for i in range(len(l_std_mut[2]))],
                        [l_std_mut[2][i][0][0][2] - l_std_mut[2][i][0][0][0] for i in range(len(l_std_mut[2]))]],
                       fmt="o", c='blue', capsize=2)
    axs[1][1].errorbar(x - 7, [l_std_mut[2][i][1][0][0] for i in range(len(l_std))],
                       [[l_std_mut[2][i][0][0][0] - l_std_mut[2][i][0][0][1] for i in range(len(l_std_mut[2]))],
                        [l_std_mut[2][i][0][0][2] - l_std_mut[2][i][0][0][0] for i in range(len(l_std_mut[2]))]],
                       fmt="o", c='green', capsize=2)
    axs[1][1].errorbar(x + 7, [l_std_mut[2][i][2][0][0] for i in range(len(l_std_mut[2]))],
                       [[l_std_mut[2][i][0][0][0] - l_std_mut[2][i][0][0][1] for i in range(len(l_std_mut[2]))],
                        [l_std_mut[2][i][0][0][2] - l_std_mut[2][i][0][0][0] for i in range(len(l_std_mut[2]))]],
                       fmt="o", c='orange', capsize=2)
    axs[1][1].set_title('Point Estimate, Param {}'.format(param_list[2]))
    axs[1][1].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[1][1].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[1][1].set_yticks(np.arange(0, 1.01, 0.1))
    axs[1][1].set_ylim([0, 1])
    for n, label in enumerate(axs[1][1].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[1][1].set_yticklabels(np.round(axs[1][1].get_yticks(), 1), rotation=45)
    axs[1][1].grid(visible=True, which='major', color='black', alpha=0.3)

    plt.tight_layout()

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    if svg_plot:
        plt.savefig(os.path.join('plot_results', '{}_{}_std_just_paper.svg'.format(model, mut_name)), format='svg', transparent=True)
    else:
        plt.savefig(os.path.join('plot_results', '{}_{}_std_just_paper.png'.format(model, mut_name)), dpi=400)
        plt.show()


def plot_pop_std(l_std, model, mut_name, param):
    fig, axs = plt.subplots(2, 2, figsize=(7., 3.8))
    l_std = np.array(l_std)

    x = np.array([25, 50, 75, 100, 125, 150, 175, 200])
    # Plot on the estimate point
    axs[0][0].errorbar(x, [l_std[i][0][0][0] for i in range(len(l_std))],
                       [[l_std[i][0][0][0] - l_std[i][0][0][1] for i in range(len(l_std))],
                        [l_std[i][0][0][2] - l_std[i][0][0][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    axs[0][0].errorbar(x - 7, [l_std[i][1][0][0] for i in range(len(l_std))],
                       [[l_std[i][1][0][0] - l_std[i][1][0][1] for i in range(len(l_std))],
                        [l_std[i][1][0][2] - l_std[i][1][0][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    axs[0][0].errorbar(x + 7, [l_std[i][2][0][0] for i in range(len(l_std))],
                       [[l_std[i][2][0][0] - l_std[i][2][0][1] for i in range(len(l_std))],
                        [l_std[i][2][0][2] - l_std[i][2][0][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    axs[0][0].set_title('Point Estimate')
    axs[0][0].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[0][0].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[0][0].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[0][0].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[0][0].set_yticklabels(np.round(axs[0][0].get_yticks(), 1), rotation=45)
    axs[0][0].set_ylim([0, 1])
    axs[0][0].set_xlabel('Sampled population size')
    axs[0][0].set_ylabel('Probability')
    axs[0][0].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the Credible interval lower bound
    axs[0][1].errorbar(x, [l_std[i][0][1][0] for i in range(len(l_std))],
                       [[l_std[i][0][1][0] - l_std[i][0][1][1] for i in range(len(l_std))],
                        [l_std[i][0][1][2] - l_std[i][0][1][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    axs[0][1].errorbar(x - 7, [l_std[i][1][1][0] for i in range(len(l_std))],
                       [[l_std[i][1][1][0] - l_std[i][1][1][1] for i in range(len(l_std))],
                        [l_std[i][1][1][2] - l_std[i][1][1][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    axs[0][1].errorbar(x + 7, [l_std[i][2][1][0] for i in range(len(l_std))],
                       [[l_std[i][2][1][0] - l_std[i][2][1][1] for i in range(len(l_std))],
                        [l_std[i][2][1][2] - l_std[i][2][1][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    axs[0][1].set_title('Lower Bound of CI')
    axs[0][1].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[0][1].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[0][1].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[0][1].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[0][1].set_yticklabels(np.round(axs[0][1].get_yticks(), 1), rotation=45)
    axs[0][1].set_ylim([0, 1])
    axs[0][1].set_xlabel('Sampled population size')
    axs[0][1].set_ylabel('Probability')
    axs[0][1].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the Credible interval upper bound
    axs[1][0].errorbar(x, [l_std[i][0][2][0] for i in range(len(l_std))],
                       [[l_std[i][0][2][0] - l_std[i][0][2][1] for i in range(len(l_std))],
                        [l_std[i][0][2][2] - l_std[i][0][2][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    axs[1][0].errorbar(x - 7, [l_std[i][1][2][0] for i in range(len(l_std))],
                       [[l_std[i][1][2][0] - l_std[i][1][2][1] for i in range(len(l_std))],
                        [l_std[i][1][2][2] - l_std[i][1][2][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    axs[1][0].errorbar(x + 7, [l_std[i][2][2][0] for i in range(len(l_std))],
                       [[l_std[i][2][2][0] - l_std[i][2][2][1] for i in range(len(l_std))],
                        [l_std[i][2][2][2] - l_std[i][2][2][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    axs[1][0].set_title('Upper Bound of CI')
    axs[1][0].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[1][0].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[1][0].set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(axs[1][0].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[1][0].set_yticklabels(np.round(axs[1][0].get_yticks(), 1), rotation=45)
    axs[1][0].set_ylim([0, 1])
    axs[1][0].set_xlabel('Sampled population size')
    axs[1][0].set_ylabel('Probability')
    axs[1][0].grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the p(B_s < B_m)
    axs[1][1].errorbar(x, [l_std[i][0][3][0] for i in range(len(l_std))],
                       [[l_std[i][0][3][0] - l_std[i][0][3][1] for i in range(len(l_std))],
                        [l_std[i][0][3][2] - l_std[i][0][3][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    axs[1][1].errorbar(x - 7, [l_std[i][1][3][0] for i in range(len(l_std))],
                       [[l_std[i][1][3][0] - l_std[i][1][3][1] for i in range(len(l_std))],
                        [l_std[i][1][3][2] - l_std[i][1][3][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    axs[1][1].errorbar(x + 7, [l_std[i][2][3][0] for i in range(len(l_std))],
                       [[l_std[i][2][3][0] - l_std[i][2][3][1] for i in range(len(l_std))],
                        [l_std[i][2][3][2] - l_std[i][2][3][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)

    axs[1][1].set_title(r'$p(B_s < B_m)$')
    axs[1][1].set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    axs[1][1].set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    axs[1][1].set_yticks(np.arange(0, 1.01, 0.1))
    axs[1][1].set_ylim([0.5, 1])
    for n, label in enumerate(axs[1][1].yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    axs[1][1].set_yticklabels(np.round(axs[1][1].get_yticks(), 1), rotation=45)
    axs[1][1].set_xlabel('Sampled population size')
    axs[1][1].set_ylabel('Probability')
    axs[1][1].grid(visible=True, which='major', color='black', alpha=0.3)

    plt.tight_layout()

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    if not os.path.isdir(os.path.join('plot_results', model)):
        os.mkdir(os.path.join('plot_results', model))

    path = os.path.join('plot_results', model, 'std')

    if not os.path.isdir(path):
        os.mkdir(path)

    if svg_plot:
        if mut_name == 'original':
            plt.savefig(os.path.join(path, '{}_{}_std.svg'.format(model, mut_name)), format=svg)
        else:
            plt.savefig(os.path.join(path, '{}_{}_{}_std.svg'.format(model, mut_name, param)), format=svg)
    else:
        if mut_name == 'original':
            plt.savefig(os.path.join(path, '{}_{}_std.png'.format(model, mut_name)), dpi=400)
            plt.show()
        else:
            plt.savefig(os.path.join(path, '{}_{}_{}_std.png'.format(model, mut_name, param)), dpi=400)
            plt.show()


def jack_estimate(N, list_of_rep, list_of_rep_mut, ci='mean'):
    # Support of beta dist
    # Starting at 1e-8 for stability
    x = np.arange(1e-8, 1, 0.01)

    # list of the different estimate
    estim_list, ci_list, p_over_list = [], [], []

    for (list_, list_mut) in zip(list_of_rep, list_of_rep_mut):
        # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
        approx_pdf = np.mean(
            [beta.pdf(x, list_[i] * N + 1, N - list_[i] * N + 1) for i in range(len(list_))], 0)
        approx_pdf_mut = np.mean(
            [beta.pdf(x, list_mut[i] * N + 1, N - list_mut[i] * N + 1) for i in range(len(list_mut))], 0)

        # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
        # to get alpha and beta
        # We initialize the parameters at the average of each alpha/beta of each posterior
        param, _ = curve_fit(beta_func, x, approx_pdf,
                             p0=[np.mean(np.array(list_) * N + 1), np.mean(N - np.array(list_) * N + 1)],
                             method='trf', bounds=[0, np.inf], maxfev=2000)
        param_mut, _ = curve_fit(beta_func, x, approx_pdf_mut,
                                 p0=[np.mean(np.array(list_mut) * N + 1), np.mean(N - np.array(list_mut) * N + 1)],
                                 method='trf', bounds=[0, np.inf], maxfev=2000)

        # credible interval
        ci_list.append(credible_interval(param_mut, ci=ci))
        # estimate
        estim_list.append(estimate_calc(param_mut, ci=ci))
        # p(B_s < B_s)
        p_over_list.append(g(param_mut[0], param_mut[1], param[0], param[1]))

    estim_list, ci_list, p_over_list = np.array(estim_list), np.array(ci_list), np.array(p_over_list)

    # jackknife estimate list
    jack_estim_list, jack_b_lo_list, jack_b_up_list, jack_p_over_list = [], [], [], []
    for k in range(len(estim_list)):
        mask = [True if i != k else False for i in range(len(estim_list))]
        jack_estim_list.append(np.sum(estim_list[mask]) / len(estim_list))
        jack_b_lo_list.append(np.sum([ci_list[mask][i][0] for i in range(len(estim_list) - 1)]) / len(estim_list))
        jack_b_up_list.append(np.sum([ci_list[mask][i][1] for i in range(len(estim_list) - 1)]) / len(estim_list))
        jack_p_over_list.append(np.sum(p_over_list[mask]) / len(estim_list))

    # calculating each MCE
    jack_estim, jack_b_lo, jack_b_up, jack_p_over = 0, 0, 0, 0
    for i in range(len(estim_list)):
        jack_estim += (jack_estim_list[i] - np.mean(jack_estim_list)) ** 2
        jack_b_lo += (jack_b_lo_list[i] - np.mean(jack_b_lo_list)) ** 2
        jack_b_up += (jack_b_up_list[i] - np.mean(jack_b_up_list)) ** 2
        jack_p_over += (jack_p_over_list[i] - np.mean(jack_p_over_list)) ** 2

    jack_estim = np.sqrt(((len(estim_list) - 1) / len(estim_list)) * jack_estim)
    jack_b_lo = np.sqrt(((len(estim_list) - 1) / len(estim_list)) * jack_b_lo)
    jack_b_up = np.sqrt(((len(estim_list) - 1) / len(estim_list)) * jack_b_up)
    jack_p_over = np.sqrt(((len(estim_list) - 1) / len(estim_list)) * jack_p_over)

    # Return the MCE estimate with the MCE error
    return [(np.mean(estim_list), jack_estim), (np.mean([ci_list[i][0] for i in range(len(estim_list))]), jack_b_lo),
            (np.mean([ci_list[i][1] for i in range(len(estim_list))]), jack_b_up), (np.mean(p_over_list), jack_p_over)]


def plot_fig_params(N, list1, list2, model, mut_name, ci='mean'):

    b_lo, b_lo_list, b_up, b_up_list, estimate, estimate_list, p_over, p_over_list = calcul_param(N, list1, list2, ci)
    # print(estimate_list)
    # print([b_up_list[k] - b_lo_list[k] for k in range(len(list2))])
    # print(p_over_list)
    print("Plotting diagnostic curve...")

    x, y = np.mgrid[slice(0, 1, 0.01), slice(0, 1, 0.01)]
    mat_res = np.zeros((3, x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            mat_res[0, i, j] = np.sum([estimate_list[k] > x[i][j] and (b_up_list[k] - b_lo_list[k]) < y[i][j]
                                       for k in range(len(estimate_list))]) + (
                                       estimate <= x[i][j] or (b_up - b_lo) >= y[i][j])
            mat_res[1, i, j] = np.sum([estimate_list[k] > x[i][j] and p_over_list[k] > y[i][j]
                                       for k in range(len(estimate_list))]) + (estimate <= x[i][j] or p_over <= y[i][j])
            mat_res[2, i, j] = np.sum([p_over_list[k] > x[i][j] and (b_up_list[k] - b_lo_list[k]) < y[i][j]
                                       for k in range(len(estimate_list))]) + (
                                       p_over <= x[i][j] or (b_up - b_lo) >= y[i][j])

    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(2, 2)

    cmap = get_cmap('RdBu', len(list2) + 2)  # define the colormap

    ax = plt.subplot(gs[0, 0], rasterized=True)
    plt.pcolormesh(x, y, mat_res[0], cmap=cmap)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\tau$')
    ax.set_xlim([estimate, 1])
    ax.set_xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    ax.set_yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    every_nth = 2
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_yticklabels(ax.get_yticks())
    ax.grid(visible=True, which='major', color='black', alpha=0.3)

    ax = plt.subplot(gs[0, 1], rasterized=True)
    plt.pcolormesh(x, y, mat_res[1], cmap=cmap)
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')
    ax.set_xlim([estimate, 1])
    ax.set_ylim([0.5, 1])
    ax.set_xticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    every_nth = 2
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_yticklabels(ax.get_yticks())
    ax.grid(visible=True, which='major', color='black', alpha=0.3)

    ax = plt.subplot(gs[1, :], rasterized=True)
    plt.pcolormesh(x, y, mat_res[2], cmap=cmap, vmin=0, vmax=6)
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.set_xlabel(r'$\phi_2$')
    ax.set_ylabel(r'$\tau$')
    ax.set_xlim([0.5, 1])
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    ax.set_yticks([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
    every_nth = 2
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)
    ax.set_yticklabels(ax.get_yticks())
    ax.grid(visible=True, which='major', color='black', alpha=0.3)

    cb = plt.colorbar(orientation="horizontal")
    # Hack to have the color bar fit!
    cb.set_ticks(np.arange(0, len(list2) + 2, 1) + 1 / (len(list2) + 2) * (3 - np.arange(0, len(list2) + 2, 1)))
    cb.set_ticklabels(np.arange(0, len(list2) + 2, 1))
    # get the xtick labels
    tl = cb.ax.get_xticklabels()
    # set the alignment for the first and the last
    tl[0].set_horizontalalignment('left')
    tl[-1].set_horizontalalignment('right')
    plt.tight_layout()

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    path = os.path.join('plot_results', model, model+'_'+mut_name+'_param')

    if not os.path.isdir(path):
        os.mkdir(path)

    if svg_plot:
        # Create dir to put svg plots, since it is rasterized (so some additional png figure)
        if not os.path.isdir(os.path.join(path, '{}_{}_param'.format(model, mut_name))):
            os.mkdir(os.path.join(path, '{}_{}_param'.format(model, mut_name)))
        plt.savefig(
            os.path.join(path, '{}_{}_param'.format(model, mut_name), '{}_{}_param.svg'.format(model, mut_name)), format='svg', transparent=True)
    else:
        plt.savefig(os.path.join(path, '{}_{}_param.png'.format(model, mut_name)), dpi=400)
        plt.show()


def calcul_param(N, list1, list2, ci='mean'):
    # Support of beta dist
    # Starting at 1e-8 for stability
    x = np.arange(1e-8, 1, 0.01)

    print("Approximating original instances posterior")
    # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
    approx_pdf = np.mean(
        [beta.pdf(x, list1[i] * N + 1, N - list1[i] * N + 1) for i in range(len(list1))], 0)
    # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
    # to get alpha and beta
    # We initialize the parameters at the average of each alpha/beta of each posterior
    param_, _ = curve_fit(beta_func, x, approx_pdf,
                          p0=[np.mean(np.array(list1) * N + 1), np.mean(N - np.array(list1) * N + 1)],
                          method='trf', bounds=[0, np.inf], maxfev=2000)
    # credible interval
    b_lo, b_up = credible_interval(param_, ci=ci)
    # estimate
    estimate = estimate_calc(param_, ci=ci)
    # p(B_s < B_s)
    p_over = g(param_[0], param_[1], param_[0], param_[1])
    # print(estimate)
    # print(b_up - b_lo)
    # print(p_over)
    # print("****")
    # Calc for each mutated bagged posterior
    b_lo_list, b_up_list, estimate_list, p_over_list = [], [], [], []
    print("Approximating mutated instances posterior")
    for ind, l in enumerate(list2):
        # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
        approx_pdf = np.mean(
            [beta.pdf(x, l[i] * N + 1, N - l[i] * N + 1) for i in range(len(l))], 0)

        # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
        # to get alpha and beta
        # We initialize the parameters at the average of each alpha/beta of each posterior
        param, _ = curve_fit(beta_func, x, approx_pdf,
                             p0=[np.mean(np.array(l) * N + 1), np.mean(N - np.array(l) * N + 1)],
                             method='trf', bounds=[0, np.inf], maxfev=2000)
        # ci
        temp, temp2 = credible_interval(param, ci=ci)

        b_lo_list.append(temp)
        b_up_list.append(temp2)
        # estimate
        estimate_list.append(estimate_calc(param, ci=ci))

        # p(B_s < B_m)
        p_over_list.append(g(param[0], param[1], param_[0], param_[1]))

    return b_lo, b_lo_list, b_up, b_up_list, estimate, estimate_list, p_over, p_over_list


def plot_fig_exp(N, list1, list2, model, mut_name, params, ci='mean'):
    fig = plt.figure(figsize=(3.5, 3.5))

    cmap = get_cmap('Set1')
    colors = list(cmap.colors)
    # removing yellow which is not very nice on plot
    del colors[5]

    colors = iter(colors)

    # Support of beta dist
    # Starting at 1e-8 for stability + step 1e-4 for better plot
    x = np.arange(1e-8, 1, 0.001)

    color = next(colors)
    # Plotting the B posteriors (calculated over each bootstrap samples)
    for i in range(len(list1)):
        plt.plot(x, beta.pdf(x, list1[i] * N + 1, N - list1[i] * N + 1), color=color, alpha=0.05)

    # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
    approx_pdf = np.mean(
        [beta.pdf(x, list1[i] * N + 1, N - list1[i] * N + 1) for i in range(len(list1))], 0)

    # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
    # to get alpha and beta
    # We initialize the parameters at the average of each alpha/beta of each posterior
    param, _ = curve_fit(beta_func, x, approx_pdf,
                         p0=[np.mean(np.array(list1) * N + 1), np.mean(N - np.array(list1) * N + 1)],
                         method='trf', bounds=[0, np.inf], maxfev=2000)

    plt.plot(x, approx_pdf, color=color, linestyle='dashed')
    plt.plot(x, beta.pdf(x, param[0], param[1]), color=color, label='original')

    # credible interval
    b_lo, b_up = credible_interval(param, ci=ci)
    # Adding/Substracting 1e-3 for better plot (so interval representation doesn't cut short)
    b_lo = b_lo-1e-3
    b_up = b_up+1e-3
    # print(b_lo, b_up)
    plt.fill_between(np.arange(b_lo, b_up, 0.001), beta.pdf(np.arange(b_lo, b_up, 0.001), param[0], param[1]),
                     color=color, alpha=0.5)

    # Using Mean as Bayes Estimator
    plt.axvline(estimate_calc((param[0], param[1]), ci), color=color, linestyle='dashed')

    # print("********")
    # Plot for each mutated bagged posterior
    for ind, l in enumerate(list2):
        color = next(colors)
        # Plotting the B posteriors (calculated over each bootstrap samples)
        for i in range(len(l)):
            plt.plot(x, beta.pdf(x, l[i] * N + 1, N - l[i] * N + 1), alpha=0.05, color=color)

        # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
        approx_pdf = np.mean(
            [beta.pdf(x, l[i] * N + 1, N - l[i] * N + 1) for i in range(len(l))], 0)

        # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
        # to get alpha and beta
        # We initialize the parameters at the average of each alpha/beta of each posterior
        param, _ = curve_fit(beta_func, x, approx_pdf,
                             p0=[np.mean(np.array(l) * N + 1), np.mean(N - np.array(l) * N + 1)],
                             method='trf', bounds=[0, np.inf], maxfev=2000)

        plt.plot(x, approx_pdf, color=color, linestyle='dashed')
        plt.plot(x, beta.pdf(x, param[0], param[1]), color=color, label=params[ind])

        # credible interval
        b_lo, b_up = credible_interval(param, ci=ci)
        b_lo = b_lo-1e-3
        b_up = b_up+1e-3
        plt.fill_between(np.arange(b_lo, b_up, 0.001), beta.pdf(np.arange(b_lo, b_up, 0.001), param[0], param[1]),
                         color=color, alpha=0.5)

        # Using Mean as Bayes Estimator
        plt.axvline(estimate_calc((param[0], param[1]), ci), color=color, linestyle='dashed')

    plt.xlim([0, 1])
    plt.ylim([0, 40])
    plt.xlabel(r'$\pi$')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    path = os.path.join('plot_results', model)

    if not os.path.isdir(path):
        os.mkdir(path)

    if svg_plot:
        plt.savefig(os.path.join(path, '{}_{}.svg'.format(model, mut_name)), format='svg', transparent=True, dpi=400)
    else:
        plt.savefig(os.path.join(path, '{}_{}.png'.format(model, mut_name)), dpi=400)
        plt.show()
