import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from scipy.optimize import curve_fit
from scipy.optimize import fmin
from scipy.special import gamma, gammaln, betainc
from scipy.stats import beta

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


def hellinger_distance(a1, b1, a2, b2):
    """
    Calculate the Hellinger distance between two beta distributions

    """
    return np.sqrt(np.clip(1 - beta_coeff((a1+a2)/2, (b1+b2)/2)/(np.sqrt(beta_coeff(a1, b1)*beta_coeff(a2, b2))), 0, 1))

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

    plt.savefig(os.path.join('plot_results', 'exp_intro.png'), dpi=400)
    plt.show()


def plot_pop_std(l_std, model, mut_name, param):
    plt.rc('font', size=10)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5., 3.8))
    l_std = np.array(l_std)

    x = np.array([25, 50, 75, 100, 125, 150, 175, 200])
    # Plot on the estimate point
    ax1.errorbar(x, [l_std[i][0][0][0] for i in range(len(l_std))],
                       [[l_std[i][0][0][0] - l_std[i][0][0][1] for i in range(len(l_std))],
                        [l_std[i][0][0][2] - l_std[i][0][0][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    ax1.errorbar(x - 7, [l_std[i][1][0][0] for i in range(len(l_std))],
                       [[l_std[i][1][0][0] - l_std[i][1][0][1] for i in range(len(l_std))],
                        [l_std[i][1][0][2] - l_std[i][1][0][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    ax1.errorbar(x + 7, [l_std[i][2][0][0] for i in range(len(l_std))],
                       [[l_std[i][2][0][0] - l_std[i][2][0][1] for i in range(len(l_std))],
                        [l_std[i][2][0][2] - l_std[i][2][0][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    ax1.set_title('Mean')
    ax1.set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    ax1.set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    ax1.set_yticks(np.arange(0, 1.01, 0.1))
    for n, label in enumerate(ax1.yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    ax1.set_yticklabels(np.round(ax1.get_yticks(), 1), rotation=45)
    ax1.set_ylim([0, 1])
    ax1.set_xlabel('Sampled population size')
    ax1.set_ylabel('Value')
    ax1.grid(visible=True, which='major', color='black', alpha=0.3)

    # Plot on the Credible interval lower bound
    ax2.errorbar(x, [l_std[i][0][1][0] for i in range(len(l_std))],
                       [[l_std[i][0][1][0] - l_std[i][0][1][1] for i in range(len(l_std))],
                        [l_std[i][0][1][2] - l_std[i][0][1][0] for i in range(len(l_std))]],
                       fmt="o", c='blue', capsize=2)
    ax2.errorbar(x - 7, [l_std[i][1][1][0] for i in range(len(l_std))],
                       [[l_std[i][1][1][0] - l_std[i][1][1][1] for i in range(len(l_std))],
                        [l_std[i][1][1][2] - l_std[i][1][1][0] for i in range(len(l_std))]],
                       fmt="o", c='green', capsize=2)
    ax2.errorbar(x + 7, [l_std[i][2][1][0] for i in range(len(l_std))],
                       [[l_std[i][2][1][0] - l_std[i][2][1][1] for i in range(len(l_std))],
                        [l_std[i][2][1][2] - l_std[i][2][1][0] for i in range(len(l_std))]],
                       fmt="o", c='orange', capsize=2)
    ax2.set_title('Variance')
    ax2.set_xticks([25, 50, 75, 100, 125, 150, 175, 200])
    ax2.set_xticklabels([25, 50, 75, 100, 125, 150, 175, 190], rotation=45)
    ax2.set_yticks(np.arange(0, 0.11, 0.01))
    for n, label in enumerate(ax2.yaxis.get_ticklabels()):
        if n % 2 != 0:
            label.set_visible(False)
    ax2.set_yticklabels(np.round(ax2.get_yticks(), 2), rotation=45)
    ax2.set_ylim([0, 0.1])
    ax2.set_xlabel('Sampled population size')
    ax2.grid(visible=True, which='major', color='black', alpha=0.3)

    plt.tight_layout()

    # Making sure directory exists
    if not os.path.isdir('plot_results'):
        os.mkdir('plot_results')

    if not os.path.isdir(os.path.join('plot_results', model)):
        os.mkdir(os.path.join('plot_results', model))

    path = os.path.join('plot_results', model, 'std')

    if not os.path.isdir(path):
        os.mkdir(path)

    if mut_name == 'original':
        plt.savefig(os.path.join(path, '{}_{}_std.png'.format(model, mut_name)), dpi=400)
        plt.show()
    else:
        plt.savefig(os.path.join(path, '{}_{}_{}_std.png'.format(model, mut_name, param)), dpi=400)
        plt.show()


def jack_estimate(N, list_of_rep_mut):
    # Support of beta dist
    # Starting at 1e-8 for stability
    x = np.arange(1e-8, 1, 0.01)

    mean_list, var_list = [], []
    for list_mut in list_of_rep_mut:
        # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
        approx_pdf_mut = np.mean(
            [beta.pdf(x, list_mut[i] * N + 1, N - list_mut[i] * N + 1) for i in range(len(list_mut))], 0)

        # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
        # to get alpha and beta
        # We initialize the parameters at the average of each alpha/beta of each posterior
        param_mut, _ = curve_fit(beta_func, x, approx_pdf_mut,
                                 p0=[np.mean(np.array(list_mut) * N + 1), np.mean(N - np.array(list_mut) * N + 1)],
                                 method='trf', bounds=[0, np.inf], maxfev=2000)

        mean_list.append(beta.mean(param_mut[0], param_mut[1]))
        var_list.append(beta.var(param_mut[0], param_mut[1]))

    mean_list, var_list = np.array(mean_list), np.array(var_list)

    # jackknife estimate list
    jack_mean_list, jack_var_list = [], []
    for k in range(len(mean_list)):
        mask = [True if i != k else False for i in range(len(mean_list))]
        jack_mean_list.append(np.sum(mean_list[mask]) / len(mean_list))
        jack_var_list.append(np.sum(var_list[mask]) / len(mean_list))

    # calculating each MCE
    jack_mean, jack_var = 0, 0
    for i in range(len(mean_list)):
        jack_mean += (jack_mean_list[i] - np.mean(jack_mean_list)) ** 2
        jack_var += (jack_var_list[i] - np.mean(jack_var_list)) ** 2

    jack_mean = np.sqrt(((len(mean_list) - 1) / len(mean_list)) * jack_mean)
    jack_var = np.sqrt(((len(mean_list) - 1) / len(mean_list)) * jack_var)

    # Return the MCE estimate with the MCE error
    return [(np.mean(mean_list), jack_mean), (np.mean(var_list), jack_var)]


def calcul_param(N, list1, list2, ci='mean'):
    # Support of beta dist
    # Starting at 1e-8 for stability
    x = np.arange(1e-8, 1, 0.001)

    param_mut = [101, 1]
    param_healthy = [1, 101]


    print("Approximating original instances posterior")
    # Using Bayes Bagging, bagged posterior is approximately the average of each bootstrap posterior
    approx_pdf = np.mean(
        [beta.pdf(x, list1[i] * N + 1, N - list1[i] * N + 1) for i in range(len(list1))], 0)
    # Since we know the bagged posterior should be beta, fit approximate pdf data using least square
    # to get alpha and beta
    # We initialize the parameters at the average of each alpha/beta of each posterior
    param, _ = curve_fit(beta_func, x, approx_pdf,
                          p0=[np.mean(np.array(list1) * N + 1), np.mean(N - np.array(list1) * N + 1)],
                          method='trf', bounds=[0, np.inf], maxfev=2000)
    # credible interval
    # b_lo, b_up = credible_interval(param_, ci=ci)
    # estimate
    # estimate = estimate_calc(param_, ci=ci)
    # p(B_s < B_s)
    # p_over = g(param_[0], param_[1], param_[0], param_[1])
    orig_to_healthy = hellinger_distance(param[0], param[1], param_healthy[0], param_healthy[1])
    orig_to_mut = hellinger_distance(param[0], param[1], param_mut[0], param_mut[1])
    
    # Calc for each mutated bagged posterior
    # b_lo_list, b_up_list, estimate_list, p_over_list = [], [], [], []
    post_to_healthy = []
    post_to_mut = []
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
        # temp, temp2 = credible_interval(param, ci=ci)

        # b_lo_list.append(temp)
        # b_up_list.append(temp2)
        # estimate
        # estimate_list.append(estimate_calc(param, ci=ci))

        # p(B_s < B_m)
        # p_over_list.append(g(param[0], param[1], param_[0], param_[1]))
        post_to_healthy.append(hellinger_distance(param[0], param[1], param_healthy[0], param_healthy[1]))
        post_to_mut.append(hellinger_distance(param[0], param[1], param_mut[0], param_mut[1]))

        
    return orig_to_healthy, orig_to_mut, post_to_healthy, post_to_mut


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

    plt.savefig(os.path.join(path, '{}_{}.png'.format(model, mut_name)), dpi=400)
    plt.show()
