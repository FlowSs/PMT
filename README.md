## Replication package for "Mutation Testing of Deep Learning: Are We There Yet?" paper

This replication package contains all the scripts/data necessary to plot 
figures from our paper and redo experiments of our paper "Mutation Testing of Deep Learning: Are We There Yet?"
submitted to the 33rd IEEE International Symposium on Software Reliability Engineering (ISSRE2022).

We also provide a quick way to adapt the framework for custom models/datasets/mutations.

Note that the package was designed to be functional and facilitate replication of the experiment,
as such it's not as optimized speed wise as it could be.

### A note on Replication

If you wish to replicate completely from scratch, you will need to download
the trained instances ([here](#generating-the-accuracy-data)) and rerun the script as described
below. Note that, however, some script
can take a few hours even with parallelization and the archive for all models instances is quite large (> 100 GB Total).

Note that if you were to retrain the instances, you wouldn't get exactly the same
results as us, but it should be pretty similar to us (it is the point of the method
after all). Yet, it requires quite some time to train the instances.

In any case, we provide all data necessary to run the script below if one just wish to replicate our figures
as well as the ones in `plot_results/`.

### Index

* [Architecture](#architecture)
* [Requirements](#requirements)
* [Training models based on DeepCrime mutations](#training-models-based-on-deepcrime-mutations)
* [Generating the accuracy data](#generating-the-accuracy-data)
* [Running mutation testing on DeepCrime models](#running-mutation-testing-on-deepcrime-models)
* [Calculating posterior distributions and plotting them](#calculating-posterior-distributions-and-plotting-them)
* [Calculating estimates regions and plotting them](#calculating-estimates-regions-and-plotting-them)
* [Calculating the Monte Carlo error over the instances](#calculating-the-monte-carlo-error-over-the-instances)
* [Calculating the sampling effect for a given population size](#calculating-the-sampling-effect-for-a-given-population-size)
* [Generating the figure from the paper](#generating-the-figure-from-the-paper)
* [Making it works with your models/mutations/datasets](#making-it-works-with-your-modelsmutationsdatasets)

### Architecture 

. <br>
├── Datasets/ # directory for datasets<br>
├── README.md <br>
├── comp_deepcrime.py # Comparison script with deepcrime<br>
├── exp.py # Monte-Carlo simulation generation<br>
├── exp_into.py # Script for Figure 1<br>
├── generate_acc_files.py # Generating accuracy file for MNIST<br>
├── generate_acc_files_lenet.py # Generating accuracy file for UnityEyes<br>
├── generate_acc_files_movie.py # Generating accuracy file for MovieRecomm<br>
├── mce_estim.py # Calculate MCE (RQ2, Section III.D)<br>
├── mutated_models # Files for training models <br>
│   ├── lenet <br>
│   ├── mnist <br>
│   └── movie <br>
├── mutations.py # DeepCrime file, necessary for training mutated models<br>
├── operators/ # DeepCrime files, necessary for training mutated models <br>
├── plot_param.py # Plotting figure such as Figure 4<br>
├── plot_posterior.py # Plotting figure such as Figure 3<br>
├── plot_results # Directory with all figures<br>
│   ├── exp_intro.pgf # Figure 1<br>
│   ├── lenet <br>
│   ├── lenet_change_optimisation_function_std_just_paper.pgf # Figure 5<br>
│   ├── mnist <br>
│   ├── mnist_delete_training_data_std_just_paper.pgf # Figure 5 <br>
│   ├── movie_recomm <br>
│   ├── p.npy # Data for Figure 1<br>
│   ├── p2.npy # Data for Figure 1 <br>
│   └── p3.npy # Data for Figure 1 <br>
├── pop_var.py # To generate figure similar to Figure 5<br>
├── pop_var_just_paper.py # Script for Figure 5<br>
├── raw_data # Raw data (accuracy files)<br>
│   ├── deepcrime_comp <br>
│   ├── lenet <br>
│   ├── mnist <br>
│   └── movie_recomm <br>
├── rep_mce # Data of Monte-Carlo simulations (200 instances)<br>
│   ├── lenet <br>
│   ├── mnist <br>
│   └── movie_recomm <br>
├── rep_practicality # Data of 30 repetitions of Monte-Carlo simulation for different population size<br>
│   ├── lenet <br>
│   ├── mnist <br>
│   └── movie_recomm <br>
├── requirements.txt <br>
├── run_mp.py # Getting 30 repetitions of the Monte-Carlo simulation of different population size<br>
├── settings.py # Settings file<br>
├── trained_models/ # Directory to put our trained models<br>
├── trained_models_dc/ # Directory to put DeepCrime directory<br>
├── utils/ # DeepCrime files, necessary for training mutated models <br>
└── utils.py # utils file (plot functions as well as math related functions)<br>

### Requirements

The framework uses *Python 3.8* and we recommend creating a `conda` environment
to manage the different packages.

The base requirements are the same as DeepCrime provided *requirements38.txt*.

```
numpy~=1.18.5 
tensorflow~=2.3.0 
tensorflow-gpu~=2.3.0 
Keras~=2.4.3 
matplotlib~=3.3.0 
progressbar~=2.5 
scikit-learn~=0.23.1 
termcolor~=1.1.0 
h5py~=2.10.0 
pandas~=1.1.0 
statsmodels~=0.11.1 
opencv-python~=4.3.0.36 
networkx~=2.5.1 
patsy~=0.5.1 
scipy~=1.4.1 
```

We extend those requirements with some additional packages needed in our scripts:
```
tqdm~=4.64.0
colorama~=0.4.4 
tqdm-multiprocess~=0.0.11
```

All package required are present in the `requirements.txt` file.

### Training models based on DeepCrime mutations

*Files/Directory concerned*: <br>

* `utils/`
* `operators/`
* `mutations.py`
* `mutated_models/`
* `Datasets/`


We extracted files relevant to train models based on given a given mutation, 
mainly properties and operators DeepCrime used. We generated the training files
according to DeepCrime definition, they all can be found in `mutated_models/`. Files are 
by default programmed to train 200 instances, only if instance *i-th* instance
doesn't exist, for the defined mutations. Models used dataset from DeepCrime
(not included in package) put into `Datasets/`.


### Generating the accuracy data

*Files/Directory concerned*: <br>

* `trained_models/`
* `trained_models_dc/`
* `raw_data/`

Trained models can be downloaded as `.zip` files. 

All trained instances are hosted anonymously on [Zenodo](https://zenodo.org/).

*Zenodo Packages*
* [zenodo_1](https://zenodo.org/record/6561382) corresponds to part of mnist models.
* [zenodo_2](https://zenodo.org/record/6577005) corresponds to rest of mnist models, movie models and deepcrime models for comparison.
* [zenodo_3](https://zenodo.org/record/6581962) corresponds to lenet models.

All files inside the `trained_models/` directory inside the `.zip`
should then be extracted into relevant directory (`trained_models_dc/` for 
`deepcrime_models.zip`, `trained_models/` for all others). Then, `generate_acc_files*.py`
can be executed (where `*` can be ` `, `_movie` or `_lenet`) to generate
the accuracy files for each model/mutation. Usage is:
```bash
python generate_acc_files.py --name [mutation] (--comp)
```
where `--comp` flag instructs program to use `trained_models_dc/` instead of
`trained_models/`. For instance:
```bash
python generate_acc_files.py --name 'change_label_mutated0_MP_3.12'
```
will generate the accuracy file `mnist_change_label_mutated0_MP_3.12.csv` in `raw_data/mnist/`.

### Running mutation testing on DeepCrime models

*Files/Directory concerned*: <br>

* `raw_data/deepcrime_comp/`
* `utils.py`
* `comp_deepcrime.py`

Once accuracy files for DeepCrime comparison models are in `raw_data/deepcrime_comp/`
(see [here](#generating-the-accuracy-data)), running `comp_deepcrime.py` will yield the `p-value` and `cohen's d`
, as well as the decision (Killed or not) for the test when comparing sound instances against mutated instances as presented in
the Section III.A of our paper. Results are already presented in `raw_data/deepcrime_comp/` in the `[model]_results_kill_DC.txt`
files. 

Usage is:
```bash
python comp_deepcrime.py --model [model] --mut [mutation]
```

For instance:
```bash
python comp_deepcrime.py --model 'mnist' --mut 'change_label'
```

### Calculating posterior distributions and plotting them

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `plot_posterior.py`
* `plot_results/`

To calculate the posterior distribution as we detailed in Section III.B-C,
after calculating/putting accuracy files in the correct directory in `raw_data/`,
one can execute `plot_posterior.py`. This will generate a figure of the same
type as Figure 3 from our paper.

Usage is:
```bash
python plot_posterior.py --model [model] --mut [mutation]
```

For instance:
```bash
python plot_posterior.py --model 'mnist' --mut 'change_label'
```

To allow for replication, the seed is fixed in this file. By default,
the figures are saved in `plot_results/[model]/` as a `.png` file. In our
replication package, we provide them as `.pgf` files. This can be chosen by setting
`False` or `True` the parameter `pgf_plot` in `utils.py`.

### Calculating estimates regions and plotting them

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `plot_param.py`
* `plot_results/`

This allow to plot heatmaps of couple estimates as Figure 4 in our paper. After calculating/putting accuracy files in the correct directory in `raw_data/`,
one can execute `plot_param.py`. 

Usage is:
```bash
python plot_param.py --model [model] --mut [mutation] [--calc  φ1 τ φ2]
```

For instance:
```bash
python plot_param.py --model 'mnist' --mut 'change_label'
```

To allow for replication, the seed is fixed in this file. By default,
the figures are saved in `plot_results/[model]/` as a `.png` file. In our
replication package, we provide them as `.pgf` files in the directory `plot_results/[model]/[model]_[mutation]_param/`. 
This can be chosen by setting `False` or `True` the parameter `pgf_plot` in `utils.py`.

To calculate the number of mutations killed one can use:

For instance:
```bash
python plot_param.py --model 'mnist' --mut 'delete_training_data' --calc 0.8 0.2 0.95
```

which will return the number of mutations killed along with which mutation
didn't get killed (if any). For instance:
```
With φ1: 0.8, τ: 0.2, φ2: 0.95, the test set kills 5 mutations
Mutations not killed: [3.1]
```

If we reduce φ2:
```
With φ1: 0.8, τ: 0.2, φ2: 0.83, the test set kills 6 mutations
Mutations not killed: []
```

Note that *not* killing the sound instances posterior increment the number
of mutation killed (since it correctly rejected it as a mutation).

## Calculating the Monte Carlo error over the instances

*Execution time:* ~ 20 min for `exp.py` and ~ 10 sec for `mce_estim.py`

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `exp.py`
* `rep_mce/`
* `mce_estim.py`

To calculate the Monte-Carlo error (MCE) as we detailed in Section III.D,
after calculating/putting accuracy files in the correct directory in `raw_data/`,
one needs to first execute `exp.py` to generate monte-carlo simulation data.

Usage is:
```bash
python exp.py --model [model] --mut [mutation] [--param [parameter] ] [--proc n]
```

The `model` and `mut` parameters are the same as before. `param` controls the
mutation magnitude/parameter. By default (if flag is not used), uses `original` models.
`proc` parameter control the number of core to use (for parallalelisation).
By default, only one is used.

For instance:
```bash
python exp.py --model 'mnist' --mut 'change_label' --param 3.12 --proc 8
```

This will generate a file such as `mnist_change_label_3.12_200_pop_size.npy`
in `rep_mce/mnist/` using 8 cores for instance.

Then use `mce_estim.py` to calculate the jackknife estimates as described in Section III.D.
Note that this script requires both the 'original' data (e.g. `mnist_original_200_pop_size.npy`)
and the data for the given mutation (since we need to estimate `p(B_m > B_s)`).

Usage is:
```bash
python mce_estim.py --model [model] --mut [mutation] [--param [parameter] ]
```

The definition is the same as `exp.py`.

For instance:
```bash
python mce_estim.py --model 'mnist' --mut 'change_label' --param 3.12
```

Which will returns:

```
Model mnist, Mutation change_label (Param 3.12)
Point estimate: Avg 0.3729445274106166, 95% Confidence Interval (0.3703295930758904, 0.3755594617453428)
Credible Interval bounds:
 Lower bound Avg 0.11744689035372904, 95% Confidence Interval (0.11443951735324402, 0.12045426335421407)
 Upper bound 95% Avg 0.6284421644675042, 95% Confidence Interval (0.6235663552272548, 0.6333179737077537)
p(B_s < B_m): Avg 0.9950197301756303, 95% Confidence Interval: (0.9945900292545282, 0.9954494310967323)
```

## Calculating the sampling effect for a given population size

*Execution time:* ~2 hours with 30 cores for `run_mp.py` (see note below to improve speed)
and ~ 5 min with 8 cores for `pop_var.py` (instant if data have already been calculated and
we just want to plot/print results).

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `pop_var.py`
* `rep_practicality/`
* `run_mp.py`
* `plot_results/`

To calculate the Sampling Effect as we detailed in RQ2,
after calculating/putting accuracy files in the correct directory in `raw_data/`,
one needs to first execute `run_mp.py` to generate monte-carlo simulations for different population size.

Usage is:
```bash
python run_mp.py --model [model] --mut [mutation] [--size [size] ] [--param [parameter] ] [--proc n]
```

The `model` and `mut` parameters are the same as before. `param` controls the
mutation magnitude/parameter. By default (if flag is not used), uses `original` models.
`size` controls the sample size (default 100). `proc` parameter control the number of core to use (for parallalelisation).
By default, only one is used.

For instance:
```bash
python run_mp.py --model 'mnist' --mut 'change_label' --size 25 --param 3.12 --proc 8
```

This will generate a file such as `mnist_change_label_3.12_30_rep_25_size_pop.npy`
in `rep_practicality/mnist/` using 8 cores for instance.

*Note:* this part can actual be long (around ~2 hours with 30 cores) with
base parameters. You can reduce the value of `exp_n` (number of monte carlo simulation)
to reduce the computation time without affecting much the results. One may also reduce the value of 
`B` (number of bootstrap) to further decrease it but at the possible cost of increased error.

Then use `pop_var.py` to calculate confidence interval over the jackknife estimates 
similarly to Figure 5. Note that this script requires both the 'original' data (e.g. `mnist_original_30_rep_X_size_pop.npy`)
and the data for the given mutation (since we need to estimate `p(B_m > B_s)`).


Usage is:
```bash
python pop_var.py --model [model] --mut [mutation] [--param [parameter] ] [--pop_size [size] ] [--proc n]
```

All parameters are as before except `--pop_size` instead of `--size`. If the flag
is not provided, the program will generate the figure similarly to Figure 5, except that it will
return all estimates for one mutation/parameter instead of one estimate for
mutliple models/parameters (see [here](#generating-the-figure-from-the-paper) to replicate figure). In the paper, we chose to return one estimate
for different parameters/mutations for comparison purpose and due to the size limite. Yet, it is normally intended
as it is presented here.

For instance:
```bash
python pop_var.py --model 'mnist' --mut 'change_label' --param 3.12 --proc 8
```

Which will return a figure in `plot_results/mnist/std/`. To only output the results for a given population size.

For instance:
```bash
python pop_var.py --model 'mnist' --mut 'change_label' --param 3.12 --proc 8 --pop_size 25
```

```
*** Results ***
Pop size 25

Average and Std of each estimate:
Point estimate Mean: 0.36755610481538925, CI: (0.02465568699155961, 0.9781961154261367)
Credible Interval bounds:
 Lower bound Mean: 0.0005214651377456592, CI: (0.0, 0.9143175665448149)
 Upper bound Mean: 0.7810812342875815, CI: (0.07236306718265624, 1.0)
p(B_s < B_m) Mean: 0.8754518593020448, CI: (0.11752088132758924, 0.9950673692429179)

Average and Std of each estimate lower bound:
Point estimate Mean: 0.33807077036262945, CI: (0.02379516342999382, 0.9711134582096383)
Credible Interval bounds:
 Lower bound Mean: 0.0, CI: (-7.153503865406474e-05, 0.8879297094081527)
 Upper bound Mean: 0.7409383441300782, CI: (0.06984524305466161, 0.9999999999999978)
p(B_s < B_m) Mean: 0.8633018499133897, CI: (0.10262331035687283, 0.9944409394343534)

Average and Std of each estimate upper bound:
Point estimate Mean: 0.39704143926814905, CI: (0.025516210553125394, 0.9852787726426351)
Credible Interval bounds:
 Lower bound Mean: 0.0011305221329106837, CI: (0.0, 0.9407054236814771)
 Upper bound Mean: 0.8212241244450846, CI: (0.07488089131065088, 1.0008172199472283)
p(B_s < B_m) Mean: 0.8876018686906999, CI: (0.13241845229830565, 0.9960048363213437)
```

### Generating the figure from the paper

In all case, the figures are already present in their respective directory,
however here are how to re-generate them:

*Figure 1*

Run the following script
```bash
python exp_intro.py [--proc n]
```

where `--proc` is the number of cores to use as defined previously. If `plot_results/p*.npy` 
are not present, they will be recomputed and figure might differ slightly. Otherwise,
load them to get the exact figure.

*Figure 3*

Run the following script
```bash
python plot_posterior.py --model 'mnist' --mut 'delete_training_data'
```

Data need to be present in `raw_data/mnist/` (see [here](#generating-the-accuracy-data)). Figure will be
saved to `plot_results/mnist/`. By default, as a `.png` but it can be plotted as `.pgf` by
setting `pgf_plot = True` in `utils.py`.

*Figure 4*

Run the following script
```bash
python plot_param.py --model 'mnist' --mut 'delete_training_data'
```

Data need to be present in `raw_data/mnist/` (see [here](#generating-the-accuracy-data)). Figure will be
saved to `plot_results/mnist/`. By default, as a `.png` but it can be plotted as `.pgf` by
setting `pgf_plot = True` in `utils.py`.

*Figure 5*

Run the following script
```bash
python pop_var_just_paper.py --model 'mnist' --mut 'delete_training_data'
python pop_var_just_paper.py --model 'lenet' --mut 'change_optimisation_function'
```

Data need to be present in `rep_practicality/{mnist|lenet}/data_plot/` (see [here](#calculating-the-sampling-effect-for-a-given-population-size)). Figure will be
saved to `plot_results/`. By default, as a `.png` but it can be plotted as `.pgf` by
setting `pgf_plot = True` in `utils.py`.

### Making it works with your models/mutations/datasets

If the code is intended to be more of a replication package than a general
framework, it is pretty easy to change it to have it work with any model/mutation/dataset.

To do so, the only requirements are:

* Having trained the instances (sound and mutated) one wish to apply the framework on.
* Having generated the accuracy files similarly to [here](#generating-the-accuracy-data). Please make sure the structure
is the same as other files in `raw_data/`. Note that the files `generate_acc_files*.py` works only for the models we studied. Yet they
can serve as a template for custom models. The files then need to be put in `raw_data/[model]/`.

Once this is done, you need to edit `settings.py` and add to `main_dict`
you model as well as the mutation label/parameter similarly to studied models/mutations.

After that, you can calculate the number of mutations killed for given parameters or
study the parameters space to kill a given number of mutation such as explained
in [Calculating estimates regions and plotting them](#calculating-estimates-regions-and-plotting-them).
With a sufficient number of training instances, it is not even needed to
calculate Monte-Carlo error or Sample size effect as we showed in the paper,
so the more time-consuming operations are removed.



