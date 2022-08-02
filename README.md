## Replication package for "A Probabilistic Framework for Mutation Testing in Deep Neural Networks" paper

This replication package contains all the scripts/data necessary to plot 
figures from our paper and redo experiments of our paper "A Probabilistic Framework for Mutation Testing in Deep Neural
Networks?" submitted to the journal of Information and Software Technology.

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
* [Calculating estimates](#calculating-estimates)
* [Calculating the Monte Carlo error over the instances (Bagged posterior stability)](#calculating-the-monte-carlo-error-over-the-instances-(bagged-posterior-stability))
* [Calculating the sampling effect for a given population size](#calculating-the-sampling-effect-for-a-given-population-size)
* [Generating the figure from the paper](#generating-the-figure-from-the-paper)
* [Making it works with your models/mutations/datasets](#making-it-works-with-your-modelsmutationsdatasets)

### Architecture 

. <br>
├── Datasets/ # directory for datasets<br>
├── README.md <br>
├── comp_deepcrime.py # Comparison script with deepcrime<br>
├── exp.py # Monte-Carlo simulation generation<br>
├── generate_acc_files.py # Generating accuracy file for MNIST<br>
├── generate_acc_files_lenet.py # Generating accuracy file for UnityEyes<br>
├── generate_acc_files_movie.py # Generating accuracy file for MovieRecomm<br>
├── mce_estim.py # Calculate MCE<br>
├── mutated_models # Files for training models <br>
│   ├── lenet <br>
│   ├── mnist <br>
│   └── movie <br>
├── mutations.py # DeepCrime file, necessary for training mutated models<br>
├── operators/ # DeepCrime files, necessary for training mutated models <br>
├── plot_param.py # Calculating estimates<br>
├── plot_posterior.py # Plotting figure such as Figure 3<br>
├── plot_results # Directory with all figures<br>
│   ├── lenet <br>
│   ├── mnist <br>
│   ├── movie_recomm <br>
├── pop_var.py # To generate figure similar to Figure 4/5<br>
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

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `comp_deepcrime.py`

If `--dc` is used, the scripts uses data in `raw_data/deepcrime_comp/` to yield the `p-value` and `cohen's d`, as well as the decision (Killed or not) for the test when comparing healthy instances against mutated instances. The instances used in that vae are the ones provided in DeepCrime's replication package. Results are already presented in `raw_data/deepcrime_comp/` in the `[model]_results_kill_DC.txt`.

If `--dc` is NOT used, the script will use DeepCrime's mutation test over multiple experiences using our instances, returning the average number of times the mutation test passed for each magntiude following the protocol we detailled in the Motivating Example of Section 4 in our paper.

files. 

Usage is:
```bash
python comp_deepcrime.py --model [model] --mut [mutation] [--dc]
```

For instance, using only their instances and their single test:
```bash
python comp_deepcrime.py --model 'mnist' --mut 'delete_training_data' --dc
```

```
Exp: delete_training_data_3.1
p-value 0.907, effect_size 0.03689417305183907
*********************
Exp: delete_training_data_9.29
p-value 0.048, effect_size 0.6241987215473714
Killed
*********************
Exp: delete_training_data_12.38
p-value 0.387, effect_size 0.2735018636400621
*********************
Exp: delete_training_data_18.57
p-value 0.001, effect_size 1.0074392666953291
Killed
*********************
Exp: delete_training_data_30.93
p-value 0.0, effect_size 1.6320878597970647
Killed
*********************
```

For instance, using multiple iterations of the test over our instances:
```bash
python comp_deepcrime.py --model 'mnist' --mut 'delete_training_data'
```

```
Average number of mutation test passed for healthy instances vs healthy instances: 0.06 (0.051)
Average number of mutation test passed for healthy instances vs mutated instances (delete_training_data_3.1): 0.13 (0.093)
Average number of mutation test passed for healthy instances vs mutated instances (delete_training_data_9.29): 0.45 (0.120)
Average number of mutation test passed for healthy instances vs mutated instances (delete_training_data_12.38): 0.47 (0.143)
Average number of mutation test passed for healthy instances vs mutated instances (delete_training_data_18.57): 0.85 (0.091)
Average number of mutation test passed for healthy instances vs mutated instances (delete_training_data_30.93): 1.00 (0.001)
```
### Calculating posterior distributions and plotting them

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `plot_posterior.py`
* `plot_results/`

To calculate the posterior distribution as we detailed in Section 5.1-3,
after calculating/putting accuracy files in the correct directory in `raw_data/`,
one can execute `plot_posterior.py`. This will generate a figure of the same
type as Figure 3 from our paper.

Usage is:
```bash
python plot_posterior.py --model [model] --mut [mutation]
```

For instance:
```bash
python plot_posterior.py --model 'mnist' --mut 'delete_training_data'
```

To allow for replication, the seed is fixed in this file. By default,
the figures are saved in `plot_results/[model]/` as a `.png` file. 

### Calculating estimates

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `plot_param.py`

This allows to print either the value of the estimates for a given model/mutation (if `--calc` is not provided) or which mutations are killed for a given set of estimates 
(if `--calc` is provided). To allow for replication, the seed is fixed in this file. 

Usage is:
```bash
python plot_param.py --model [model] --mut [mutation] [--calc  φ1 τ φ2]
```

For instance:
```bash
python plot_param.py --model 'mnist' --mut 'delete_training_data'
```

which will return the value of the estimates. For instance:
```
Healthy posterior: phi_1 0.06367970300566617, phi_2 0.500000000000011, CI [0.0;0.13612135106639317]
Mutation 3.1 posterior: phi_1 0.14022188211246783, phi_2 0.8356131547141236, CI [0.0;0.2832834778617216]
Mutation 9.29 posterior: phi_1 0.47628508431498234, phi_2 0.9991323744869056, CI [0.19224558883780712;0.7603245797921576]
Mutation 12.38 posterior: phi_1 0.4732371771875955, phi_2 0.9996577796614884, CI [0.2203072560269339;0.7261670983482571]
Mutation 18.57 posterior: phi_1 0.7967611130374709, phi_2 0.9999999939378351, CI [0.6042349599839285;0.9892872660910133]
Mutation 30.93 posterior: phi_1 0.9900980341560961, phi_2 1.0000000000000004, CI [0.970881423693573;1.0]
```

To calculate the number of mutations killed one can use:

For instance:
```bash
python plot_param.py --model 'mnist' --mut 'delete_training_data' --calc 0.8 0.4 0.95
```

which will return the number of mutations killed along with which mutation
didn't get killed (if any). For instance:
```
With φ1: 0.8, τ: 0.4, φ2: 0.95, the test set kills 2 mutations
Mutations not killed: [3.1, 9.29, 12.38, 18.57]
```

If we reduce φ1:
```
With φ1: 0.79, τ: 0.4, φ2: 0.95, the test set kills 3 mutations
Mutations not killed: [3.1, 9.29, 12.38]
```

Note that *not* killing the healthy instances posterior increment the number
of mutation killed (since it correctly rejected it as a mutation).

## Calculating the Monte Carlo error over the instances (Bagged posterior stability)

*Execution time:* ~ 20 min for `exp.py` and ~ 10 sec for `mce_estim.py`

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `exp.py`
* `rep_mce/`
* `mce_estim.py`

To calculate the Monte-Carlo error (MCE) as we detailed in Section 5.4,
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

Then use `mce_estim.py` to calculate the jackknife estimates as described in Section 5.4.
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

To calculate the Sampling Effect after calculating/putting accuracy files in the correct directory in `raw_data/`,
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
similarly to Figure 4/5. Note that this script requires both the 'original' data (e.g. `mnist_original_30_rep_X_size_pop.npy`)
and the data for the given mutation (since we need to estimate `p(B_m > B_s)`).


Usage is:
```bash
python pop_var.py --model [model] --mut [mutation] [--param [parameter] ] [--pop_size [size] ] [--proc n]
```

All parameters are as before except `--pop_size` instead of `--size`. If the flag
is not provided, the program will generate the figure similarly to Figure 4/5, except that it will
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

*Motivating example*

Run the following script
```bash
python comp_deepcrime.py --model mnist --mut delete_training_data
```

*Figure 3*

Run the following script
```bash
python plot_posterior.py --model 'mnist' --mut 'delete_training_data'
python plot_posterior.py --model 'mnist' --mut 'change_activation_function'
python plot_posterior.py --model 'movie_recomm' --mut 'delete_training_data'
python plot_posterior.py --model 'movie_recomm' --mut 'unbalance_train_data'
python plot_posterior.py --model 'lenet' --mut 'delete_training_data'
python plot_posterior.py --model 'lenet' --mut 'change_label'
```

Data need to be present in `raw_data/{model}/` (see [here](#generating-the-accuracy-data)). Figure will be
saved to `plot_results/{model}/`. 

*Figure 4*

Run the following script
```bash
python pop_var.py --model 'mnist' --mut 'delete_training_data' --param 3.1
python pop_var.py --model 'mnist' --mut 'delete_training_data' --param 9.29
python pop_var.py --model 'mnist' --mut 'delete_training_data' --param 30.93
```

Data need to be present in `raw_data/{model}/` (see [here](#generating-the-accuracy-data)). Figure will be
saved to `plot_results/{model}/data_plot/`.

*Figure 5*

Run the following script
```bash
python pop_var.py --model 'mnist' --mut 'change_label' --param 3.12
python pop_var.py --model 'movie_recomm' --mut 'change_label' --param 3.12
python pop_var.py --model 'lenet' --mut 'change_label' --param 3.12
```

Data need to be present in `raw_data/{model}/` (see [here](#generating-the-accuracy-data)). Figure will be
saved to `plot_results/{model}/data_plot/`.

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
in [Calculating estimates](#calculating-estimates).
With a sufficient number of training instances, it is not even needed to
calculate Monte-Carlo error or Sample size effect as we showed in the paper,
so the more time-consuming operations are removed.



