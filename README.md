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
* [Calculating Ratio](#calculating-ratio)
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
* [zenodo_1](https://zenodo.org/record/6561382) corresponds to part of mnist models (*source* level mutations)
* [zenodo_2](https://zenodo.org/record/6577005) corresponds to rest of mnist models, movie models and deepcrime models for comparison. (*source* level mutations)
* [zenodo_3](https://zenodo.org/record/6581962) corresponds to lenet models. (*source* level mutations)
* zenodo_4 (TBD) corresponds to *model* level mutations of mnist/lenet.

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

`--same` needs to be used for *model* level mutation so the healthy instances are compared to their mutated counter-part.

files. 

Usage is:
```bash
python comp_deepcrime.py --model [model] --mut [mutation] [--dc] [--same]
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
python plot_posterior.py --model [model] --mut [mutation] [--same]
```

`--same` needs to be used for *model* level mutation so the healthy instances are compared to their mutated counter-part.

For instance:
```bash
python plot_posterior.py --model 'mnist' --mut 'delete_training_data'
```

To allow for replication, the seed is fixed in this file. By default,
the figures are saved in `plot_results/[model]/` as a `.png` file. 

### Calculating Ratio

*Execution time:* ~1 min/mutation

*Files/Directory concerned*: <br>

* `raw_data/`
* `utils.py`
* `plot_param.py`

This allows to print the Hellinger distances between the considered mutation and the ideal non-mutant/mutant posteriors, as well as the similarity ratio. 

Usage is:
```bash
python plot_param.py --model [model] --mut [mutation] [--same]
```

`--same` needs to be used for *model* level mutation so the healthy instances are compared to their mutated counter-part.

For instance:
```bash
python plot_param.py --model 'mnist' --mut 'delete_training_data'
```

which will return the value of the ratio and Hellinger distances. For instance:
```
Original: Hellinger distance to ideal non-mutant posterior: 0.75663, Hellinger distance to ideal mutant posterior: 1.00000
Ratio:  0.7566258775312085
Mut: 3.1, Hellinger distance to ideal non-mutant posterior: 0.91377, Hellinger distance to ideal mutant posterior: 0.9999999994857593
Ratio:  0.9137732097912725
Mut: 9.29, Hellinger distance to ideal non-mutant posterior: 0.99836, Hellinger distance to ideal mutant posterior: 0.9992365636693191
Ratio:  0.9991240438353294
Mut: 12.38, Hellinger distance to ideal non-mutant posterior: 0.99942, Hellinger distance to ideal mutant posterior: 0.999789474624726
Ratio:  0.9996269920003796
Mut: 18.57, Hellinger distance to ideal non-mutant posterior: 1.00000, Hellinger distance to ideal mutant posterior: 0.9544110921908092
Ratio:  1.0477663646846322
Mut: 30.93, Hellinger distance to ideal non-mutant posterior: 1.00000, Hellinger distance to ideal mutant posterior: 0.0035531532521594113
Ratio:  281.44015442966185
```

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
python exp.py --model [model] --mut [mutation] [--param [parameter] ] [--proc n] [--same]
```

The `model` and `mut` parameters are the same as before. `param` controls the
mutation magnitude/parameter. By default (if flag is not used), uses `original` models.
`proc` parameter control the number of core to use (for parallalelisation).
By default, only one is used.

`--same` needs to be used for *model* level mutation so the healthy instances are compared to their mutated counter-part.

For instance:
```bash
python exp.py --model 'mnist' --mut 'change_label' --param 3.12 --proc 8
```

This will generate a file such as `mnist_change_label_3.12_200_pop_size.npy`
in `rep_mce/mnist/` using 8 cores for instance.

Then use `mce_estim.py` to calculate the jackknife estimates as described in Section 4.5.

Usage is:
```bash
python mce_estim.py --model [model] --mut [mutation] [--param [parameter] ]
```

For instance:
```bash
python mce_estim.py --model 'mnist' --mut 'change_label' --param 3.12
```

Which will returns:

```
Model mnist, Mutation change_label (Param 3.12)
Mean: Avg 0.372944527433851, 95% Confidence Interval (0.37032959308138835, 0.3755594617863136)
Variance: Avg 0.017058179255132526, 95% Confidence Interval: (0.016651012000556725, 0.017465346509708327)
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
python run_mp.py --model [model] --mut [mutation] [--size [size] ] [--param [parameter] ] [--proc n] [--same]
```

The `model` and `mut` parameters are the same as before. `param` controls the
mutation magnitude/parameter. By default (if flag is not used), uses `original` models.
`size` controls the sample size (default 100). `proc` parameter control the number of core to use (for parallalelisation).
By default, only one is used.

`--same` needs to be used for *model* level mutation so the healthy instances are compared to their mutated counter-part.

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
similarly to Figure 4/5. 


Usage is:
```bash
python pop_var.py --model [model] --mut [mutation] [--param [parameter] ] [--pop_size [size] ] [--proc n]
```

All parameters are as before except `--pop_size` instead of `--size`. If the flag
is not provided, the program will generate the figure similarly to Figure 4/5, except that it will
return all estimates for one mutation/parameter instead of one estimate for
mutliple models/parameters (see [here](#generating-the-figure-from-the-paper) to replicate figure).

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
Mean: 0.36755610481538925, Confidence Interval: (0.02465568699155961, 0.9781961154261367)
Variance: 0.03652421636542531, Confidence Interval: (0.00045654335750781644, 0.09541987606691003)

Average and Std of each estimate lower bound:
Mean: 0.33807077036262945, Confidence Interval: (0.02379516342999382, 0.9711134582096383)
Variance: 0.03225569290650174, Confidence Interval: (0.00042724310191832773, 0.0942625596384591)

Average and Std of each estimate upper bound:
Mean: 0.39704143926814905, Confidence Interval: (0.025516210553125394, 0.9852787726426351)
Variance: 0.04079273982434889, Confidence Interval: (0.0004858436130973051, 0.0972828016883109)
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

After that, you can plot the posterior for analysis such as [Calculating posterior distributions and plotting them](#calculating-posterior-distributions-and-plotting-them)
or calculate the similarity ratio similarly to [Calculating Ratio](#calculating-ratio), in order to decide which mutation is killed.
With a sufficient number of training instances, it is not even needed to calculate Monte-Carlo error or Sample size effect as we showed in the paper,
so the more time-consuming operations are removed.



