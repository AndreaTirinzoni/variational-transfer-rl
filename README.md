# Transfer of Value Functions via Variational Methods

This repository contains the code used for the empirical evaluation of our
Variational Transfer Algorithms, presented in our paper "Transfer of Value Functions via Variational Methods"
(NIPS 2018), together with the instruction on how to reproduce our results.

## Abstract

We consider the problem of transferring value functions in reinforcement learning. We propose an approach that uses the given source tasks to learn a prior distribution over optimal value functions and provide an efficient variational approximation of the corresponding posterior in a new target task. We show our approach to be general, in the sense that it can be combined with complex parametric function approximators and distribution models, while providing two practical algorithms based on Gaussians and Gaussian mixtures. We theoretically analyze them by deriving a finite-sample analysis and provide a comprehensive empirical evaluation in four different domains.

## Requirements

```
python 3.5
numpy 1.14.2
PyTorch 0.4.1
joblib
matplotlib
OpenAI gym

```

## Repository Structure

* __algorithms:__ this folder contains the implementation of the algorithms proposed. It also includes an implementation of DDQN [2] in PyTorch.
* __approximators__: it includes different function approximators implementation such as linear regressor and Feedforward Neural Network.
* __envs__: it includes the implementation of the evaluation environments used (rooms, cartpole, mountain car and maze).
* __experiments__: here, organized by folder, are the main scripts to run the experiments presented in the empirical evaluation of our paper.
* __features__: it includes implementations of some features functions.
* __misc__: it contains implementation of varied auxiliary functions and data structures.
* __operators__: it contains implementations of different Bellman operators.


## How to reproduce our results

In the folder *experiments/*, further folders, corresponding to each experimental environment, can be found. To reproduce our results it enough to use ```python 3.5``` to run the scripts. Each of these uses, by default, the ```sources.pkl``` with the data from the source tasks used to transfer and produces a pkl file with the data to plot the performance.

* ```run_ft.py``` runs the fine-tuning.
* ```run_gvt.py``` runs the Gaussian Variational Transfer (GVT) experiment.
* ```run_mgvt.py``` runs the Mixture of Gaussian Variational Transfer (MGVT). By default, it uses 1 component for the posterior representation. To run for 3 components is only required to add the command line argument ```--post_components 3```.
* ```run_nt.py``` runs a non transfer algorithm. By default, it runs our algorithm based in the minimization of the Mellow Bellman Error. To run DDQN, it is enough to add the command-line argument ```--dqn 1```

Particularly, for the *Rooms* environment, we have further scripts that correspond to the additional experiments.

* ```run_*_likelihood``` runs the GVT or MGVT (depending of the script) using the sources with Gaussian distribution governing the door position. By default, it runs the 2-rooms environment.
* ```run_*_sequential``` runs the GVT or MGVT (depending of the script) by evaluating the performance with different number of source task given to the algorithm.
* To run the experiment with the distribution shift (tasks distribution of the sources restricted) is enough to run ```python3 run_*.py --source_file path/to/sources_gen```

## References

[2] Hado Van Hasselt, Arthur Guez, and David Silver. Deep reinforcement learning with double q-learning. 2016.
