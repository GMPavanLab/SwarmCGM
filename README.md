# SwarmCGᴍ

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [Reproduction instructions](#reproduction-instructions)
- [License](./LICENSE)
- [Issues](https://github.com/GMPavanLab/SwarmCGM/issues)

# Overview

Automatic data-driven approaches are increasingly used to develop accurate molecular models. But the parameters of such automatically-optimised models are typically untransferable. Using a multi-reference approach in combination with an automatic optimisation engine (SwarmCGᴍ), here we show that it is possible to optimise coarse-grained (CG) lipid models that are also transferable, generating optimised lipid force fields. The parameters of the CG lipid models are iteratively and simultaneously optimised against higher-resolution simulations (bottom-up) and experimental data (top-down references). Including different types of lipid bilayers in the training set guarantees the transferability of the optimised force field parameters. Tested against state-of-the-art CG lipid force fields, we demonstrate that SwarmCGᴍ can systematically improve their parameters, enhancing the agreement with the experiments even for lipid types not included in the training set. The approach is general and can be used to improve existing CG lipid force fields, as well as to develop new custom ones.

# System Requirements

The code provided herein is designed for running on High Performance Computing (HPC) resources, but can just as well be executed on a classic desktop computer for testing purposes, or even for obtaining optimized CG lipid models given enough execution time.
Due to the embarrassingly parallel nature of the tasks performed by SwarmCGᴍ, best performances are achieved on HPC resources, in terms of required wall-clock time for obtaining well-optimized CG models.

The execution wall-clock times are directly dependent on:
- the number of different lipids included in the training set
- the number of temperatures at which each of these lipids will be simulated
- the number of parameters to be optimized in the FF, according to user requirements
- the simulation times used in MD equilibration and production runs used for FF evaluation
- the number of MD simulations that can run in parallel on the computational infrastructure in use
- the wall-clock time required to perform each MD simulation using a given system

Tested operating systems:
- Ubuntu 18.04  
- SUSE Linux Enterprise Server 15 SP1

## Local usage

For testing purposes, the following specs will be sufficient:

- RAM: 4+ GB  
- CPU: 4+ cores, 2.0+ GHz/core
- GPU: optional

For decent performance, it is recommended to use:

- RAM: 8+ GB
- CPU: 36+ cores, 3.6+ Ghz/core
- GPU: optional

## HPC usage

For optimal performance, and for obtaining accurate FFs via the usage of many different lipids in the training set, SwarmCGᴍ can be executed on HPC resources.

Supported HPC resource managers:
- SLURM

# Installation Guide

Standalone files can be directly copied as [provided here](https://github.com/GMPavanLab/SwarmCGM/src) and are ready for usage.

Users should install the following python packages prior to executing the code:

```
pip install numpy scipy matplotlib MDAnalysis pyemd fst-pso
```

which will install in about 1-3 minutes on a machine with the minimal specs.

If you are having troubles installing or using this software, please [drop us an Issue](https://github.com/GMPavanLab/SwarmCGM/issues). 

Tested package versions:
- numpy 1.19.2
- scipy 1.5.2
- matplotlib 3.3.2
- MDAnalysis 1.0.0
- pyemd 0.5.1
- fst-pso 1.7.15

# Demo

For a quick demonstration of the software, one can execute the following command:

```
# cd src
python3 optimize_lipids.py -cfg minimalistic_demo.yaml
```

This minimalistic example should execute within 12-24 hours on the "testing purpose" hardware described, and produce suboptimal CG models of the 2 PC lipids used in the testing set due to:
- the short MD simulation times employed
- the low number of particles used in the swarm

Which are necessary limits to be introduced here for quick demonstration purposes, and are also well-suited for testing the software with your own input of parameters for a specific optimization run.

As the input of SwarmCGᴍ requires a preliminary CG mapping choice, several parameters allow to define precisely which parameters should be optimized, or which other ones stay at given fixed values, according to your requirements.
To this end, SwarmCGᴍ makes use of [YAML](https://yaml.org/) config files to simplify this process and help keeping track of the hyper-parameters used for an optimization run. 

Please refer to the documentation for a breakdown of each parameter and step-by-step guidance on how to parametrize an optimization procedure. Additional example config files are provided in section [Reproduction instructions](#reproduction-instructions) and can also be related to the content and explanations provided in the paper.

# Results

All optimized CG models and their associated FFs can be found in directory: [results](https://github.com/GMPavanLab/SwarmCGM/results)

# Reproduction instructions

The config files cited in the 3 next subsections are available in directory: [reproducibility](https://github.com/GMPavanLab/SwarmCGM/reproducibility)

## Example 1: Optimisation of Martini-based CG models of PC lipids in explicit solvent

```
# cd src
python3 optimize_lipids.py -cfg example1.yaml
```

## Example 2: Optimisation of Martini-based CG models of lipids in implicit solvent

```
# cd src
python3 optimize_lipids.py -cfg example2.yaml
```

## Example 3: Conception of custom low-resolution CG lipid models in implicit solvent

```
# cd src
python3 optimize_lipids.py -cfg example3.yaml
```
