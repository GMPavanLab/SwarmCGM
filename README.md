# SwarmCGᴍ

## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results & reproduction instructions](#results-&-reproduction-instructions)
- [License](./LICENSE)
- [Issues](https://github.com/GMPavanLab/SwarmCGM/issues)
- [Citation](#citation)

# Overview

Automatic data-driven approaches are increasingly used to develop accurate molecular models. But the parameters of such automatically-optimised models are typically untransferable. Using a multi-reference approach in combination with an automatic optimisation engine (SwarmCGᴍ), here we show that it is possible to optimise coarse-grained (CG) lipid models that are also transferable, generating optimised lipid force fields.

The parameters of the CG lipid models are iteratively and simultaneously optimised against higher-resolution simulations (bottom-up) and experimental data (top-down references). Including different types of lipid bilayers in the training set guarantees the transferability of the optimised force field parameters. Tested against state-of-the-art CG lipid force fields, we demonstrate that SwarmCGᴍ can systematically improve their parameters, enhancing the agreement with the experiments even for lipid types not included in the training set. The approach is general and can be used to improve existing CG lipid force fields, as well as to develop new custom ones.

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
- CPU: 8+ cores, 2.0+ GHz/core
- GPU: optional

For decent performance, it is recommended to use:

- RAM: 8+ GB
- CPU: 36+ cores, 3.6+ Ghz/core
- GPU: optional

## HPC usage

For optimal performance, SwarmCGᴍ can be executed on HPC resources.
This enables calibrating accurate CG lipid FFs via the usage of many different lipids in the training set, which allows to fully take advantage of the transferability constraint imposed to the FF (*i.e.* the FF parameters are tested at each iteration via multiple CG simulations of lipid bilayers). 

Supported HPC resource managers:
- SLURM

# Installation Guide

Standalone files can be directly copied as [provided here](https://github.com/GMPavanLab/SwarmCGM/src) and are ready for usage.

Users must install the following python packages prior to executing the code:

```
pip install numpy scipy matplotlib MDAnalysis pyemd fst-pso
```

which will install in about 2-5 minutes on a machine with the "testing purposes" specifications.

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

This minimalistic example should execute within 12-24 hours on the "testing purpose" hardware described. It will be possible to see that the FF score decreases during optimization, however this will not result in realistically optimal lipid models, due to:
- the small size of the training set (limited transferability constraint: only 2 different lipids)
- the short MD simulation times employed (limited sampling: only 3 ns of equilibration and 50 ns of production)
- the low number of particles used in the swarm (limited exploration of the FF parameters: only 3 particles)

which are necessary limits to be introduced here for quick demonstration purposes. These values can be even further decreased for the purpose of testing the consistency of your own input configuration files for a specific optimization run.

As the input of SwarmCGᴍ requires a preliminary CG mapping choice (*i.e.* defining the positions and types of the CG beads, bonds and angles used for building the molecular models which parameters will be optimized), several parameters allow to define precisely which parameters should be optimized, or which other ones stay at given fixed values, according to your requirements.
To this end, SwarmCGᴍ makes use of [YAML](https://yaml.org/) config files to simplify this process and help keeping track of the hyper-parameters used for an optimization run. 

Please refer to [the manual](https://github.com/GMPavanLab/SwarmCGM/Manual.pdf) for a breakdown of each parameter and step-by-step guidance on how to parametrize an optimization procedure.
Additional example config files are provided in section [Reproduction instructions](#reproduction-instructions) and can also be related to the content and explanations provided in the paper.

# Results & reproduction instructions

All the optimized CG models produced in the paper, and their associated FFs, can be found in folder [results](https://github.com/GMPavanLab/SwarmCGM/results).
The config files cited in the 3 next subsections are available in the 3 corresponding subdirectories in [results](https://github.com/GMPavanLab/SwarmCGM/results).

### Example 1: Optimisation of Martini-based CG models of PC lipids in explicit solvent

For this demonstration, we use reference data from 7 different phosphatidylcholine (PC) lipids and create optimized CG models of high resolution, in the framework of Martini and in explicit solvent.

```
# cd src
python3 optimize_lipids.py -cfg example1.yaml
```

### Example 2: Optimisation of Martini-based CG models of lipids in implicit solvent

For this demonstration, we use reference data from 4 different phosphatidylcholine (PC) lipids and create optimized CG models of high resolution, in the framework of Martini and in implicit solvent.

```
# cd src
python3 optimize_lipids.py -cfg example2.yaml
```

### Example 3: Conception of custom low-resolution CG lipid models in implicit solvent

For this demonstration, we use reference data from 4 different phosphatidylcholine (PC) lipids and create optimized CG models of low resolution, ab-initio and in implicit solvent.

```
# cd src
python3 optimize_lipids.py -cfg example3.yaml
```

# Citation

The preprint for SwarmCGᴍ can be found [on ArXiv](arxiv.org/abs/2107.01012).
