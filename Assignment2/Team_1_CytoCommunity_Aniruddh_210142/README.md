# Implementation of the CytoCommunity Method

Tested by Aniruddh Pramod (210142)

## Introduction

This directory contains code for implementing and inferencing with the CytoCommunity method on the MERFISH and OSMFISH. It is built on top of the original [CytoCommunity repository](https://github.com/tanlabcode/CytoCommunity). 

## Description of Directory

This directory contains the following key files and subdirectories:

- `CytoCommunity/`: A clone of the original CytoCommunity repository.
- `CytoCommunity/environment.yml`: Conda environment file for setting up the CytoCommunity environment.
- `environment.yml`: Conda environment file for setting up the preprocessing environment.
- `Dataset1_MERFISH.h5ad`: MERFISH dataset provided for this task.
- `Dataset2_OSMFISH.h5ad`: OSMFISH dataset provided for this task.
- `preprocessing.ipynb`: Jupyter notebook to process the dataset into the format required by CytoCommunity
- `Step1_ConstructCellularSpatialGraphs.py`: Script to construct cellular spatial graphs.
- `Step2_TCNLearning_Unsupervised.py`: Script for unsupervised learning and TCN Assignment.
- `Step3_TCNEnsemble.R`: R script to perform TCN Assignment using an Ensemble Learning Framework.
- `Step4_ResultVisualization.py`: Script to visualize the results of the TCN models.
- `postprocessing.py`: Batch Script to process the output of Step 3 into a CSV submission file. 
- `tuner.bat`: Batch script for performing hyperparameter tuning.
- `best_res.bat`: Batch script to run all the steps with the optimal hyperparameters to reproduce the final results
- `CytoCommunity_{Dataset}.csv` - Final submission files for the task.

## Environment Setup 

Two environments are required, one is the CytoCommunity environment from the `.\CytoCommunity\environment.yml` file. The other is the preprocessing environment from the `.\environment.yml` file.
```sh
    conda env create -f ./CytoCommunity/environment.yml
    conda activate CytoCommunity
```
```sh
    conda env create -f ./environment.yml
    conda activate preprocessing
```

## Usage Instructions

Run the `preprocessing.ipynb` to completion and then run the `best_res.bat` script. The final output is stored in `CytoCommunity_{Dataset}.csv` and the graphs are stored in `./Step4_Output_typistmerfish/` and `./Step4_Output_osmfish/`. 
