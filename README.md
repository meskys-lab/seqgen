# Model to generate stable proteins

"Algoritmo, generuojančio sintetines stabilių baltymų sekas, kūrimas"

## Description

This repository contains code for a model that generates stable proteins. This algorithm is designed to be 
easily trained on data obtain from high throughput testing. 



## Installation

All dependencies and libraries are in environment file. To run and train GPU is required. This algorithm should work 
with any modern Nvidia GPU (tested on RTX 2080). To create conda environment run: 

```
conda env create -f environment.yml
```

## Train model

TODO

Then run the command as shown below

```
python -m seqgen.train --train_csv train_split.p --val_csv val_split.p
```

## To generate sequences

To generate sequences run command below:

Note: predictions are run on GPU.

```
python -m seqgen.generate --model models/TODO --fasta example/to_predict.fasta

```
