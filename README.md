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

## To prepare data for training

In order to prepare data for training provide csv file that contains sequence id and sequence itself and then run

```
python -m seqgen.convert --input_file examples/input.csv --output_path examples/data.p
```


## Train model

Then run the command as shown below

```
python -m seqgen.train --train_data examples/data.p --val_data examples/data.p
```

## To generate sequences

To generate sequences run command below:

Note: generation is run on GPU.

```
python -m seqgen.generate --model models/model.pt --output results/sample.fasta
```
