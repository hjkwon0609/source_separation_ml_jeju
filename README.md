# Monaural Source Separation Project for ML Jeju camp 2017

## In Progress... will be updated with scripts for better setup

## Overview

Two methods of separating monaural sources have been implemented during the month spent at ML Jeju Camp. 

## Projects

First project implemented Vector Product Neural Networks, referencing the paper 'Music Signal Processing Using Vector Product Neural Networks' by Fan et al. Find paper [here](http://mac.citi.sinica.edu.tw/~yang/pub/fan17dlm.pdf)

Files related to this model are model.py, run.py, and config.py

Second project implemented Deep Clustering, referencing the paper 'Deep Clustering: Discriminiative embeddings for segmentation and separation' by Hershey et al. Find paper [here](https://arxiv.org/abs/1508.04306)

Files related to this model are embedding_model.py, embedding_run.py, and embedding_config.py

## Setup

Acquire iKala dataset -- need to sign a form and send a request to get access to the dataset.
More details could be found [here](http://mac.citi.sinica.edu.tw/ikala/).
After acquiring data, move the files from the iKala/Wavfile directory to the directory source_separation_ml_jeju/data/raw_data.

For preprocessing of data to TFRecords and split train/test dataset,

```
cd data_scripts
python preprocesing.py

```

(Need to create directories data/test and data/train)

## Run
To train the VPNN model,
```
python run.py
```

after training, to evaluate the model, 
```
python run.py --train=False
```

To train the deep clustering model,
```
python embedding_run.py
```

To evaluate the network, find the checkpoints inside the tensorboard events directory (source_separation_ml_jeju/tensorboard/[most_recent_dir]) and move to directory (source_separation_ml_jeju/checkpoints/) and 
```
python embedding_run.py --train=False
```