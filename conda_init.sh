#!/bin/bash

conda install --file requirements.txt
conda install jsonlines
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge transformers
conda install -c conda-forge datasets 
conda install -c conda-forge clean-text 
conda install -c conda-forge elasticsearch
conda install -c conda-forge faiss-gpu
conda install -c conda-forge sentence-transformers
