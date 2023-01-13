#!/bin/bash
mkdir -p data plots
cd data
curl https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xzvf -