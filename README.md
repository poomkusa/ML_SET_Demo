# ML_SET_Demo

## Overview

Algorithms for stock's next n days up/down movement classification.

## Requirements

- Python3
- Dependecies are in requirements.yml

## Usage

You may train using one of the available algorithms (note that hyperparameters have not been optimized and must be adjusted by yourself):

150 input features (open, high, low, close, volume for the last 30 days)

Neural Network: CNN with LeakyReLU in hidden layers and softmax in output layers, regularized by dropout and batch normalization, Adam as an optimizer.

SVM: current setting is C = 100, the rest are default from sklearn.

Random Forest: current setting is number of trees = 25, the rest are default from sklearn.
