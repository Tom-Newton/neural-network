# neural-network
Making my own Python neural network module to see how hard it could be.

Training data is not included in the repo. Training datasets I used:
https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
https://www.cs.toronto.edu/~kriz/cifar.html

This Python module uses cupy essentailly a GPU accelerated version of numpy. It therefore reuires: 
A CUDA compatible GPU
CUDA drivers
cupy python library with version matching CUDA drivers (I used 10.2)
