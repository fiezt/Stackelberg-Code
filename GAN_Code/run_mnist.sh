#!/bin/sh
python MNIST_Simulation.py mnist1 stack 1
python MNIST_Simulation.py mnist2 stack 0
python MNIST_Simulation.py mnist3 simgrad 1
python MNIST_Simulation.py mnist4 simgrad 0