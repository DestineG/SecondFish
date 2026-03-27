# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph, layer
import src.functions as F


def main():
    fc = layer.Linear(10, 5)
    x = Variable(np.random.rand(1, 10))
    y = fc(x)
    print(y)

if __name__ == "__main__":
    main()