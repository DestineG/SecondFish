# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F

def main():
    x = Variable(np.array([1, 2, 3, 4, 5, 6]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y = F.sum(x, axis=0)
    y.backward()
    print(y)
    print(x.grad)

    x = Variable(np.random.randn(2, 3, 4, 5))
    y = x.sum(keepdims=True)
    print(y.shape)
    

if __name__ == "__main__":
    main()