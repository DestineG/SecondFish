# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F

def main():
    x = Variable(np.array([[1,2,3], [1,2,3]]), name="x")
    y = F.transpose(x)
    y1 = x.T
    y2 = x.transpose()
    
    x.clear_grad()
    y.backward()

    print(y)
    print(x.get_grad())

    x.clear_grad()
    y1.backward()

    print(y)
    print(x.get_grad())

    x.clear_grad()
    y2.backward()

    print(y)
    print(x.get_grad())
    

if __name__ == "__main__":
    main()