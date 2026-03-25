# main.py

import os
import numpy as np
import matplotlib.pyplot as plt
from src import Variable, plot_dot_graph
import src.functions as F

def main():
    x = Variable(np.array(np.pi / 4), name="x")
    y = F.tanh(x)
    y.name = "y"
    y.backward(create_graph=True)

    iters = 4
    for i in range(iters):
        gx = x.grad
        x.clear_grad()
        gx.backward(create_graph=True)

    # 绘制计算图
    gx = x.grad
    gx.name = "gx" + str(iters + 1)
    plot_dot_graph(gx, verbose=True, to_file=f"./figures/tanh_{gx.name}.png")

if __name__ == "__main__":
    main()