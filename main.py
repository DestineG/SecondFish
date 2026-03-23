# main.py

import os
import math
import numpy as np

from src import Variable, get_dot_graph, plot_dot_graph

def mySin(x, threshold=0.0005):
    '''sin(x) 计算
    sin = Σ ((-1)^i / (2i+1)!) * x^(2i+1)
    '''
    y = 0
    for i in range(100000):
        c = (-1)**i / math.factorial(2*i+1)
        t = c * x**(2*i+1)
        y += t
        if abs(t.data) < threshold:
            break
    return y

def main():
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    x0 = Variable(np.array(np.pi/4), name="x0")
    y0 = mySin(x0)
    y0.backward()
    y0.name = "y0"
    print(f"y0={y0.data}, x0.grad={x0.grad}")
    plot_dot_graph(y0, verbose=True, to_file=os.path.join(save_dir, "taylor_sin.png"))


if __name__ == "__main__":
    main()
