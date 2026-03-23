# main.py

import os
import numpy as np

from src import Variable, get_dot_graph, plot_dot_graph

def sphere(x, y):
    z = x ** 2 + y ** 2
    return z


def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z


def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def main():
    save_dir = "figures"
    os.makedirs(save_dir, exist_ok=True)
    x = Variable(np.array(1.0), name="x")
    y = Variable(np.array(1.0), name="y")

    z = sphere(x, y)
    z.name = "z"
    x.clear_grad()
    y.clear_grad()
    z.backward()
    print(f"Sphere Function: z={z.data}, x.grad={x.grad}, y.grad={y.grad}")
    plot_dot_graph(z, verbose=True, to_file=os.path.join(save_dir, "sphere_function.png"))

    z = matyas(x, y)
    z.name = "z"
    x.clear_grad()
    y.clear_grad()
    z.backward()
    print(f"Matyas Function: z={z.data}, x.grad={x.grad}, y.grad={y.grad}")
    plot_dot_graph(z, verbose=True, to_file=os.path.join(save_dir, "matyas_function.png"))

    z = goldstein(x, y)
    z.name = "z"
    x.clear_grad()
    y.clear_grad()
    z.backward()
    print(f"Goldstein-Price Function: z={z.data}, x.grad={x.grad}, y.grad={y.grad}")
    plot_dot_graph(z, verbose=True, to_file=os.path.join(save_dir, "goldstein_price_function.png"))


if __name__ == "__main__":
    main()
