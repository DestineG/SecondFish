# main.py

import numpy as np

from src import Variable

def main():
    x = Variable(np.array(1.0))
    y = Variable(np.array(2.0))
    z = x + y
    print(z)  # 输出 3.0
    z.backward()
    print(x.grad)  # 输出 1.0
    print(y.grad)  # 输出 1.0


if __name__ == "__main__":
    main()
