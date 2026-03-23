# main.py

import numpy as np
from src import Variable

def main():
    # 单变量高阶导数测试
    x = Variable(np.array(2.0), name="x")
    y = x ** 4 - 2 * (x ** 2)

    # 第一次反向传播：求一阶导数 dy/dx = 4x^3 - 4x
    # 设置 create_graph=True 才能保留梯度的计算图
    y.backward(create_graph=True)
    gx = x.grad
    print(f"一阶导数 (x=2): {gx.data}") # 4*8 - 4*2 = 24

    # 求二阶导数：先清除 x 的一阶梯度
    x.clear_grad()

    # 第二次反向传播：对 gx 求导 d(gx)/dx = 12x^2 - 4
    gx.backward()
    print(f"二阶导数 (x=2): {x.grad.data}") # 12*4 - 4 = 44

if __name__ == "__main__":
    main()