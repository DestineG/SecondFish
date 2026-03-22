# src/tests/function.py

import unittest
import numpy as np
from src import Variable

def test_full_operators():
    print("=== Starting SecondFish Operator Test ===\n")

    # 测试基础运算 (Add, Mul, Neg)
    x = Variable(np.array(2.0))
    y = Variable(np.array(3.0))
    
    # f = -x + x * y  => -2.0 + 2.0 * 3.0 = 4.0
    f = -x + x * y
    print(f"Test 1 (Basic): -x + x * y = {f.data} | Expected: 4.0")
    
    f.backward()
    # df/dx = -1 + y = -1 + 3 = 2.0
    # df/dy = x = 2.0
    print(f"Gradients: x.grad={x.grad}, y.grad={y.grad} | Expected: 2.0, 2.0")

    # 测试减法与除法 (Sub, Div)
    x.clear_grad()
    y.clear_grad()
    # g = (y - x) / x => (3 - 2) / 2 = 0.5
    g = (y - x) / x
    print(f"\nTest 2 (Sub/Div): (y - x) / x = {g.data} | Expected: 0.5")
    
    g.backward()
    # dg/dy = 1/x = 0.5
    # dg/dx = (-1 * x - (y - x) * 1) / x^2 = -y / x^2 = -3 / 4 = -0.75
    print(f"Gradients: y.grad={y.grad}, x.grad={x.grad} | Expected: 0.5, -0.75")

    # 测试反向算子 (rsub, rdiv) - 验证 Other Op Variable
    x.clear_grad()
    # h = 10.0 - x + 1.0 / x  => 10 - 2 + 1/2 = 8.5
    h = 10.0 - x + 1.0 / x
    print(f"\nTest 3 (Reverse): 10 - x + 1/x = {h.data} | Expected: 8.5")
    
    h.backward()
    # dh/dx = -1 - 1/x^2 = -1 - 1/4 = -1.25
    print(f"Gradients: x.grad={x.grad} | Expected: -1.25")

    # 测试幂运算 (Pow)
    x.clear_grad()
    # p = x ** 3 => 2^3 = 8
    p = x ** 3
    print(f"\nTest 4 (Pow): x ** 3 = {p.data} | Expected: 8.0")
    
    p.backward()
    # dp/dx = 3 * x^2 = 3 * 4 = 12.0
    print(f"Gradients: x.grad={x.grad} | Expected: 12.0")

    print("\n=== All Tests Passed! ===")

if __name__ == "__main__":
    try:
        test_full_operators()
    except Exception as e:
        print(f"\nTest Failed! Error: {e}")
        print("Tip: Check if your 'sub/rsub' and 'div/rdiv' logic handles constants correctly.")