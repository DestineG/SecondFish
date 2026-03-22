# src/tests/function.py

import unittest
import numpy as np
import time

from ..function import square, add, mul
from ..variable import Variable
from ..config import no_grad

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)

class AddTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        expected = np.array(5.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = add(x0, x1)
        y.backward()
        expected_x0 = np.array(1.0)
        expected_x1 = np.array(1.0)
        self.assertEqual(x0.grad, expected_x0)
        self.assertEqual(x1.grad, expected_x1)

class MulTest(unittest.TestCase):
    def test_forward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = mul(x0, x1)
        expected = np.array(6.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(3.0))
        y = mul(x0, x1)
        y.backward()
        expected_x0 = np.array(3.0)
        expected_x1 = np.array(2.0)
        self.assertEqual(x0.grad, expected_x0)
        self.assertEqual(x1.grad, expected_x1)

class NoGradTest(unittest.TestCase):
    def test_no_grad_context(self):
        with no_grad():
            x = Variable(np.array(2.0))
            y = square(x)
            self.assertIsNone(y.creator)
            self.assertEqual(y.data, np.array(4.0))

        x = Variable(np.array(2.0))
        y = square(x)
        self.assertIsNotNone(y.creator)
        print("no_grad context check passed: Link properly cut.")

    def test_performance_with_no_grad(self):
        steps = 10000
        x = Variable(np.random.rand(100))
        
        # 记录梯度的耗时
        start = time.time()
        curr = x
        for _ in range(steps):
            curr = add(curr, curr)
        time_with_grad = time.time() - start

        # 不记录梯度的耗时
        start = time.time()
        with no_grad():
            curr = x
            for _ in range(steps):
                curr = add(curr, curr)
        time_without_grad = time.time() - start

        print(f"Time with grad: {time_with_grad:.5f}s")
        print(f"Time without grad: {time_without_grad:.5f}s")
        
        self.assertLess(time_without_grad, time_with_grad)

class TestVarOperators(unittest.TestCase):
    def test_var_interaction(self):
        x = Variable(np.array(2.0))
        y = Variable(np.array(3.0))

        # 使用 * 和 + 构建组合运算
        # f = x * y + x * x
        # 预期值: 2*3 + 2*2 = 10
        f = x * y + x * x
        
        self.assertEqual(f.data, 10.0)
        print(f"Forward Result: {f.data} (Expected: 10.0)")

        # 验证反向传播梯度
        # df/dx = y + 2x = 3 + 2*2 = 7
        # df/dy = x = 2
        f.backward()
        
        self.assertEqual(x.grad, 7.0)
        self.assertEqual(y.grad, 2.0)
        print(f"x.grad: {x.grad} (Expected: 7.0)")
        print(f"y.grad: {y.grad} (Expected: 2.0)")

if __name__ == "__main__":
    unittest.main()