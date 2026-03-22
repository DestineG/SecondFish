# src/tests/function.py

import unittest
import numpy as np
import time

from ..function import square, add
from ..variable import Variable

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

class BackwardTest(unittest.TestCase):
    def test_backward_logic_and_speed(self):
        x_val = np.array([1.0, 2.0])
        x = Variable(x_val)
        y = add(x, x)
        y.backward()
        
        expected_grad = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(x.grad, expected_grad)
        print("Basic logic check passed: dy/dx = 2")


        # 菱形结构
        x.clear_grad()
        a = add(x, x)
        b = add(x, x)
        y = add(a, b)
        y.backward()
        np.testing.assert_array_almost_equal(x.grad, np.array([4.0, 4.0]))
        print("Diamond graph check passed: dy/dx = 4")

        # 性能与深度验证
        steps = 1000 
        x_large = Variable(np.random.rand(10))
        curr = x_large
        
        start_time = time.time()
        for _ in range(steps):
            curr = add(curr, curr)
        curr.backward()
        end_time = time.time()
        
        expected_large_grad = np.power(2.0, steps)
        for g in x_large.grad:
            self.assertAlmostEqual(g, expected_large_grad)            
        print(f"Deep graph check passed (steps={steps})")
        print(f"Backward speed for {steps} layers: {end_time - start_time:.5f}s")

if __name__ == "__main__":
    unittest.main()
