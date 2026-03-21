# variable.py

import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data

    @classmethod
    def test(cls):
        data = np.array(1.0)
        x = cls(data)
        print(x.data)

        x.data = np.array(2.0)
        print(x.data)

if __name__ == "__main__":
    Variable.test()