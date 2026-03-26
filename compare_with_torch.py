import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def functional_normalize(x, eps=1e-8):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / (std + eps)

class CustomLinear(nn.Module):
    def __init__(self, input_dim, output_dim, normalize=False, activation=None):
        super().__init__()
        # 初始化：random.randn * 0.01
        self.W = nn.Parameter(torch.randn(input_dim, output_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(output_dim))
        self.normalize = normalize
        self.activation = activation

    def forward(self, x):
        y = torch.matmul(x, self.W) + self.b
        if self.normalize:
            y = functional_normalize(y)
        if self.activation:
            y = self.activation(y)
        return y

class Regression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomLinear(input_dim, hidden_dim, normalize=False, activation=None),
            CustomLinear(hidden_dim, hidden_dim, normalize=True, activation=torch.tanh),
            CustomLinear(hidden_dim, hidden_dim, normalize=False, activation=torch.tanh),
            CustomLinear(hidden_dim, hidden_dim, normalize=True, activation=torch.tanh),
            CustomLinear(hidden_dim, hidden_dim, normalize=False, activation=torch.tanh)
        ])
        self.fc_out = CustomLinear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.fc_out(x)

def target_func(x):
    return np.sin(2 * np.pi * x)

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    model = Regression(1, 1)
    lr = 0.01
    iters = 100000

    # 生成数据并转为 Torch Tensor
    factor = 6
    extend = 1
    x_train_np = np.random.rand(200, 1) * factor
    y_train_np = target_func(x_train_np) + np.random.randn(200, 1) * 0.01
    
    x_train = torch.from_numpy(x_train_np).float()
    y_train = torch.from_numpy(y_train_np).float()

    # 使用原生 SGD 优化算法 (算法不变)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # 对应 (diff**2).sum() / len

    print(f"Starting training for {iters} iterations...")

    for i in range(iters):
        # 前向传播
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        # 反向传播 (等同于 model.clear_grad() + loss.backward())
        optimizer.zero_grad()
        loss.backward()

        # 参数更新 (等同于 model.optimize(lr))
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iteration {i:5d} | Loss: {loss.item():.6f}")

    # 结果可视化
    print("\nTraining complete. Plotting results...")
    
    model.eval()
    with torch.no_grad():
        x_test_np = np.linspace(0, factor + extend, 300).reshape(300, 1)
        x_test = torch.from_numpy(x_test_np).float()
        y_test_pred = model(x_test).numpy()
        y_test_true = target_func(x_test_np)

    plt.figure(figsize=(10, 6))
    plt.plot(x_test_np, y_test_true, 'g--', label="Target Function (sin)")
    plt.plot(x_test_np, y_test_pred, 'r-', linewidth=2, label="PyTorch Regression Result")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function Fitting Migrated to PyTorch")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()