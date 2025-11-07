from turtle import forward
import torch
from torch.func import grad
import torch.nn as nn


print("=" * 60)
print("PyTorch 自动微分机制实验")
print("=" * 60)

# ========== 实验 1: 简单梯度计算 ==========
print("\n【实验 1】简单梯度计算")
print("-" * 40)

x = torch.tensor([2.0, 3.0], requires_grad=True)
print(f"输入 x: {x}")

y = x ** 2
print(f"y = x^2: {y}")

loss = y.sum()
print(f"loss = sum(y): {loss.item()}")

# 反向传播
loss.backward()
print(f"梯度 dx/dloss: {x.grad}")
print(f"理论值: [2*2, 2*3] = [4.0, 6.0]")

# ========== 实验 2: 多层神经网络 ==========
print("\n【实验 2】两层神经网络训练")
print("-" * 40)

class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10,20)
        self.fc2 = nn.Linear(20,1)

    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")

model = TinyNet().to(device)
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

torch.manual_seed(42)
x = torch.randn(5,10, device=device)
y_true = torch.randn(5,1,device=device)


# 训练一步
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
y_pred = model(x)
loss = ((y_pred - y_true)**2).mean()

print(f"\n训练前损失: {loss.item():.6f}")

loss.backward()
optimizer.step()

# 再次前向传播查看损失
with torch.no_grad():
    y_pred_new = model(x)
    loss_new = ((y_pred_new - y_true) ** 2).mean()

print(f"训练后损失: {loss_new.item():.6f}")
print(f"损失下降: {loss.item() - loss_new.item():.6f}")

# ========== 实验 3: 梯度流可视化 ==========
print("\n【实验 3】梯度流分析")
print("-" * 40)

# 重新初始化模型以查看梯度
model = TinyNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

y_pred = model(x)
loss = ((y_pred - y_true) ** 2).mean()
loss.backward()


print("各层梯度范数:")
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"  {name:15s}: {grad_norm:.6f}")

# ========== 实验 4: 梯度累积演示 ==========
print("\n【实验 4】梯度累积 vs 大批量")
print("-" * 40)


# 方法 1: 大批量
model1 = TinyNet().to(device)
opt1 = torch.optim.SGD(model1.parameters(), lr=0.01)
x_large = torch.randn(8, 10, device=device)
y_large = torch.randn(8, 1, device=device)

opt1.zero_grad()
pred1 = model1(x_large)
loss1 = ((pred1 - y_large) ** 2).mean()
loss1.backward()

# 保存梯度
grad1 = model1.fc1.weight.grad.clone()

# 方法 2: 梯度累积（2次 batch_size=4）
model2 = TinyNet().to(device)
model2.load_state_dict(model1.state_dict())  # 使用相同初始化
opt2 = torch.optim.SGD(model2.parameters(), lr=0.01)

opt2.zero_grad()
for i in range(2):
    x_micro = x_large[i*4:(i+1)*4]
    y_micro = y_large[i*4:(i+1)*4]
    pred2 = model2(x_micro)
    loss2 = ((pred2 - y_micro) ** 2).mean() / 2  # 除以累积次数
    loss2.backward()

grad2 = model2.fc1.weight.grad

# 对比梯度
grad_diff = (grad1 - grad2).abs().max().item()
print(f"大批量梯度范数: {grad1.norm().item():.6f}")
print(f"梯度累积梯度范数: {grad2.norm().item():.6f}")
print(f"最大差异: {grad_diff:.8f}")
print(f"结论: {'梯度几乎相同！' if grad_diff < 1e-5 else '有微小数值误差'}")

print("\n" + "=" * 60)
print("✅ 所有实验完成！")
print("=" * 60)

print("\n📚 关键要点:")
print("1. requires_grad=True 使张量可以计算梯度")
print("2. .backward() 自动计算所有梯度")
print("3. 梯度累积可以模拟大批量训练")
print("4. MPS 对神经网络训练同样有效")
