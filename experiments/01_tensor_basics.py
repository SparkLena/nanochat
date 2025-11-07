from sympy.core.evalf import as_mpmath
import torch
import time

print("=" * 60)
print("Pytorch 张量操作和MPS性能测试")
print("=" * 60)

# 1. 张量创建
print("\n1. 创建张量")
x = torch.randn(3,4)
print(f"Shape: {x.shape}, Device: {x.device}, Dtype: {x.dtype}")

# 2. 设备迁移测试
print("\n2. 设备检测和迁移")
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"使用设备: {device}")
x = x.to(device)
print(f"张量已迁移到: {x.device}")

# 3. 基础运算
print("\n3. 矩阵乘法测试")
y = torch.randn(4,5, device=device)

z = x @ y  # 矩阵乘法
print(f"输入 x: {x.shape}, y: {y.shape}")
print(f"输出 z: {z.shape}")


# 4. 性能基准测试
print("\n4. MPS vs CPU 性能对比")
print("-" * 60)
print(f"{'矩阵大小':<15} {'MPS时间(ms)':<15} {'CPU时间(ms)':<15} {'加速比':<10}")
print("-" * 60)

sizes = [64, 128, 256, 512, 1024]

for size in sizes:
    if device == "mps":
        a_mps = torch.randn(size, size, device="mps")
        b_mps = torch.randn(size, size, device="mps")

        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(100):
            c_mps = a_mps @ b_mps
        torch.mps.synchronize()
        t1 = time.time()
        time_mps = (t1 - t0) * 10
    else:
        time_mps = 0

    # CPU 测试
    a_cpu = torch.randn(size, size, device="cpu")
    b_cpu = torch.randn(size, size, device="cpu")
    
    t0 = time.time()
    for _ in range(100):
        c_cpu = a_cpu @ b_cpu
    t1 = time.time()
    time_cpu = (t1 - t0) * 10  # ms per matmul

    # 输出结果
    speedup = time_cpu / time_mps if time_mps > 0 else 1.0
    if device == "mps":
        print(f"{size}x{size:<10} {time_mps:<15.2f} {time_cpu:<15.2f} {speedup:<10.2f}x")
    else:
        print(f"{size}x{size:<10} {'N/A':<15} {time_cpu:<15.2f} {'N/A':<10}")

print("-" * 60)
print("\n✅ 测试完成！")

if device == "mps":
    print("\n🎉 您的 M 系列芯片 MPS 加速已启用！")
    print("这将显著提升训练和推理速度。")
else:
    print("\n⚠️  MPS 不可用，使用 CPU 模式。")
    print("训练速度会较慢，但足够学习使用。")        
