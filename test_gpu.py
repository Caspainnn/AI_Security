# gpu_test.py —— 测试 PyTorch 是否能用 GPU
import torch

print("=" * 50)
print("PyTorch GPU 测试")
print("=" * 50)

# 1. 检查 CUDA 是否可用
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前 GPU: {torch.cuda.current_device()}")
    print(f"GPU 名称: {torch.cuda.get_device_name(0)}")

    # 2. 简单 GPU 运算测试
    print("\n正在测试 GPU 运算...")
    x = torch.randn(2048, 2048).cuda()
    y = torch.matmul(x, x)
    print("GPU 矩阵运算完成 ✅")

else:
    print("❌ 没有可用 GPU，将使用 CPU")

print("\n测试结束")