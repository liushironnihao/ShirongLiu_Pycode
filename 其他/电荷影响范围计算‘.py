import numpy as np
import matplotlib.pyplot as plt

# 定义已知常量（你可以根据实际情况修改这些值）
k = 1.0        # 库仑常数 (例如 8.99e9 N·m²/C²，这里设为1用于相对值)
q = 1.0        # 电荷量
z = 1.0        # z坐标
h = 0.5        # 平面距离参数

# 定义函数 F(x)
def F(x, k=k, q=q, z=z, h=h):
    term1 = z / (x**2 + z**2)**1.5
    term2 = (z + 2*h) / (x**2 + (z + 2*h)**2)**1.5
    return k * q**2 * (term1 - term2)

# 生成x值范围（对称范围，因为公式对x偶对称）
x = np.linspace(0, 10, 1000)

# 计算对应的F值
F_values = F(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制F(x)曲线
plt.plot(x, F_values, 'b-', linewidth=2, label=f'F(x)\n(z={z}, h={h}, q={q})')

# 添加坐标轴和标签
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.xlabel('x', fontsize=12)
plt.ylabel('F(x)', fontsize=12)
plt.title('F vs x: $F = kq^2\\left[\\frac{z}{(x^2+z^2)^{3/2}} - \\frac{z+2h}{(x^2+(z+2h)^2)^{3/2}}\\right]$', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# 显示最大值和最小值点
max_idx = np.argmax(F_values)
min_idx = np.argmin(F_values)

plt.plot(x[max_idx], F_values[max_idx], 'ro', markersize=8, label=f'Max: ({x[max_idx]:.2f}, {F_values[max_idx]:.4f})')
plt.plot(x[min_idx], F_values[min_idx], 'go', markersize=8, label=f'Min: ({x[min_idx]:.2f}, {F_values[min_idx]:.4f})')

plt.legend()
plt.tight_layout()
plt.show()

# 打印一些关键点的值
print("关键点数值:")
print(f"F(0) = {F(0):.6f}")
print(f"F(±1) = {F(1):.6f}")
print(f"F(±2) = {F(2):.6f}")
print(f"F(±5) = {F(5):.6f}")