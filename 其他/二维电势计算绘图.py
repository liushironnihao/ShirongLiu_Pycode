import numpy as np
import matplotlib.pyplot as plt


# 物理常数
epsilon0 = 8.854187817e-12  # F/m
e=1.602e-19

# 参数
sigma = 5.03e-12  # C/m 
h = 1.5e-6      # m
E_ext = 1e5   # V/m
y_zero=1/(sigma/e)

# 电势函数
def V(y, sigma, h, E_ext):
    term1 = sigma / (2 * np.pi * epsilon0) * np.log(1 + 2*h / y)
    return term1 + E_ext * y

# 生成 y 值
y_vals = np.linspace(y_zero, 10e-6, 500)
V_vals = V(y_vals, sigma, h, E_ext)

# 寻找电势最低点 - 先求导数零点
# dV/dy = -sigma/(2πϵ₀) * (2h/(y(y+2h))) + E_ext
# 令导数为0: E_ext = sigma/(2πϵ₀) * (2h/(y(y+2h)))
# 解方程: y² + 2hy - A = 0, 其中 A = (sigma*h)/(πϵ₀*E_ext)

A = (sigma * h) / (np.pi * epsilon0 * E_ext)
# 解二次方程 y² + 2hy - A = 0
y_min_analytic = -h + np.sqrt(h**2 + A)

# 使用解析解计算最低电势（关键修改）
V_min_analytic = V(y_min_analytic, sigma, h, E_ext)

print(f"通过解析解得到的电势最低点: y_min = {y_min_analytic:.2e} m")
print(f"解析解得到的最低电势: V_min = {V_min_analytic:.6f} V")

# 计算y_zero处的电势
V_ref = V(y_zero, sigma, h, E_ext)
potential_diff = V_min_analytic - V_ref  # 使用解析解的电势差

print(f"\ny=0.02e-6 = {y_zero:.1e} m 处的电势: V = {V_ref:.6f} V")
print(f"电势差 ΔV = V_min - V(y=0.02e-6) = {potential_diff:.6f} V")

# 计算曲线最左端和最低点的电势差和距离差（全部使用解析解）
y_left = 0.02e-6  # 曲线最左端
V_left = V_ref
V_min_point = V_min_analytic  # 使用解析解
y_min_point = y_min_analytic  # 使用解析解

potential_diff_left_min = V_left - V_min_point
distance_diff_left_min = y_min_point - y_left

print(f"\n【势阱内高度分析（全部使用解析解）】")
print(f"最左端: y = {y_left:.2e} m, V = {V_left:.6f} V")
print(f"最低点: y = {y_min_point:.2e} m, V = {V_min_point:.6f} V")
print(f"电势差 ΔV = V_left - V_min = {potential_diff_left_min:.6f} V")
print(f"距离差 Δy = y_min - y_left = {distance_diff_left_min:.2e} m = {distance_diff_left_min*1e9:.2f} nm")

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(y_vals, V_vals, 'b-', linewidth=4, label=f'$E_{{ext}} = {E_ext:.0e}$ V/m')

# 标记最低点（使用解析解）
plt.plot(y_min_analytic, V_min_analytic, 'ro', markersize=12, 
         label=f'Min: y={y_min_analytic:.2e}m, V={V_min_analytic:.3f}V')

# 标记y_zero处的点
plt.axvline(x=y_zero, color='green', linestyle='--', alpha=0.5)
plt.plot(y_zero, V_ref, 'go', markersize=12, label=f'$y=0.02e-6$: {V_ref:.3f}V')

plt.xlabel('y (m)', fontsize=12)
plt.ylabel(r'$V(y)$ (V)', fontsize=12)
plt.title(r'$V(y)=\frac{\sigma}{2\pi\varepsilon_0}\ln\left(1+\frac{2h}{y}\right)+E_{\mathrm{ext}} y$', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10, loc='upper right')
plt.tight_layout()

# 设置合适的y轴范围以显示最低点
y_margin = 0.3 * (np.max(V_vals) - np.min(V_vals))
plt.ylim(V_min_analytic - y_margin, np.max(V_vals) + 0.1*y_margin)  # 使用解析解

plt.show()