import numpy as np
import matplotlib.pyplot as plt

# 物理参数
sigma = 1e-5  # C/m^2
eps0 = 8.854e-12  # F/m
H = 1e-6  # m
E_ext = -1e5  # V/m, 方向向下 (-z)

coeff = sigma / (2 * eps0)  # 电场系数 V/m

# 不同半径 R 的取值（单位：米）
R_list = [1e-6, 2e-6, 5e-6, 7e-6, 10e-6]  # 例如 1μm, 2μm, 5μm, 10μm
R_labels = ['1 μm', '2 μm', '5 μm','7 μm', '10 μm']

# 轴线上的z点 (z>0)
z_m = np.linspace(0, 10e-6, 1000)  # 从0到10微米

# 创建图形
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

# 循环计算每个 R 对应的曲线
for i, R in enumerate(R_list):
    def E_disks(z, R_val=R):
        term1 = (z + 2*H) / np.sqrt((z + 2*H)**2 + R_val**2)
        term2 = z / np.sqrt(z**2 + R_val**2)
        return coeff * (term1 - term2)

    def V_disks_inf_ref(z, R_val=R):
        term1 = np.sqrt(z**2 + R_val**2)
        term2 = np.sqrt((z + 2*H)**2 + R_val**2)
        return coeff * (term1 - term2)

    # 总电场
    E_total = E_disks(z_m) + E_ext
    # 总电势
    V_total = V_disks_inf_ref(z_m) - E_ext * z_m

    # 绘制电场曲线
    ax1.plot(z_m*1e6, E_total/1e3, label=f'R = {R_labels[i]}')
    # 绘制电势曲线
    ax2.plot(z_m*1e6, V_total, label=f'R = {R_labels[i]}')

    # --- 找到并标注电势最低点与 z=0 的差值 ---
    V0 = V_total[0]  # z=0 处的电势
    min_idx = np.argmin(V_total)  # 最低点的索引
    z_min = z_m[min_idx] * 1e6  # 最低点位置（μm）
    V_min = V_total[min_idx]  # 最低点电势
    delta_V = V0 - V_min  # 电势下降幅度

    # 标注文本：包含 z_min 和 ΔV
    label_text = f'z_min={z_min:.2f} μm\nΔV={delta_V:.2f} V'
    
    # 在最低点处添加标注
    ax2.annotate(label_text, 
                 xy=(z_min, V_min), 
                 xytext=(z_min + 0.5, V_min + 0.1 * delta_V), 
                 arrowprops=dict(arrowstyle='->', lw=0.5, color='gray'),
                 fontsize=8, 
                 ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

# 设置电场图
ax1.set_xlabel('z (μm)')
ax1.set_ylabel('E (kV/m)')
ax1.set_title('Total Electric Field on Axis (with external field)')
ax1.legend()
ax1.grid(True)

# 设置电势图
ax2.set_xlabel('z (μm)')
ax2.set_ylabel('V (V)')
ax2.set_title('Total Electric Potential (infinity reference)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 打印关键点的值（以最后一个 R 为例）
R = R_list[-1]  # 取最后一个 R 打印示例
def E_disks(z, R_val=R):
    term1 = (z + 2*H) / np.sqrt((z + 2*H)**2 + R_val**2)
    term2 = z / np.sqrt(z**2 + R_val**2)
    return coeff * (term1 - term2)

def V_disks_inf_ref(z, R_val=R):
    term1 = np.sqrt(z**2 + R_val**2)
    term2 = np.sqrt((z + 2*H)**2 + R_val**2)
    return coeff * (term1 - term2)

E_total = E_disks(z_m) + E_ext
V_total = V_disks_inf_ref(z_m) - E_ext * z_m

print(f"R = {R*1e6} μm 时的关键点值：")
print("z(μm)\tE_total(kV/m)\tV_total(V)")
for z in [0, 0.5, 1, 2, 5, 10]:
    idx = np.argmin(np.abs(z_m - z*1e-6))
    print(f"{z}\t{E_total[idx]/1e3:.3f}\t{V_total[idx]:.3f}")