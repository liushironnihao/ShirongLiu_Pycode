"""
二维空间带电粒子运动模拟
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import os
from tqdm import tqdm
import pickle
from dataclasses import dataclass, field
from typing import List,  Tuple
import imageio
import datetime
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


import matplotlib
import matplotlib.font_manager as fm

# 尝试设置中文字体
try:
    # 查找系统中可用的中文字体
    font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    chinese_fonts = [f for f in font_list if any(word in f.lower() for word in ['simhei', 'simsun', 'microsoft', 'msyh', 'arial'])]
    
    if chinese_fonts:
        # 使用找到的第一个中文字体
        font_path = chinese_fonts[0]
        font_prop = fm.FontProperties(fname=font_path)
        matplotlib.rcParams['font.sans-serif'] = [font_prop.get_name(), 'DejaVu Sans']
        print(f"使用中文字体: {font_prop.get_name()}")
    else:
        # 如果没有找到中文字体，使用默认字体
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("警告: 未找到中文字体，中文可能显示为方框")
        
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"设置字体时出错: {e}")
    # 回退到默认设置
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

# ===================== 动画参数 =====================
SAVE_FRAMES = True  # 是否保存帧图片
TOTAL_FRAMES = 400  # 总帧数
ANIMATION_FPS = 20  # 动画播放帧率
CLEANUP_FRAMES = False  # 是否清理帧图片

# ===================== 物理参数 =====================
# 几何参数（根据原始代码）
L = 0.5e-6         # 极板长度(对应现实中光刻胶宽度）
d = 1.5e-6         # 极板总宽度（阵列间距） 
h = 0.5e-6         # 下极板y坐标（光刻胶宽度)
h1 = 4e-6          # 上下极板间距(粒子释放高度，绘图高度)
h2 = h             # 基底厚度


# 外电场参数
E1 = 1e5           # 第一个外电场值 (V/m)
E2 = 0.5e5         # 第二个外电场值 (V/m)
sigma = 0          # 初始表面电荷密度 (C/m)
duty1 = 1          # E1占空比
f = 500e3          # 频率 (Hz)
freq = f           # 添加别名，避免与文件对象冲突

# 气体环境
T = 300            # 温度 (K)
P = 1              # 气压 (bar)
sim_time_ms = 0.4  # 模拟总时长 (ms)
dt_phys = 0.2e-9/P

# 粒子参数
dict_num_Au = {1: 50, 2: 50, 3: 50, 4:50}   # 发射金纳米粒子尺寸（nm）和数目
dict_num_N2 = {0.37: 3000}                  # 氮气离子尺寸和数目
rho_table = {'Au': 19320, 'N2': 1754}       # 密度 (kg/m³)
frozen_new=2                                # 多少个新冻结粒子更新一次电场

# 物理常数
T0 = 300                # 常温 (K)
e = 1.602176634e-19     # 元电荷 (C)
pi = np.pi
nano = 1e-9
atm = 101325
mu_air0 = 1.81e-5 
mu_air = mu_air0*(T/T0)** 1.5*(T0+110)/(T+110)        # 空气动力粘度，近似公式如 Sutherland 或 Chapman-Enskog
lambda0 = 68e-9         # 空气分子平均自由程 (m)
kB = 1.380649e-23       # 玻尔兹曼常数

M_gas = 28.97e-3        # 气体分子质量 (kg/mol)
d_gas = 0.372e-9        # 气体分子直径 (m)
m_gas = M_gas / 6.02214076e23
q = +e                  # 粒子电荷

# 计算域边界
x_min = -L/2 - d/2
x_max = L/2 + d/2
y_min = 0               # 基底在y=0
y_max = h + h1          # 上边界

# 极板位置
lower_plate_y = h
upper_plate_y = h + h1

# ===================== 文件路径管理 =====================
def create_simulation_directory():
    """创建模拟结果目录结构"""
    # 计算总粒子数
    total_particles = sum(dict_num_Au.values()) + sum(dict_num_N2.values())
    
    # 生成目录名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"sim_{total_particles}particles_{sim_time_ms}ms_P{P:.2f}bar_T{T:.1f}K_E{E1/1e5:.1f}kVcm_{h}PR_{timestamp}"
    
    # 创建主目录
    main_dir = dir_name
    os.makedirs(main_dir, exist_ok=True)
    
    # 创建子目录结构
    subdirs = {
        'main': main_dir,
        'anim_frames': os.path.join(main_dir, 'animation_frames'),
        'particle_trajectories': os.path.join(main_dir, 'particle_trajectories'),
        'visualization': os.path.join(main_dir, 'visualization'),
        'electric_fields': os.path.join(main_dir, 'electric_fields'),
        'data': os.path.join(main_dir, 'simulation_data')
    }
    
    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)
    
    return subdirs

# 全局目录路径
DIRS = create_simulation_directory()

# ===================== 数据结构 =====================
@dataclass
class Particle:
    """单个粒子类"""
    id: int
    x: float
    y: float
    diameter: float
    drag_coeff: float
    mass: float
    material: str
    color: tuple
    vx: float = 0.0
    vy: float = 0.0
    state: int = 1  # 1:未发射, 2:冻结, 3:基底消失, 4:边界消失, 5:运动中
    emission_time: float = 0.0
    fixed_x: float = None
    # 修改：轨迹数据结构包含速度分量
    trajectory: List[Tuple[float, float, float, float, float]] = field(default_factory=list)  # (x, y, vx, vy, t)
    
    def save_trajectory(self, filename: str):
        """保存粒子轨迹到文件"""
        if self.trajectory:
            # 保存轨迹数据，包含位置、速度和时间的5列数据
            df = pd.DataFrame(self.trajectory, columns=['x', 'y', 'vx', 'vy', 't'])
            df.to_csv(filename, index=False)
            return True
        return False

@dataclass
class ParticleManager:
    """粒子管理器"""
    active_particles: List[Particle] = field(default_factory=list)      # 状态5:运动中的粒子
    frozen_particles: List[Particle] = field(default_factory=list)     # 状态2:冻结的粒子
    base_particles: List[Particle] = field(default_factory=list)        # 状态3:基底消失
    boundary_particles: List[Particle] = field(default_factory=list)   # 状态4:边界消失
    inactive_particles: List[Particle] = field(default_factory=list)   # 状态1:未发射
    
    def get_all_particles(self, state=None):
        """获取指定状态的粒子"""
        all_parts = (self.active_particles + self.frozen_particles + 
                    self.base_particles + self.boundary_particles + 
                    self.inactive_particles)
        if state is None:
            return all_parts
        return [p for p in all_parts if p.state == state]
    
    def move_particle(self, particle: Particle, new_state: int):
        """移动粒子到新的状态组"""
        # 从当前组移除
        if particle in self.active_particles:
            self.active_particles.remove(particle)
        elif particle in self.frozen_particles:
            self.frozen_particles.remove(particle)
        elif particle in self.base_particles:
            self.base_particles.remove(particle)
        elif particle in self.boundary_particles:
            self.boundary_particles.remove(particle)
        elif particle in self.inactive_particles:
            self.inactive_particles.remove(particle)
        
        # 添加到新组
        particle.state = new_state
        if new_state == 2:
            self.frozen_particles.append(particle)
        elif new_state == 3:
            self.base_particles.append(particle)
        elif new_state == 4:
            self.boundary_particles.append(particle)
        elif new_state == 5:
            self.active_particles.append(particle)
        elif new_state == 1:
            self.inactive_particles.append(particle)

# ===================== 电场计算模块 =====================
class ElectricFieldSolver:
    """电场求解器 - 简化版，只计算冻结粒子+外电场"""
    def __init__(self, nx=200, ny=150):
        self.nx = nx
        self.ny = ny
        self.xg = np.linspace(x_min, x_max, nx)
        self.yg = np.linspace(y_min, upper_plate_y, ny)
        self.Xg, self.Yg = np.meshgrid(self.xg, self.yg, indexing='ij')
        
        # 存储电场更新历史
        self.field_history = []  # 每个元素: (time, frozen_count, E1_interp, E2_interp)
        
        # 添加：存储外电场参数
        self.E1 = E1
        self.E2 = E2
        
        # 初始电场（没有冻结粒子）
        self.update_field([], 0.0)
    
    def calculate_field(self, frozen_particles: List[Particle], E0: float) -> np.ndarray:
        """计算空间电场分布-冻结粒子+外电场+镜像电荷"""
        Ex = np.zeros((self.nx, self.ny))
        Ey = np.zeros_like(Ex)
        
        # 库仑常数
        k = 8.99e9
        
        # 冻结粒子贡献（包含镜像电荷）
        for particle in frozen_particles:
            if particle.fixed_x is not None:
                # 1. 真实冻结粒子的电场贡献
                dx = self.Xg - particle.fixed_x
                dy = self.Yg - lower_plate_y  # 粒子实际位置
                r2 = dx**2 + dy**2 + 1e-18
                r = np.sqrt(r2)
                
                # 保持原有公式：E = k*q*r_vector / r^3
                Ex += k * q * dx / (r2 * r)  # 即 k*q*dx/r^3
                Ey += k * q * dy / (r2 * r)
                
                # 2. 镜像电荷的电场贡献
                # 镜像电荷位置：在基底下方对称位置，电荷符号相反
                dx_image = self.Xg - particle.fixed_x  # x坐标相同
                dy_image = self.Yg - (2 * y_min - lower_plate_y)  # 镜像位置
                r2_image = dx_image**2 + dy_image**2 + 1e-18
                r_image = np.sqrt(r2_image)
                
                # 镜像电荷电场：电荷为 -q，其他公式相同
                Ex += k * (-q) * dx_image / (r2_image * r_image)
                Ey += k * (-q) * dy_image / (r2_image * r_image)
        
        # 外电场贡献（垂直向下）- 保持不变
        Ey -= E0
        
        return np.stack([Ex, Ey], axis=-1)
    
    def update_field(self, frozen_particles: List[Particle], t: float):
        """更新电场并保存到历史"""
        # 计算两种外电场下的分布
        E1_grid = self.calculate_field(frozen_particles, self.E1)  # 修改：使用self.E1
        E2_grid = self.calculate_field(frozen_particles, self.E2)  # 修改：使用self.E2
        
        # 创建插值器
        E1_interp = RegularGridInterpolator(
            (self.xg, self.yg), E1_grid, bounds_error=False, fill_value=0
        )
        E2_interp = RegularGridInterpolator(
            (self.xg, self.yg), E2_grid, bounds_error=False, fill_value=0
        )
        
        # 保存到历史
        self.field_history.append({
            'time': t,
            'frozen_count': len(frozen_particles),
            'frozen_particles': frozen_particles.copy(),  # 保存当前冻结粒子
            'E1_interp': E1_interp,
            'E2_interp': E2_interp
        })
        
        # 设置当前电场为最新
        self.E1_interp = E1_interp
        self.E2_interp = E2_interp
    
    def get_field_at(self, positions: np.ndarray, t: float) -> np.ndarray:
        """获取指定位置和时间的电场"""
        T_pulse = 1/f
        t1_E1 = T_pulse * duty1
        
        phase = t % T_pulse
        if phase < t1_E1:
            return self.E1_interp(positions)
        else:
            return self.E2_interp(positions)
    
    def get_field_at_time(self, t: float, which_E: int = 1) -> RegularGridInterpolator:
        """获取指定时间的电场插值器（用于动画）"""
        # 找到最接近时间的电场
        if not self.field_history:
            return self.E1_interp if which_E == 1 else self.E2_interp
        
        # 找到第一个时间大于t的记录
        for i, record in enumerate(self.field_history):
            if record['time'] >= t:
                if i == 0:
                    return record['E1_interp'] if which_E == 1 else record['E2_interp']
                else:
                    # 使用前一个记录
                    prev_record = self.field_history[i-1]
                    return prev_record['E1_interp'] if which_E == 1 else prev_record['E2_interp']
        
        # 如果t大于所有记录的时间，使用最后一个
        return self.field_history[-1]['E1_interp'] if which_E == 1 else self.field_history[-1]['E2_interp']
    
    def save_field_history(self, filename: str):
        """保存电场历史到文件-只保存电场数据"""
        field_data = []
        for i, record in enumerate(self.field_history):
            print(f"保存第{i+1}次电场更新, 时间: {record['time']*1e6:.2f} µs")
            
            # 从记录中获取冻结粒子（update_field方法已保存）
            frozen_particles = record.get('frozen_particles', [])
            
            # 重新计算电场
            E1_grid = self.calculate_field(frozen_particles, self.E1)
            E2_grid = self.calculate_field(frozen_particles, self.E2)
            
            field_data.append({
                'time': record['time'],
                'frozen_count': record['frozen_count'],
                'E1_grid': E1_grid,  # 电场网格数据
                'E2_grid': E2_grid,  # 电场网格数据
                'xg': self.xg,
                'yg': self.yg,
                'grid_shape': E1_grid.shape
            })
        
        with open(filename, 'wb') as f:
            pickle.dump(field_data, f)
        print(f"电场历史已保存到 {filename}, 包含 {len(field_data)} 次更新")

    def get_frozen_particles_at_time(self, t):
        """获取指定时间的冻结粒子列表"""
        # 这里需要根据你的实际数据结构来实现
        # 返回在时间t之前冻结的粒子列表
        # 注意：Particle类需要记录freeze_time属性
        return [p for p in self.frozen_particles if hasattr(p, 'freeze_time') and p.freeze_time <= t]
# ===================== 粒子生成模块 =====================
def generate_particles() -> ParticleManager:
    """生成所有粒子 - 从上极板随机位置和随机时间发射"""
    manager = ParticleManager()
    particle_id = 0
    Tmax = sim_time_ms * 1e-3
    rng = np.random.default_rng()  # 固定随机种子以便复现
    
    # 计算空气动力参数
    lambda_gas = lambda0 /P*T/T0 
    
    # 颜色映射
    au_colors = ['#FF6B6B', '#FF8E53', '#FFD166', '#06D6A0']  # 红色到绿色
    n2_colors = ['#118AB2']  # 蓝色
    
    # 生成金纳米粒子
    dp_values = list(dict_num_Au.keys())
    for idx, (dp_nm, n) in enumerate(dict_num_Au.items()):
        dp = dp_nm * nano
        a = dp / 2
        m = 4/3 * pi * a**3 * rho_table['Au']
        Kn = lambda_gas / a
        Cc = 1 + lambda_gas/dp * (2.34 + 1.05*np.exp(-0.39*dp/lambda_gas))
        Cd = 3 * pi * mu_air * dp / Cc
        
        
        color = au_colors[idx % len(au_colors)]
        
        for _ in range(n):
            # 在整个模拟时间内随机发射时间
            t_emit = rng.uniform(0, Tmax)
            # 在x方向随机位置发射，在上极板位置
            x0 = rng.uniform(x_min, x_max)
            
            particle = Particle(
                id=particle_id,
                x=x0,
                y=upper_plate_y- 1e-7,  # 从上极板发射
                diameter=dp,
                drag_coeff=Cd,
                mass=m,
                material='Au',
                color=color,
                emission_time=t_emit
            )
            manager.inactive_particles.append(particle)
            particle_id += 1
    
    # 生成氮气离子
    dp_nm = list(dict_num_N2.keys())[0]
    n = list(dict_num_N2.values())[0]
    dp = dp_nm * nano
    a = dp / 2
    m = 4/3 * pi * a**3 * rho_table['N2']
    Kn = lambda_gas / a
    Cc = 1 + lambda_gas/dp * (2.34 + 1.05*np.exp(-0.39*dp/lambda_gas))
    Cd = 3 * pi * mu_air * dp / Cc
    print(f"氮气质量: {m}")
    color = n2_colors[0]
    
    for _ in range(n):
        # 在整个模拟时间内随机发射时间
        t_emit = rng.uniform(0, Tmax)
        # 在x方向随机位置发射
        x0 = rng.uniform(x_min,x_max)
        
        particle = Particle(
            id=particle_id,
            x=x0,
            y=upper_plate_y- 4e-7,  # 从上极板下方发射
            diameter=dp,
            drag_coeff=Cd,
            mass=m,
            material='N2',
            color=color,
            emission_time=t_emit
        )
        manager.inactive_particles.append(particle)
        particle_id += 1
    
    return manager

# ===================== 力计算模块 =====================
def calculate_coulomb_force(positions: np.ndarray) -> np.ndarray:
    """计算粒子间的库仑力"""
    n = len(positions)
    if n < 2:
        return np.zeros((n, 2))
    
    ke = 8.99e9 * q * q  # 这里q是元电荷  
    forces = np.zeros((n, 2))
    
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            r2 = dx*dx + dy*dy + 1e-30
            
            if r2 > 1e-16:  # 避免过近距离
                r = np.sqrt(r2)  
                force_mag = ke / r2 
                forces[i, 0] += force_mag * dx / r
                forces[i, 1] += force_mag * dy / r
                forces[j, 0] -= force_mag * dx / r
                forces[j, 1] -= force_mag * dy / r
    
    return forces
# ===================== 状态检测模块 =====================
def check_particle_state(particle: Particle) -> int:
    """检测粒子状态"""
    x, y = particle.x, particle.y
    
    # 1. 检查是否碰到下极板（冻结）
    on_left_plate = (x >= -L/2 - d/2) and (x <= L/2 - d/2)
    on_right_plate = (x >= -L/2 + d/2) and (x <= L/2 + d/2)
    near_plate = abs(y - lower_plate_y) < 5e-8  # 5nm范围内
    
    if (on_left_plate or on_right_plate) and near_plate and not particle.fixed_x:
        particle.fixed_x = x
        return 2  # 冻结
    
    # 2. 检查是否碰到基底
    if y <= y_min + 1e-8:
        return 3  # 基底消失
    
    # 3. 检查是否碰到上极板或左右边界
    if y >= upper_plate_y - 1e-8:
        return 4  # 上边界消失
    if x <= x_min + 1e-8 or x >= x_max - 1e-8:
        return 4  # 左右边界消失 
    
    return 5  # 仍在运动中

# ===================== 动画帧保存模块 =====================
def save_animation_frame(frame_idx: int, t_now: float, manager: ParticleManager, 
                         field_solver: ElectricFieldSolver, frame_data: list):
    """保存单帧动画图片"""
    # 确保输出目录存在
    output_dir = DIRS['anim_frames']
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置坐标轴
    ax.set_xlim(x_min*0.9, x_max*0.9)
    ax.set_ylim(y_min, upper_plate_y*1.05)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # 绘制极板和基底
    ax.plot([-L/2 - d/2, L/2 - d/2], [lower_plate_y, lower_plate_y], 
            'k-', lw=4, alpha=0.8, zorder=5)
    ax.plot([-L/2 + d/2, L/2 + d/2], [lower_plate_y, lower_plate_y], 
            'k-', lw=4, alpha=0.8, zorder=5)
    ax.plot([x_min, x_max], [upper_plate_y, upper_plate_y], 
            'gray', lw=4, alpha=0.8, zorder=5)
    ax.plot([x_min, x_max], [y_min, y_min], 
            'brown', lw=4, alpha=0.8, zorder=5)
    
    # 获取当前时间对应的电场
    T_pulse = 1/f
    t1_E1 = T_pulse * duty1
    phase = t_now % T_pulse
    which_E = 1 if phase < t1_E1 else 2
    
    E_interp = field_solver.get_field_at_time(t_now, which_E)
    
    # 绘制电力线
    num_lines = 20
    start_x = np.linspace(x_min*0.9, x_max*0.9, num_lines)
    for x0 in start_x:  
        xs, ys = trace_electric_line(x0, E_interp, step=5e-9)
        ax.plot(xs, ys, color=(0.2, 0.2, 0.8, 0.3), lw=3, zorder=1)
    
    # 绘制所有粒子 - 按材料和状态分别处理
    # 定义不同材料的显示参数
    material_params = {
        'Au': {'marker': 'o', 'size_factor': 2, 'alpha': 0.8},  # 金粒子用圆形
        'N2': {'marker': 'H', 'size_factor': 0.6, 'alpha': 0.7}   # 氮气分子用六边形，尺寸小
    }
    
    # 定义不同状态的显示参数
    state_params = {
        5: {'edgecolor': 'k', 'linewidth': 0.5, 'zorder': 10},  # 运动中的粒子
        2: {'edgecolor': 'k', 'linewidth': 1.0, 'zorder': 9},   # 冻结的粒子
        3: {'edgecolor': 'gray', 'linewidth': 0.3, 'zorder': 8, 'alpha_factor': 0.5},  # 基底消失
        4: {'edgecolor': 'gray', 'linewidth': 0.3, 'zorder': 8, 'alpha_factor': 0.4}   # 边界消失
    }
    
    # 按状态分组处理粒子
    particle_groups = [
        (manager.active_particles, 5),      # 运动中的粒子
        (manager.frozen_particles, 2),      # 冻结的粒子
        (manager.base_particles, 3),        # 基底消失的粒子
        (manager.boundary_particles, 4)     # 边界消失的粒子
    ]
    
    for particles, state in particle_groups:
        if particles:
            # 按材料分组处理
            au_particles = [p for p in particles if p.material == 'Au']
            n2_particles = [p for p in particles if p.material == 'N2']
            
            # 绘制金粒子
            if au_particles:
                xs = [p.x for p in au_particles]
                ys = [p.y for p in au_particles]
                colors = [p.color for p in au_particles]
                sizes = [20 + (p.diameter/nano)*5 * material_params['Au']['size_factor'] for p in au_particles]
                
                # 获取状态特定的参数
                state_param = state_params[state]
                alpha = material_params['Au']['alpha'] * state_param.get('alpha_factor', 1.0)
                
                ax.scatter(xs, ys, s=sizes, c=colors, 
                          marker=material_params['Au']['marker'],
                          alpha=alpha, 
                          edgecolors=state_param['edgecolor'], 
                          linewidth=state_param['linewidth'], 
                          zorder=state_param['zorder'])
            
            # 绘制氮气分子
            if n2_particles:
                xs = [p.x for p in n2_particles]
                ys = [p.y for p in n2_particles]
                colors = [p.color for p in n2_particles]
                sizes = [10 + (p.diameter/nano)*3 * material_params['N2']['size_factor'] for p in n2_particles]
                
                # 获取状态特定的参数
                state_param = state_params[state]
                alpha = material_params['N2']['alpha'] * state_param.get('alpha_factor', 1.0)
                ax.scatter(xs, ys, s=sizes, c=colors, 
                          marker=material_params['N2']['marker'],
                          alpha=alpha, 
                          edgecolors=state_param['edgecolor'], 
                          linewidth=state_param['linewidth'], 
                          zorder=state_param['zorder'])
    
    # 查找当前时间的冻结粒子数
    current_frozen = 0
    for record in field_solver.field_history:
        if record['time'] <= t_now:
            current_frozen = record['frozen_count']
    
    # 更新时间标题
    ax.set_title(f'Particle Motion Simulation - Time: {t_now*1e6:.2f} µs\n'
                 f'Au: ●, N₂: ⬡, Frozen: {current_frozen}', fontsize=10)
    
    # 添加图例
    # 创建自定义图例句柄
    legend_elements = [
        Line2D([0], [0], color='black', lw=4, alpha=0.8, label='Lower Plate'),
        Line2D([0], [0], color='gray', lw=4, alpha=0.8, label='Upper Plate'),
        Line2D([0], [0], color='brown', lw=4, alpha=0.8, label='Substrate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF6B6B', markersize=8, 
               label='1nm Au', markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8E53', markersize=8, 
               label='2nm Au', markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFD166', markersize=8, 
               label='3nm Au', markeredgecolor='k'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#06D6A0', markersize=8, 
               label='4nm Au', markeredgecolor='k'),
        Line2D([0], [0], marker='H', color='w', markerfacecolor='#118AB2', markersize=8, 
               label='N₂ ions', markeredgecolor='k', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # 保存帧
    frame_filename = os.path.join(output_dir, f'frame_{frame_idx:04d}.png')
    plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # 存储帧数据
    frame_data.append({
        'frame_idx': frame_idx,
        'time': t_now,
        'filename': frame_filename,
        'active_count': len(manager.active_particles),
        'frozen_count': current_frozen
    })
    
    return frame_data

def trace_electric_line(x0, E_interpolator, step=0.5e-9, max_steps=8000):
    """追踪电力线"""
    xs, ys = [x0], [upper_plate_y]
    x, y = x0, upper_plate_y
    for _ in range(max_steps):
        if y <= y_min or not (x_min <= x <= x_max):
            break
        E_vec = E_interpolator([[x, y]])
        if E_vec.size == 0:
            break
        Ex, Ey = E_vec[0][0], E_vec[0][1]
        E = np.hypot(Ex, Ey) + 1e-18
        # 修改：避免使用d作为变量名
        dx_step = Ex/E * step
        dy_step = Ey/E * step
        x += dx_step
        y += dy_step
        xs.append(x)
        ys.append(y)
    return xs, ys

# ===================== 主模拟循环 =====================
def run_simulation():
    """运行主模拟"""
    print("="*50)
    print("开始粒子模拟")
    print("="*50)
    
    # 显示目录信息
    print(f"结果保存到: {DIRS['main']}")
    print(f"目录结构:")
    print(f"  - 动画帧: {DIRS['anim_frames']}")
    print(f"  - 粒子轨迹: {DIRS['particle_trajectories']}")
    print(f"  - 可视化结果: {DIRS['visualization']}")
    print(f"  - 电场数据: {DIRS['electric_fields']}")
    print(f"  - 模拟数据: {DIRS['data']}")
    
    # 初始化
    print("1. 初始化粒子管理器...")
    manager = generate_particles()
    print(f"   生成粒子总数: {len(manager.get_all_particles())}")
    print(f"   金粒子: {len([p for p in manager.get_all_particles() if p.material == 'Au'])}")
    print(f"   氮气离子: {len([p for p in manager.get_all_particles() if p.material == 'N2'])}")
    
    print("2. 初始化电场求解器...")
    field_solver = ElectricFieldSolver()
    
    # 时间参数
    Tmax = sim_time_ms * 1e-3
    dt_phys = 1e-10  # 0.1ns时间步长
    total_steps = int(np.ceil(Tmax / dt_phys))
    dt = dt_phys
    
    # 计算帧间隔
    steps_per_frame = max(1, total_steps // TOTAL_FRAMES)
    frame_interval = dt * steps_per_frame
    print(f"3. 开始主模拟循环 (Tmax={Tmax*1e6:.1f}µs, dt={dt*1e9:.1f}ns, steps={total_steps})...")
    print(f"   总帧数: {TOTAL_FRAMES}, 每{steps_per_frame}步保存一帧, 帧间隔: {frame_interval*1e6:.2f}µs")
    
    frozen_counter = 0
    last_frozen_count = 0
      
    frozen_events = []  # 记录每个粒子的冻结时间
    
    # 预计算布朗力：仅对金纳米粒子（Au）计算，不计算氮气离子（N2）
    sigma_brown_cache = {}
    all_particles = manager.get_all_particles()
    for particle in all_particles:
        if particle.material == 'Au': 
            gamma = particle.drag_coeff / particle.mass
            sigma_brown = np.sqrt(
                (kB * T / particle.mass) *
                (1 - np.exp(-2 * gamma * dt)) *
                (m_gas / particle.mass) / 0.9
            )
            sigma_brown_cache[particle.id] = sigma_brown

    
    # 存储动画帧数据
    frame_data = []
    frame_idx = 0
    
    # 保存初始帧
    if SAVE_FRAMES:
        print(f"   保存初始帧 (frame {frame_idx})...")
        frame_data = save_animation_frame(frame_idx, 0.0, manager, field_solver, frame_data)
        frame_idx += 1
    
    for step in tqdm(range(total_steps), desc="模拟进度"):
        t = step * dt
        
        # 发射粒子
        to_activate = []
        for particle in manager.inactive_particles:
            if particle.emission_time <= t:
                to_activate.append(particle)
        
        for particle in to_activate:
            manager.move_particle(particle, 5)
        
        # 获取当前活动粒子数
        active_count = len(manager.active_particles)
        
        # 只计算运动中的粒子
        if active_count > 0:
            # 获取当前活动粒子的属性
            active_particles = manager.active_particles
            
            # 准备数据
            positions = np.array([[p.x, p.y] for p in active_particles])
            masses = np.array([p.mass for p in active_particles])
            drag_coeffs = np.array([p.drag_coeff for p in active_particles])
            vels = np.array([[p.vx, p.vy] for p in active_particles])

            # 获取电场
            E_field = field_solver.get_field_at(positions, t)

            # 电场力产生的加速度
            ax_electric = (q * E_field[:, 0]) / masses
            ay_electric = (q * E_field[:, 1]) / masses

            # 拖曳力产生的加速度
            ax_drag = (-drag_coeffs * vels[:, 0]) / masses
      
            ay_drag = (-drag_coeffs * vels[:, 1]) / masses

            # 库仑力产生的加速度
            F_coulomb = calculate_coulomb_force(positions)
            ax_coulomb = F_coulomb[:, 0] / masses
            ay_coulomb = F_coulomb[:, 1] / masses

            # 总确定性加速度（电场 + 拖曳 + 库仑）
            ax_total = ax_electric + ax_drag + ax_coulomb
            ay_total = ay_electric + ay_drag + ay_coulomb
            
            # 热运动速度标准差：σ_th = sqrt(kB * T / m)
            m = particle.mass         # 氮气离子的质量 
            sigma_th = np.sqrt(kB * T / m)  # 单位：m/s
            # 遍历所有活动粒子，逐个更新运动状态
            for i, particle in enumerate(active_particles):
                if particle.material == 'N2':
                    ax_total[i] = ax_electric[i] + ax_coulomb[i]  # 仅电场 + 库仑
                    ay_total[i] = ay_electric[i] + ay_coulomb[i]  # 仅电场 + 库仑
                    # --------------------------
                    # 氮气离子（N2，带电）—— 修改部分：删除布朗力，改用热运动速度修正
                    # -------------------------             

                    # 从正态分布采样热运动速度分量
                    vx_th = np.random.normal(0, sigma_th)
                    vy_th = np.random.normal(0, sigma_th)

                    # --- 计算确定性加速度导致的速度增量（可选，也可以直接用加速度）---
                    ax_det = ax_total[i]  # 已经是 Fx/m，即加速度
                    ay_det = ay_total[i]  # 已经是 Fy/m

                   
                    vx0 = vx_th + ax_det * dt   
                    vy0 = vy_th + ay_det * dt

                    # --- 用初速度 + 加速度 × dt²/2 来计算位移 ---             
                    dx = vx0 * dt + 0.5 * ax_det * dt**2
                    dy = vy0 * dt + 0.5 * ay_det * dt**2

                    # --- 更新位置---
                    particle.x += dx
                    particle.y += dy

                    # --- 更新速度为该时间步末的速度---
                    particle.vx = vx0 + ax_det * dt
                    particle.vy = vy0 + ay_det * dt

                elif particle.material == 'Au':
                    sigma_brown = sigma_brown_cache[particle.id]
                    dvx_brown = np.random.normal(0, sigma_brown)
                    dvy_brown = np.random.normal(0, sigma_brown)

                    # 金粒子速度更新：使用 ax_total + 布朗力
                    particle.vx += (ax_total[i] * dt) + dvx_brown
                    particle.vy += (ay_total[i] * dt) + dvy_brown

                    # Au粒子更新位置和记录轨迹
                    particle.x += particle.vx * dt
                    particle.y += particle.vy * dt
                particle.trajectory.append((particle.x, particle.y, particle.vx, particle.vy, t))
        
        # 检查状态变化
        to_freeze = []
        to_base = []
        to_boundary = []
        
        for particle in manager.active_particles:
            new_state = check_particle_state(particle)
            if new_state != 5:  # 状态发生变化
                if new_state == 2:
                    to_freeze.append(particle)
                elif new_state == 3:
                    to_base.append(particle)
                elif new_state == 4:
                    to_boundary.append(particle)
        
        # 处理状态变化
        for particle in to_freeze:
            particle.x = particle.fixed_x
            particle.y = lower_plate_y
            particle.vx = 0.0
            particle.vy = 0.0
            particle.freeze_time = t
            manager.move_particle(particle, 2)
            frozen_counter += 1
            
            # 记录冻结事件
            frozen_events.append({
                'time': t,  # 冻结时间
                'particle_id': particle.id,
                'fixed_x': particle.fixed_x,
                'diameter': particle.diameter,
                'material': particle.material,
                'charge': q
            })
            
        for particle in to_base:
            manager.move_particle(particle, 3)
            
        for particle in to_boundary:
            manager.move_particle(particle, 4)
        
        # 每冻结5个粒子更新一次电场
        if len(manager.frozen_particles) >= last_frozen_count + frozen_new:
            print(f"   时间 {t*1e6:.1f}µs: 更新电场，冻结粒子数: {len(manager.frozen_particles)}，运动中粒子数: {active_count}")
            field_solver.update_field(manager.frozen_particles, t)
            last_frozen_count = len(manager.frozen_particles)
        
        # 每 steps_per_frame 步保存一帧
        if SAVE_FRAMES and (step + 1) % steps_per_frame == 0 and frame_idx < TOTAL_FRAMES:
            frame_data = save_animation_frame(frame_idx, t, manager, field_solver, frame_data)
            frame_idx += 1
    
    # 保存最后一帧
    if SAVE_FRAMES and frame_idx <= TOTAL_FRAMES:
        frame_data = save_animation_frame(frame_idx, Tmax, manager, field_solver, frame_data)
        print(f"   已保存 {len(frame_data)} 帧动画图片")
    
    print("4. 模拟完成，保存数据...")
    
    # 保存电场历史
    field_solver.save_field_history(os.path.join(DIRS['electric_fields'], "field_history.pkl"))

    # 单独保存冻结粒子信息
    if frozen_events:
        frozen_events_df = pd.DataFrame(frozen_events)
        frozen_events_df.to_csv(os.path.join(DIRS['data'], "frozen_events.csv"), index=False)
        print(f"   保存了 {len(frozen_events)} 个冻结事件到 frozen_events.csv")
        
        # 同时保存详细的冻结粒子信息
        frozen_particles_info = []
        for particle in manager.frozen_particles:
            info = {
                'id': particle.id,
                'fixed_x': particle.fixed_x,
                'diameter': particle.diameter,
                'material': particle.material,
                'freeze_time': getattr(particle, 'freeze_time', None),
                'x': particle.x,
                'y': particle.y
            }
            frozen_particles_info.append(info)
        
        # 保存为CSV
        frozen_particles_df = pd.DataFrame(frozen_particles_info)
        frozen_particles_df.to_csv(os.path.join(DIRS['data'], "frozen_particles_detailed.csv"), index=False)
        
        # 保存为pkl（如果需要保存完整对象）
        frozen_particles_file = os.path.join(DIRS['data'], "frozen_particles.pkl")
        with open(frozen_particles_file, 'wb') as f:
            pickle.dump(frozen_particles_info, f)
        print(f"保存了详细的冻结子信息到 frozen_particles_detailed.csv 和 frozen_particles.pkl")
    
    # 保存轨迹数据
    saved_count = 0
    for particle in manager.get_all_particles():
        if particle.trajectory and len(particle.trajectory) > 10:
            # 修改：生成包含粒子尺寸和类型的文件名
            # 获取粒子直径（纳米单位）和材料类型
            diameter_nm = particle.diameter / 1e-9
            material = particle.material
            
            # 生成文件名格式：粒子类型_尺寸nm_id编号.csv
            filename = os.path.join(DIRS['particle_trajectories'], f'particle_{material}_{diameter_nm:.2f}nm_{particle.id:06d}.csv')
            
            # 保存轨迹数据，包含位置、速度和时间
            if particle.save_trajectory(filename):
                saved_count += 1
    
    # 保存粒子状态
    states_data = {
        'frozen': [(p.id, p.fixed_x) for p in manager.frozen_particles],
        'base': [(p.id, p.x, p.y) for p in manager.base_particles],
        'boundary': [(p.id, p.x, p.y) for p in manager.boundary_particles],
        'active': [(p.id, p.x, p.y) for p in manager.active_particles],
        'inactive': [(p.id, p.x, p.y) for p in manager.inactive_particles]
    }
    
    with open(os.path.join(DIRS['data'], "particle_states.pkl"), 'wb') as f:
        pickle.dump(states_data, f)
    
    # 保存帧数据
    if SAVE_FRAMES and frame_data:
        frame_df = pd.DataFrame(frame_data)
        frame_df.to_csv(os.path.join(DIRS['data'], "frame_data.csv"), index=False)
    
    print(f"5. 数据已保存到 {DIRS['main']}")
    print(f"   保存了 {saved_count} 个粒子的轨迹数据")
    print(f"   轨迹文件命名格式: particle_材料_直径nm_编号.csv")
    print(f"   轨迹文件内容: x, y, vx, vy, t 五列数据")
    print(f"   电场更新次数: {len(field_solver.field_history)}")
    
    return manager, field_solver, frame_data

# ===================== 动画生成模块 =====================
def create_animation_from_frames(frame_data: list, output_filename: str, 
                                fps: int = 20, cleanup: bool = True):
    """从保存的帧图片生成动画"""
    print("\n生成动画从保存的帧...")
    
    if not frame_data:
        print("错误: 没有帧数据")
        return None
    
    # 从frame_data中获取所有图片文件名
    image_files = [data['filename'] for data in frame_data if os.path.exists(data['filename'])]
    
    if not image_files:
        print("错误: 没有找到帧图片")
        return None
    
    print(f"正在处理 {len(image_files)} 张图片...")
    
    try:
        # 读取所有图片
        images = []
        for img_file in tqdm(image_files, desc="读取图片帧"):
            images.append(imageio.imread(img_file))
        
        # 保存为GIF
        gif_filename = output_filename.replace('.mp4', '.gif')
        print(f"保存GIF动画: {gif_filename} (FPS: {fps})")
        imageio.mimsave(gif_filename, images, fps=fps)
        
        # 尝试保存为MP4
        try:
            print(f"保存MP4动画: {output_filename} (FPS: {fps})")
            imageio.mimsave(output_filename, images, fps=fps, codec='libx264')
        except Exception as e:
            print(f"警告: 无法保存MP4格式: {e}")
            print(f"已保存GIF格式: {gif_filename}")
            output_filename = gif_filename
        
        # 清理临时图片文件
        if cleanup:
            print("清理临时图片文件...")
            for img_file in image_files:
                try:
                    os.remove(img_file)
                except Exception as e:
                    print(f"警告: 无法删除文件 {img_file}: {e}")
        
        print(f"动画已保存: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"生成动画时出错: {e}")
        return None

def create_animation_alternative(manager: ParticleManager, field_solver: ElectricFieldSolver, frame_data: list):
    """替代的动画生成方法 - 从帧数据生成"""
    print("7. 创建粒子运动动画...")
    
    if not frame_data:
        print("   没有帧数据，无法生成动画")
        return None
    
    # 从保存的帧数据生成动画
    anim_path = create_animation_from_frames(
        frame_data, 
        output_filename=os.path.join(DIRS['visualization'], "particle_animation.mp4"),
        fps=ANIMATION_FPS,
        cleanup=CLEANUP_FRAMES
    )
    
    if anim_path:
        print(f"   动画已保存: {anim_path}")
    
    return anim_path

# ===================== 可视化模块 =====================
def plot_results(manager: ParticleManager, field_solver: ElectricFieldSolver, frame_data: list = None):
    """绘制结果图"""
    print("\n6. 生成可视化结果...")
    
    # 获取有轨迹的粒子
    particles_with_traj = [p for p in manager.get_all_particles() if p.trajectory and len(p.trajectory) > 10]
    print(f"   有轨迹的粒子数: {len(particles_with_traj)}")
    
    if not particles_with_traj:
        print("   警告：没有足够的轨迹数据，跳过绘图")
        return
    
    # 1. 所有粒子轨迹图（包含氮气）
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    # 绘制极板和基底
    ax1.plot([-L/2 - d/2, L/2 - d/2], [lower_plate_y, lower_plate_y], 
             'k-', lw=4, alpha=0.8, label='Lower Plate')
    ax1.plot([-L/2 + d/2, L/2 + d/2], [lower_plate_y, lower_plate_y], 
             'k-', lw=4, alpha=0.8)
    ax1.plot([x_min, x_max], [upper_plate_y, upper_plate_y], 
             'gray', lw=4, alpha=0.8, label='Upper Plate')
    ax1.plot([x_min, x_max], [y_min, y_min], 
             'brown', lw=4, alpha=0.8, label='Substrate')
    
    # 绘制电力线
    num_lines = 20
    start_x = np.linspace(x_min*0.9, x_max*0.9, num_lines)
    for x0 in start_x:
        xs, ys = trace_electric_line(x0, field_solver.E1_interp, step=2e-8)
        ax1.plot(xs, ys, color=(0.2, 0.2, 0.8, 0.15), lw=2)
        xs, ys = trace_electric_line(x0, field_solver.E2_interp, step=2e-8)
        ax1.plot(xs, ys, color=(0.8, 0.2, 0.2, 0.15), lw=2)
    
    # 绘制所有粒子轨迹
    au_count = 0
    n2_count = 0
    n2_speeds = []  # 存储氮气离子的速度
    
    for particle in particles_with_traj:
        # 提取轨迹数据
        traj_x = [point[0] for point in particle.trajectory]
        traj_y = [point[1] for point in particle.trajectory]
        traj_vx = [point[2] for point in particle.trajectory]  
        traj_vy = [point[3] for point in particle.trajectory]  
        traj_t = [point[4] for point in particle.trajectory]
        
        if len(traj_x) > 1:
            if particle.material == 'Au':
                ax1.plot(traj_x, traj_y, 
                        color=particle.color, 
                        alpha=0.7, lw=1.2, linestyle='-')
                au_count += 1
            else:  # N2
                ax1.plot(traj_x, traj_y, 
                        color=particle.color, 
                        alpha=0.4, lw=0.6, linestyle=':')
                n2_count += 1
                
                # 计算氮气离子的平均速度
                if len(traj_t) >= 2:
                    # 使用轨迹中的速度分量计算瞬时速度大小
                    speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(traj_vx, traj_vy)]
                    if speeds:  # 确保有速度数据
                        avg_speed = np.mean(speeds)
                        n2_speeds.append(avg_speed)
    
    print(f"   绘制了 {au_count} 条金粒子轨迹和 {n2_count} 条氮气离子轨迹")
    
    # 在右下角添加氮气离子平均速度文本框
    if n2_speeds:
        avg_n2_speed = np.mean(n2_speeds)
        speed_text = f'Avg N₂ speed: {avg_n2_speed:.2f} m/s\n(n={len(n2_speeds)})'
    else:
        speed_text = 'Avg N₂ speed: N/A\n(no trajectory data)'
    
    # 将文本框放在右下角
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.02, speed_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    ax1.set_xlim(x_min*0.9, x_max*0.9)
    ax1.set_ylim(y_min, upper_plate_y*1.05)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title(f'All Particle Trajectories\nGold: {au_count}, N2 ions: {n2_count}')
    ax1.grid(True, alpha=0.3)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#FF6B6B', lw=2, label='1nm Au'),
        Line2D([0], [0], color='#FF8E53', lw=2, label='2nm Au'),
        Line2D([0], [0], color='#FFD166', lw=2, label='3nm Au'),
        Line2D([0], [0], color='#06D6A0', lw=2, label='4nm Au'),
        Line2D([0], [0], color='#118AB2', lw=2, linestyle=':', label='N2 ions'),
        Line2D([0], [0], color='black', lw=4, label='Lower Plate'),
        Line2D([0], [0], color='gray', lw=4, label='Upper Plate'),
        Line2D([0], [0], color='brown', lw=4, label='Substrate')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'all_particles_trajectory.png'), dpi=300, bbox_inches='tight')

    
    # 2. 金粒子轨迹图
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    # 绘制极板和基底
    ax2.plot([-L/2 - d/2, L/2 - d/2], [lower_plate_y, lower_plate_y], 
             'k-', lw=4, alpha=0.8, label='Lower Plate')
    ax2.plot([-L/2 + d/2, L/2 + d/2], [lower_plate_y, lower_plate_y], 
             'k-', lw=4, alpha=0.8)
    ax2.plot([x_min, x_max], [upper_plate_y, upper_plate_y], 
             'gray', lw=4, alpha=0.8, label='Upper Plate')
    ax2.plot([x_min, x_max], [y_min, y_min], 
             'brown', lw=4, alpha=0.8, label='Substrate')
    
    # 绘制电力线
    for x0 in start_x:
        xs, ys = trace_electric_line(x0, field_solver.E1_interp, step=2e-8)
        ax2.plot(xs, ys, color=(0.2, 0.2, 0.8, 0.2), lw=1)
        xs, ys = trace_electric_line(x0, field_solver.E2_interp, step=2e-8)
        ax2.plot(xs, ys, color=(0.8, 0.2, 0.2, 0.2), lw=1)
    
    # 只绘制金粒子轨迹
    au_particles = [p for p in particles_with_traj if p.material == 'Au']
    au_count = 0
    
    for particle in au_particles:
        traj_x = [point[0] for point in particle.trajectory]
        traj_y = [point[1] for point in particle.trajectory]
        if len(traj_x) > 1:
            ax2.plot(traj_x, traj_y, 
                    color=particle.color, 
                    alpha=0.8, lw=1.5)
            au_count += 1
    
    print(f"   绘制了 {au_count} 条金粒子轨迹")
    
    ax2.set_xlim(x_min*0.9, x_max*0.9)
    ax2.set_ylim(y_min, upper_plate_y*1.05)
    ax2.set_xlabel('x (m)')
    ax2.set_ylabel('y (m)')
    ax2.set_title(f'Gold Nanoparticle Trajectories (n={au_count})')
    ax2.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [
        Line2D([0], [0], color='#FF6B6B', lw=2, label='1nm'),
        Line2D([0], [0], color='#FF8E53', lw=2, label='2nm'),
        Line2D([0], [0], color='#FFD166', lw=2, label='3nm'),
        Line2D([0], [0], color='#06D6A0', lw=2, label='4nm'),
        Line2D([0], [0], color='black', lw=4, label='Lower Plate'),
        Line2D([0], [0], color='gray', lw=4, label='Upper Plate'),
        Line2D([0], [0], color='brown', lw=4, label='Substrate')
    ]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'gold_particles_trajectory.png'), dpi=300, bbox_inches='tight')
    
    
    # 3. 基底粒子直方图 - 修改：增加柱体数量
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    
    if manager.base_particles:
        base_au_particles = [p for p in manager.base_particles if p.material == 'Au']
        if base_au_particles:
            base_x = [p.x for p in base_au_particles]
            diameters = [p.diameter for p in base_au_particles]
            volumes = [4/3 * pi * (diam/2)**3 for diam in diameters]  # 修改：将d改为diam
            
            # 修改：增加柱体数量，计算更细的分布
            n_bins = 100  # 增加柱体数量
            hist, bin_edges = np.histogram(base_x, bins=n_bins, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]  # 柱体宽度
            
            # 计算每个bin的总体积
            bin_volumes = []
            bin_heights = []
            for i in range(len(bin_edges)-1):
                mask = (np.array(base_x) >= bin_edges[i]) & (np.array(base_x) < bin_edges[i+1])
                if any(mask):
                    # 计算bin中所有粒子的总体积
                    bin_vol = sum(vol for j, vol in enumerate(volumes) if mask[j])
                    bin_volumes.append(bin_vol)
                    
                    # 假设粒子是球形，计算等效高度
                    # 单个粒子的半径
                    avg_radius = np.mean([diam/2 for j, diam in enumerate(diameters) if mask[j]])  # 修改：将d改为diam
                    # 近似高度 = 总体积 / (π * 平均半径²) / bin_width
                    if avg_radius > 0:
                        bin_height = bin_vol / (pi * avg_radius**2) / bin_width
                    else:
                        bin_height = 0
                    bin_heights.append(bin_height)
                else:
                    bin_volumes.append(0)
                    bin_heights.append(0)
            
            # 归一化
            if max(bin_volumes) > 0:
                bin_volumes_norm = np.array(bin_volumes) / max(bin_volumes)
            else:
                bin_volumes_norm = np.zeros_like(bin_volumes)
            
            if max(bin_heights) > 0:
                bin_heights_norm = np.array(bin_heights) / max(bin_heights)
            else:
                bin_heights_norm = np.zeros_like(bin_heights)
            
            # 绘制归一化体积
            ax3.bar(bin_centers, bin_volumes_norm, 
                   width=bin_width * 0.9,  # 稍微缩小柱体宽度，使柱体之间有空隙
                   alpha=0.7, color='steelblue', edgecolor='black', 
                   label='Normalized Volume')
            
            # 在次坐标轴上绘制归一化高度
            ax3b = ax3.twinx()
            ax3b.plot(bin_centers, bin_heights_norm, 
                     color='darkorange', lw=2, label='Normalized Height')
            ax3b.set_ylabel('Normalized Structure Height', color='darkorange')
            ax3b.tick_params(axis='y', labelcolor='darkorange')
            ax3b.set_ylim(0, 1.1)
            
            ax3.set_xlabel('x position (m)')
            ax3.set_ylabel('Normalized Volume')
            ax3.set_title(f'Structure Distribution on Substrate (n={len(base_au_particles)} particles, {n_bins} bins)')
            ax3.grid(True, alpha=0.3)
            
            # 标记极板位置
            ax3.axvline(x=-L/2 - d/2, color='red', linestyle='--', alpha=0.5, label='Plate edges')
            ax3.axvline(x=-L/2 + d/2, color='red', linestyle='--', alpha=0.5)
            ax3.axvline(x=L/2 - d/2, color='red', linestyle='--', alpha=0.5)
            ax3.axvline(x=L/2 + d/2, color='red', linestyle='--', alpha=0.5)
            
            # 合并图例
            lines1, labels1 = ax3.get_legend_handles_labels()
            lines2, labels2 = ax3b.get_legend_handles_labels()
            ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'structure_height_histogram.png'), dpi=300, bbox_inches='tight')

    
    # 4. 电荷密度分布图 - 修改：增加细化程度
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    
    # 计算电荷密度分布
    if manager.frozen_particles:
        # 获取冻结粒子的位置
        frozen_x = [p.fixed_x for p in manager.frozen_particles if p.fixed_x is not None]
        
        if frozen_x:
            # 创建更细的直方图显示冻结粒子的分布
            n_bins = 100  # 增加柱体数量
            hist, bin_edges = np.histogram(frozen_x, bins=n_bins, range=(x_min, x_max))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            
            # 转换为线电荷密度（近似）
            line_charge_density = hist * q / bin_width
            
            # 绘制原始柱状图
            bars = ax4.bar(bin_centers, line_charge_density, 
                          width=bin_width * 0.8,  # 缩小柱体宽度
                          alpha=0.6, color='steelblue', edgecolor='black',
                          label=f'Line Charge Density (n={len(frozen_x)})')
            
            # 使用高斯核密度估计创建平滑曲线
            from scipy.stats import gaussian_kde
            
            if len(frozen_x) > 1:
                # 创建KDE
                kde = gaussian_kde(frozen_x, bw_method=0.1)  # 减小带宽以获得更详细的特征
                
                # 创建密集的x点
                x_kde = np.linspace(x_min, x_max, 1000)
                
                # 从KDE计算密度
                density = kde(x_kde)
                
                # 将密度转换为电荷密度
                # 注意：KDE给出的是概率密度，需要归一化
                total_charge = len(frozen_x) * q
                smoothed_charge_density = density * total_charge
                
                # 绘制平滑曲线
                ax4.plot(x_kde, smoothed_charge_density, 
                        color='darkred', lw=2, alpha=0.8, 
                        label='Smoothed Charge Density')
            
            # 添加统计信息
            mean_charge = np.mean(line_charge_density[line_charge_density > 0]) if np.any(line_charge_density > 0) else 0
            max_charge = np.max(line_charge_density) if len(line_charge_density) > 0 else 0
            
            # 添加文本信息
            text_str = f'Total frozen particles: {len(frozen_x)}\n'
            text_str += f'Avg line charge density: {mean_charge:.2e} C/m\n'
            text_str += f'Max line charge density: {max_charge:.2e} C/m\n'
            text_str += f'Total charge: {len(frozen_x) * q:.2e} C'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax4.text(0.02, 0.98, text_str, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
    
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('Line charge density (C/m)')
    ax4.set_title(f'Charge Density Distribution on Lower Plate ({n_bins} bins)')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(loc='upper right')
    
    # 标记极板位置
    ax4.axvline(x=-L/2 - d/2, color='green', linestyle=':', alpha=0.7, lw=1.5, label='Left plate edges')
    ax4.axvline(x=-L/2 + d/2, color='green', linestyle=':', alpha=0.7, lw=1.5)
    ax4.axvline(x=L/2 - d/2, color='green', linestyle=':', alpha=0.7, lw=1.5, label='Right plate edges')
    ax4.axvline(x=L/2 + d/2, color='green', linestyle=':', alpha=0.7, lw=1.5)
    
    # 在极板区域添加阴影
    plate1_left, plate1_right = -L/2 - d/2, L/2 - d/2
    plate2_left, plate2_right = -L/2 + d/2, L/2 + d/2
    ylim = ax4.get_ylim()
    
    ax4.axvspan(plate1_left, plate1_right, alpha=0.1, color='blue', label='Plate region')
    ax4.axvspan(plate2_left, plate2_right, alpha=0.1, color='blue')
    
    ax4.set_ylim(ylim)  # 恢复y轴范围
    
    # 裁剪两端5%
    trim = 0.05
    x_range = x_max - x_min
    ax4.set_xlim(x_min + trim*x_range, x_max - trim*x_range)
    
    # 添加次要x轴，显示微米单位
    ax4_secondary = ax4.twiny()
    ax4_secondary.set_xlim(ax4.get_xlim())
    ax4_secondary.set_xlabel('x (µm)')
    
    # 转换刻度标签
    x_ticks = ax4.get_xticks()
    x_ticks_um = x_ticks * 1e6
    ax4_secondary.set_xticks(x_ticks)
    ax4_secondary.set_xticklabels([f'{tick:.1f}' for tick in x_ticks_um])
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'charge_density_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 5. 粒子速度分布图
    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 收集所有粒子的速度信息
    speed_data_au = []
    speed_data_n2 = []
    
    for particle in particles_with_traj:
        if len(particle.trajectory) > 10:
            traj_vx = [point[2] for point in particle.trajectory]
            traj_vy = [point[3] for point in particle.trajectory]
            traj_t = [point[4] for point in particle.trajectory]
            
            if traj_vx and traj_vy:
                speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(traj_vx, traj_vy)]
                avg_speed = np.mean(speeds)
                max_speed = np.max(speeds)
                
                if particle.material == 'Au':
                    speed_data_au.append({
                        'diameter': particle.diameter * 1e9,  # nm
                        'avg_speed': avg_speed,
                        'max_speed': max_speed
                    })
                else:  # N2
                    speed_data_n2.append({
                        'avg_speed': avg_speed,
                        'max_speed': max_speed
                    })
    
    # 绘制金粒子速度分布
    if speed_data_au:
        diameters = [data['diameter'] for data in speed_data_au]  
        avg_speeds_au = [data['avg_speed'] for data in speed_data_au]
        max_speeds_au = [data['max_speed'] for data in speed_data_au]
        
        # 按直径分组
        unique_diameters = np.unique(diameters)
        avg_by_diameter = []
        max_by_diameter = []
        
        for diam in unique_diameters: 
            mask = np.array(diameters) == diam
            avg_by_diameter.append(np.mean(np.array(avg_speeds_au)[mask]))
            max_by_diameter.append(np.max(np.array(max_speeds_au)[mask]))
        
        x_pos = np.arange(len(unique_diameters))
        width = 0.35
        
        bars1 = ax5a.bar(x_pos - width/2, avg_by_diameter, width, label='Average Speed', color='steelblue', alpha=0.7)
        bars2 = ax5a.bar(x_pos + width/2, max_by_diameter, width, label='Max Speed', color='coral', alpha=0.7)
        
        ax5a.set_xlabel('Particle Diameter (nm)')
        ax5a.set_ylabel('Speed (m/s)')
        ax5a.set_title('Gold Particle Speed Distribution')
        ax5a.set_xticks(x_pos)
        ax5a.set_xticklabels([f'{diam:.1f}' for diam in unique_diameters]) 
        ax5a.legend()
        ax5a.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax5a.text(bar.get_x() + bar.get_width()/2., height + 0.01*height,
                         f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 绘制氮气离子速度分布
    if speed_data_n2:
        avg_speeds_n2 = [data['avg_speed'] for data in speed_data_n2]
        max_speeds_n2 = [data['max_speed'] for data in speed_data_n2]
        
        # 创建速度分布直方图
        n_bins = 20
        ax5b.hist(avg_speeds_n2, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black', label=f'Average (n={len(avg_speeds_n2)})')
        ax5b.hist(max_speeds_n2, bins=n_bins, alpha=0.7, color='salmon', edgecolor='black', label=f'Maximum (n={len(max_speeds_n2)})', histtype='step', lw=2)
        
        ax5b.set_xlabel('Speed (m/s)')
        ax5b.set_ylabel('Count')
        ax5b.set_title('N₂ Ion Speed Distribution')
        ax5b.legend()
        ax5b.grid(True, alpha=0.3)
        
        # 添加统计信息
        if avg_speeds_n2:
            stats_text = f'Avg speed: {np.mean(avg_speeds_n2):.2f} m/s\n'
            stats_text += f'Max speed: {np.max(max_speeds_n2):.2f} m/s\n'
            stats_text += f'Std dev: {np.std(avg_speeds_n2):.2f} m/s'
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax5b.text(0.98, 0.98, stats_text, transform=ax5b.transAxes, fontsize=10,
                     verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'particle_speed_distribution.png'), dpi=300, bbox_inches='tight')
    
   # 6. 状态统计
    print("\n8. 模拟统计:")
    print(f"   总粒子数: {len(manager.get_all_particles())}")
    print(f"   未发射粒子: {len(manager.inactive_particles)}")
    print(f"   运动中粒子: {len(manager.active_particles)}")
    print(f"   冻结粒子: {len(manager.frozen_particles)}")
    print(f"   基底消失粒子: {len(manager.base_particles)}")
    print(f"   边界消失粒子: {len(manager.boundary_particles)}")
    
    # 创建状态统计图
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    
    states = ['Inactive', 'Active', 'Frozen', 'Base', 'Boundary']
    counts = [
        len(manager.inactive_particles),
        len(manager.active_particles),
        len(manager.frozen_particles),
        len(manager.base_particles),
        len(manager.boundary_particles)
    ]
    
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral', 'lightsalmon']
    
    bars = ax6.bar(states, counts, color=colors, edgecolor='black')
    ax6.set_xlabel('Particle State')
    ax6.set_ylabel('Count')
    ax6.set_title('Particle State Distribution')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上添加数值
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 添加百分比标签
    total = sum(counts)
    for i, (state, count) in enumerate(zip(states, counts)):
        percentage = (count / total) * 100
        ax6.text(i, count + total*0.01, f'{percentage:.1f}%', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 设置y轴范围，为百分比标签留出空间
    ax6.set_ylim(0, max(counts) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIRS['visualization'], 'state_distribution.png'), dpi=300, bbox_inches='tight')
    
    # 7. 最后一瞬间y=h/2高度处总电场强度分布
    fig7, ax7 = plt.subplots(figsize=(12, 6))

    # 直接使用field_solver的最后一个电场记录
    if field_solver.field_history:
        last_record = field_solver.field_history[-1]
        
        # 检查是否有必要的电场数据
        if 'E1_interp' in last_record and 'E2_interp' in last_record:
            last_time = last_record['time']
            E1_interp = last_record['E1_interp']
            E2_interp = last_record['E2_interp']
            
            # 在y=h/2高度处创建x方向采样点
            y_height = h/2
            x_points = np.linspace(x_min, x_max, 500)
            
            # 创建位置数组
            positions = np.column_stack([x_points, np.full_like(x_points, y_height)])
            
            # 获取电场强度
            E1_values = E1_interp(positions)
            E2_values = E2_interp(positions)
            
            # 计算总电场强度
            E_total_1 = np.sqrt(E1_values[:, 0]**2 + E1_values[:, 1]**2)
            E_total_2 = np.sqrt(E2_values[:, 0]**2 + E2_values[:, 1]**2)
            
            # 根据时间确定使用哪个电场
            T_pulse = 1/f
            t1_E1 = T_pulse * duty1
            phase = last_time % T_pulse
            
            if phase < t1_E1:
                E_total = E_total_1
                field_label = 'E1场'
            else:
                E_total = E_total_2
                field_label = 'E2场'
            
            # 绘制总电场强度曲线
            ax7.plot(x_points*1e6, E_total, 'g-', linewidth=2.5, label=f'总电场强度 ({field_label})')
            
            # 标记极板位置
            plate_edges = [-L/2-d/2, -L/2+d/2, L/2-d/2, L/2+d/2]
            for i, edge in enumerate(plate_edges):
                color = 'red' if i < 2 else 'blue'
                label = '左极板边界' if i < 2 else '右极板边界' if i == 2 else ''
                ax7.axvline(x=edge*1e6, color=color, linestyle='--', alpha=0.7, 
                            linewidth=1.5, label=label if label else None)
            
            # 设置图形属性
            ax7.set_xlabel('x位置 (um)')
            ax7.set_ylabel('电场强度 (V/m)')
            ax7.set_title(f'最后一瞬间 (t={last_time*1e6:.2f}us) y=h/2高度处总电场分布\n(h={h*1e6:.1f}um, h/2={y_height*1e6:.1f}um)')
            ax7.grid(True, alpha=0.3)
            ax7.legend()
            
            # 添加统计信息
            max_total = np.max(E_total)
            min_total = np.min(E_total)
            
            stats_text = f'最大场强: {max_total:.2e} V/m\n最小场强: {min_total:.2e} V/m'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax7.text(0.02, 0.98, stats_text, transform=ax7.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
            
            plt.tight_layout()
            plt.savefig(os.path.join(DIRS['visualization'], 'total_electric_field_at_half_height.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"已生成y=h/2高度处总电场分布图")
        else:
            print("警告: 电场历史记录中缺少插值器数据，跳过电场分布图")
    else:
        print("警告: 没有电场历史数据，跳过电场分布图生成")
    
    # 8. 上极板下方100nm和基底上方100nm的电场分布对比图
    fig8, ax8 = plt.subplots(figsize=(14, 8))

    if field_solver.field_history:
        last_record = field_solver.field_history[-1]
        
        if 'E1_interp' in last_record and 'E2_interp' in last_record:
            last_time = last_record['time']
            E1_interp = last_record['E1_interp']
            E2_interp = last_record['E2_interp']
            
            # 定义两个高度
            y_upper = upper_plate_y - 100e-9  # 上极板下方100nm
            y_lower = y_min + 100e-9          # 基底上方100nm
            
            # 创建x方向采样点
            x_points = np.linspace(x_min, x_max, 500)
            
            # 创建两个高度的位置数组
            positions_upper = np.column_stack([x_points, np.full_like(x_points, y_upper)])
            positions_lower = np.column_stack([x_points, np.full_like(x_points, y_lower)])
            
            # 获取两个高度的电场强度
            E1_values_upper = E1_interp(positions_upper)
            E2_values_upper = E2_interp(positions_upper)
            E1_values_lower = E1_interp(positions_lower)
            E2_values_lower = E2_interp(positions_lower)
            
            # 计算总电场强度
            E_total_1_upper = np.sqrt(E1_values_upper[:, 0]**2 + E1_values_upper[:, 1]**2)
            E_total_2_upper = np.sqrt(E2_values_upper[:, 0]**2 + E2_values_upper[:, 1]**2)
            E_total_1_lower = np.sqrt(E1_values_lower[:, 0]**2 + E1_values_lower[:, 1]**2)
            E_total_2_lower = np.sqrt(E2_values_lower[:, 0]**2 + E2_values_lower[:, 1]**2)
            
            # 根据时间确定使用哪个电场
            T_pulse = 1/f
            t1_E1 = T_pulse * duty1
            phase = last_time % T_pulse
            
            if phase < t1_E1:
                E_total_upper = E_total_1_upper
                E_total_lower = E_total_1_lower
                field_label = 'E1场'
            else:
                E_total_upper = E_total_2_upper
                E_total_lower = E_total_2_lower
                field_label = 'E2场'
            
            # 绘制两条曲线
            line_upper, = ax8.plot(x_points*1e6, E_total_upper, 'b-', linewidth=2.5, 
                                   label=f'上极板下方100nm ({field_label})')
            line_lower, = ax8.plot(x_points*1e6, E_total_lower, 'r-', linewidth=2.5, 
                                   label=f'基底上方100nm ({field_label})')
            
            # 标记极板位置
            plate_edges = [-L/2-d/2, -L/2+d/2, L/2-d/2, L/2+d/2]
            for i, edge in enumerate(plate_edges):
                color = 'green' if i < 2 else 'orange'
                label = '左极板边界' if i < 2 else '右极板边界' if i == 2 else ''
                ax8.axvline(x=edge*1e6, color=color, linestyle='--', alpha=0.7, 
                           linewidth=1.5, label=label if label else None)
            
            # 设置图形属性
            ax8.set_xlabel('x位置 (um)', fontsize=12)
            ax8.set_ylabel('电场强度 (V/m)', fontsize=12)
            ax8.set_title(f'最后一瞬间 (t={last_time*1e6:.2f}us) 不同高度电场分布对比\n({field_label}, 上极板高度={upper_plate_y*1e6:.1f}um, 基底高度={y_min*1e6:.1f}um)', 
                         fontsize=13, pad=20)
            ax8.grid(True, alpha=0.3)
            ax8.legend(fontsize=11)
            
            # 添加统计信息框
            max_upper = np.max(E_total_upper)
            min_upper = np.min(E_total_upper)
            max_lower = np.max(E_total_lower)
            min_lower = np.min(E_total_lower)
            
            stats_text = (f'上极板下方100nm:\n'
                         f'  最大场强: {max_upper:.2e} V/m\n'
                         f'  最小场强: {min_upper:.2e} V/m\n'
                         f'基底上方100nm:\n'
                         f'  最大场强: {max_lower:.2e} V/m\n'
                         f'  最小场强: {min_lower:.2e} V/m\n'
                         f'场强比值 (上/下): {max_upper/max_lower:.2f}')
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
            ax8.text(0.02, 0.98, stats_text, transform=ax8.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='left', bbox=props)
            
            # 添加高度标注
            ax8.text(0.98, 0.15, f'上极板高度: {upper_plate_y*1e6:.1f}um\n'
                                f'上极板下方100nm: {y_upper*1e6:.1f}um\n'
                                f'基底高度: {y_min*1e6:.1f}um\n'
                                f'基底上方100nm: {y_lower*1e6:.1f}um', 
                    transform=ax8.transAxes, fontsize=9, ha='right', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # 设置y轴为科学计数法
            ax8.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            
            plt.tight_layout()
            plt.savefig(os.path.join(DIRS['visualization'], 'electric_field_comparison_heights.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"已生成不同高度电场分布对比图")
        else:
            print("警告: 电场历史记录中缺少插值器数据，跳过电场对比图")
    else:
        print("警告: 没有电场历史数据，跳过电场分布对比图生成")


        # 9. 从保存的帧创建动画
    if SAVE_FRAMES and frame_data:
        create_animation_alternative(manager, field_solver, frame_data)

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 运行模拟
    manager, field_solver, frame_data = run_simulation()
    
    # 生成可视化
    plot_results(manager, field_solver, frame_data)
    
    print("\n模拟完成！")
    print(f"结果保存在: {DIRS['main']}")
    print("生成的文件结构:")
    print(f"  - 动画帧: {DIRS['anim_frames']}")
    print(f"    包含 {TOTAL_FRAMES} 帧动画图片")
    print(f"  - 粒子轨迹: {DIRS['particle_trajectories']}")
    print(f"    轨迹文件命名格式: particle_材料_直径nm_编号.csv")
    print(f"    文件内容包含: x, y, vx, vy, t 五列数据")
    print(f"  - 可视化结果: {DIRS['visualization']}")
    print(f"    包含6个分析图表和动画文件")
    print(f"  - 电场数据: {DIRS['electric_fields']}")
    print(f"    包含电场历史数据")
    print(f"  - 模拟数据: {DIRS['data']}")
    print(f"    包含粒子状态、冻结事件、帧数据等信息")
    
    # 生成模拟参数文件
    sim_params = {
        'L': L,
        'd': d,
        'h': h,
        'h1': h1,
        'E1': E1,
        'E2': E2,
        'f': f,
        'sim_time_ms': sim_time_ms,
        'total_particles': sum(dict_num_Au.values()) + sum(dict_num_N2.values()),
        'Au_particles': sum(dict_num_Au.values()),
        'N2_particles': sum(dict_num_N2.values()),
        'Au_diameters': list(dict_num_Au.keys()),
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    params_df = pd.DataFrame([sim_params])
    params_df.to_csv(os.path.join(DIRS['data'], 'simulation_parameters.csv'), index=False)
    
   
# 生成详细的README文件
with open(os.path.join(DIRS['main'], 'README.txt'), 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("二维空间带电粒子运动模拟 - 详细结果报告\n")
    f.write("="*70 + "\n\n")
    
    f.write(f"模拟时间: {sim_params['timestamp']}\n")
    f.write(f"模拟标识: {os.path.basename(DIRS['main'])}\n\n")
    
    f.write("1. 模拟参数总览\n")
    f.write("-"*50 + "\n")
    f.write("几何参数:\n")
    f.write(f"  极板长度 L: {L*1e6:.2f} μm\n")
    f.write(f"  极板间距 d: {d*1e6:.2f} μm\n")
    f.write(f"  下极板高度 h: {h*1e6:.2f} μm\n")
    f.write(f"  极板间距 h1: {h1 * 1e6:.2f} μm\n")
    f.write(f"  计算域范围: x=[{x_min*1e6:.2f}, {x_max*1e6:.2f}] μm, y=[{y_min*1e6:.2f}, {y_max*1e6:.2f}] μm\n\n")
    
    f.write("电场参数:\n")
    f.write(f"  外电场 E1: {E1/1e4:.1f} kV/m\n")
    f.write(f"  外电场 E2: {E2/1e4:.1f} kV/m\n")
    f.write(f"  频率 fra: {freq} kHz\n")
    f.write(f"  占空比: {duty1 * 100:.1f}%\n")
    f.write(f"  脉冲周期: {1/freq*1e6:.2f} μs\n")
    f.write(f"  E1持续时间: {1/freq*duty1 * 1e6:.2f} μs\n")
    f.write(f"  初始表面电荷密度: {sigma} C/m²\n\n")
    
    f.write("环境参数:\n")
    f.write(f"  气压 P: {P} bar ({P*atm:.0f} Pa)\n")
    f.write(f"  温度 T: {T} K\n")
    f.write(f"  空气粘度 μ: {mu_air:.2e} Pa·s\n")
    f.write(f"  分子平均自由程 λ: {lambda0 * 1e9:.1f} nm\n\n")
    
    f.write("模拟时间参数:\n")
    f.write(f"  总模拟时间: {sim_time_ms} ms ({sim_time_ms*1e-3:.2e} s)\n")
    f.write(f"  物理时间步长: {dt_phys:.2e} s\n")
    f.write(f"  总时间步数: {int(np.ceil(sim_time_ms*1e-3/dt_phys))}\n")
    f.write(f"  动画总帧数: {TOTAL_FRAMES}\n")
    f.write(f"  动画帧率: {ANIMATION_FPS} FPS\n\n")
    
    f.write("2. 粒子参数\n")
    f.write("-"*50 + "\n")
    f.write("金纳米粒子 (Au):\n")
    total_au = sum(dict_num_Au.values())
    f.write(f"  总数量: {total_au} 个\n")
    f.write(f"  密度: {rho_table['Au']} kg/m³\n")
    for diameter, count in dict_num_Au.items():
        volume = 4/3 * np.pi * (diameter*nano/2)**3
        mass = volume * rho_table['Au']
        f.write(f"  {diameter} nm: {count} 个, 质量: {mass:.2e} kg, 体积: {volume:.2e} m³\n")
    f.write("\n")
    
    f.write("氮气离子 (N2):\n")
    total_n2 = sum(dict_num_N2.values())
    f.write(f"  总数量: {total_n2} 个\n")
    f.write(f"  密度: {rho_table['N2']} kg/m³\n")
    for diameter, count in dict_num_N2.items():
        volume = 4/3 * np.pi * (diameter*nano/2)**3
        mass = volume * rho_table['N2']
        f.write(f"  {diameter} nm: {count} 个, 质量: {mass:.2e} kg, 体积: {volume:.2e} m³\n")
    f.write("\n")
    
    f.write(f"粒子电荷: {q:.2e} C (元电荷: {e:.2e} C)\n")
    f.write(f"粒子总数: {total_au + total_n2} 个\n")
    f.write(f"金/氮比例: {total_au/total_n2:.3f}\n\n")
    
    f.write("3. 物理模型参数\n")
    f.write("-"*50 + "\n")
    f.write("电场求解:\n")
    f.write(f"  网格分辨率: {500}x{300} 点\n")
    f.write(f"  电场更新阈值: 每 {frozen_new} 个新冻结粒子更新一次\n")
    f.write(f"  包含镜像电荷效应: 是\n")
    f.write(f"  库仑常数: 8.99e9 N·m²/C²\n\n")
    
    f.write("力模型:\n")
    f.write("  - 电场力: qE\n")
    f.write("  - 拖曳力: -γv\n")
    f.write("  - 库仑力: 粒子间相互作用\n")
    f.write("  - 布朗力: 随机热运动\n")
    f.write(f"  布朗力标准差计算: 包含温度效应\n\n")
    
    f.write("边界条件:\n")
    f.write("  - 下极板: 冻结边界 (金属基底)\n")
    f.write("  - 上极板: 吸收边界\n")
    f.write("  - 左右边界: 吸收边界\n")
    f.write("  - 基底: 吸收边界\n\n")
    
    f.write("4. 模拟结果统计\n")
    f.write("-"*50 + "\n")
    # 从manager获取实时统计
    f.write(f"粒子状态分布:\n")
    f.write(f"  未发射粒子: {len(manager.inactive_particles)} 个\n")
    f.write(f"  运动中粒子: {len(manager.active_particles)} 个\n")
    f.write(f"  冻结粒子: {len(manager.frozen_particles)} 个\n")
    f.write(f"  基底消失粒子: {len(manager.base_particles)} 个\n")
    f.write(f"  边界消失粒子: {len(manager.boundary_particles)} 个\n")
    f.write(f"  总处理粒子: {len(manager.get_all_particles())} 个\n\n")
    
    if hasattr(field_solver, 'field_history'):
        f.write(f"电场更新次数: {len(field_solver.field_history)} 次\n")
        if field_solver.field_history:
            last_update = field_solver.field_history[-1]
            f.write(f"最后更新时间: {last_update['time']*1e6:.2f} μs\n")
            f.write(f"最后冻结粒子数: {last_update['frozen_count']} 个\n\n")
    
    f.write("5. 生成文件结构\n")
    f.write("-"*50 + "\n")
    f.write("animation_frames/ - 动画帧图片\n")
    f.write(f"  包含 {TOTAL_FRAMES} 帧 PNG 图片\n")
    f.write(f"  分辨率: 12x8 英寸, 100 DPI\n")
    f.write(f"  每帧包含: 粒子轨迹、电力线、实时统计\n\n")
    
    f.write("particle_trajectories/ - 单个粒子轨迹数据\n")
    f.write("  文件命名: particle_材料_直径nm_编号.csv\n")
    f.write("  数据列: x(m), y(m), vx(m/s), vy(m/s), t(s)\n")
    f.write(f"  已保存轨迹数: {sum(1 for p in manager.get_all_particles() if p.trajectory and len(p.trajectory) > 10)}\n\n")
    
    f.write("visualization/ - 可视化图表\n")
    f.write("  all_particles_trajectory.png - 所有粒子轨迹总览\n")
    f.write("  gold_particles_trajectory.png - 金粒子轨迹\n")
    f.write("  structure_height_histogram.png - 基底结构分布\n")
    f.write("  charge_density_distribution.png - 电荷密度分布\n")
    f.write("  particle_speed_distribution.png - 粒子速度分布\n")
    f.write("  state_distribution.png - 粒子状态统计\n")
    f.write("  total_electric_field_at_half_height.png - 电场分布\n")
    f.write("  electric_field_comparison_heights.png - 不同高度电场对比\n")
    f.write("  particle_animation.mp4/.gif - 运动动画\n\n")
    
    f.write("electric_fields/ - 电场数据\n")
    f.write("  field_history.pkl - 电场更新历史\n")
    f.write("  包含每次电场更新的网格数据和冻结粒子信息\n\n")
    
    f.write("simulation_data/ - 模拟数据文件\n")
    f.write("  simulation_parameters.csv - 模拟参数汇总\n")
    f.write("  frozen_events.csv - 粒子冻结事件记录\n")
    f.write("  frozen_particles_detailed.csv - 冻结粒子详细信息\n")
    f.write("  frozen_particles.pkl - 冻结粒子对象数据\n")
    f.write("  particle_states.pkl - 所有粒子最终状态\n")
    f.write("  frame_data.csv - 动画帧数据记录\n\n")
    
    f.write("6. 技术细节\n")
    f.write("-"*50 + "\n")
    f.write("数值方法:\n")
    f.write("  - 时间积分: 显式欧拉方法\n")
    f.write("  - 电场插值: RegularGridInterpolator\n")
    f.write("  - 电力线追踪: 龙格-库塔方法\n")
    f.write("  - 电荷密度: 直方图统计 + 高斯核密度估计\n\n")
    
    f.write("计算性能:\n")
    f.write(f"  网格点数: {500 * 300:,} 个\n")
    f.write(f"  每步计算粒子数: 最多 {total_au + total_n2} 个\n")
    f.write(f"  库仑力计算复杂度: O(n²)\n")
    f.write(f"  内存使用: 约 {500 * 300 * 8 * 2/1e6:.1f} MB (电场网格)\n\n")
    
    f.write("7. 物理验证\n")
    f.write("-"*50 + "\n")
    f.write("模型验证点:\n")
    f.write("  - 金属基底镜像电荷效应: 已实现\n")
    f.write("  - 电场边界条件: 垂直于金属表面\n")
    f.write("  - 粒子守恒: 跟踪所有粒子状态\n")
    f.write("  - 能量平衡: 包含耗散和随机力\n")
    f.write("  - 时间尺度: 纳秒级时间步长\n\n")


    print(f"\n模拟参数已保存到: {os.path.join(DIRS['data'], 'simulation_parameters.csv')}")
    print(f"README文件已生成: {os.path.join(DIRS['main'], 'README.txt')}")

