import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re
from typing import Dict, List, Tuple
from scipy.stats import gaussian_kde

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_simulation_parameters(sim_dir: str) -> Dict:
    """从模拟目录加载几何参数"""
    params = {
        'L': 4e-6,  # 极板长度
        'd': 6e-6,  # 极板总宽度
        'h': 1.5e-6,  # 下极板y坐标
        'h1': 4e-6,  # 上下极板间距
        'x_min': -4e-6,  # 计算域边界
        'x_max': 4e-6    # 计算域边界
    }
    
    # 尝试从参数文件加载实际参数
    params_file = os.path.join(sim_dir, "simulation_data", "simulation_parameters.csv")
    if os.path.exists(params_file):
        try:
            params_df = pd.read_csv(params_file)
            if not params_df.empty:
                for key in ['L', 'd', 'h', 'h1']:
                    if key in params_df.columns:
                        params[key] = float(params_df[key].iloc[0])
                print(f"从参数文件加载成功: L={params['L']*1e6:.1f}μm, d={params['d']*1e6:.1f}μm")
        except Exception as e:
            print(f"加载参数文件时出错: {e}, 使用默认参数")
    
    # 计算极板位置范围
    params['plate1_x_min'] = -params['L']/2 - params['d']/2  # 左极板左边界
    params['plate1_x_max'] = params['L']/2 - params['d']/2   # 左极板右边界
    params['plate2_x_min'] = -params['L']/2 + params['d']/2  # 右极板左边界
    params['plate2_x_max'] = params['L']/2 + params['d']/2   # 右极板右边界
    
    print(f"极板位置范围:")
    print(f"左极板: [{params['plate1_x_min']*1e6:.1f}, {params['plate1_x_max']*1e6:.1f}] μm")
    print(f"右极板: [{params['plate2_x_min']*1e6:.1f}, {params['plate2_x_max']*1e6:.1f}] μm")
    
    return params

def load_particle_data(sim_dir: str) -> Tuple[Dict, List, List, List, List]:
    """加载粒子轨迹数据，只统计极板范围内的数据"""
    trajectory_dir = os.path.join(sim_dir, "particle_trajectories")
    
    if not os.path.exists(trajectory_dir):
        print(f"错误: 轨迹目录不存在: {trajectory_dir}")
        return {}, [], [], [], []
    
    # 加载模拟参数
    params = load_simulation_parameters(sim_dir)
    
    # 获取所有轨迹文件
    trajectory_files = glob.glob(os.path.join(trajectory_dir, "*.csv"))
    print(f"找到 {len(trajectory_files)} 个轨迹文件")
    
    # 按材料分类存储数据
    particle_data = {
        'Au': {'diameters': [], 'survival_times': [], 'speeds': []},
        'N2': {'diameters': [], 'survival_times': [], 'speeds': []}
    }
    
    n2_y_positions = []  # 存储所有氮气分子的y坐标
    au_y_positions = []  # 存储所有金纳米粒子的y坐标
    n2_times = []       # 存储氮气分子对应的时间
    au_times = []       # 存储金纳米粒子对应的时间
    n2_x_positions = [] # 存储氮气分子的x坐标（用于调试）
    au_x_positions = [] # 存储金纳米粒子的x坐标（用于调试）
    
    total_points = 0
    filtered_points = 0
    
    for file_path in trajectory_files:
        try:
            # 从文件名解析粒子信息
            filename = os.path.basename(file_path)
            match = re.match(r'particle_([A-Za-z0-9]+)_([\d.]+)nm_(\d+)\.csv', filename)
            
            if not match:
                continue
                
            material = match.group(1)  # Au 或 N2
            diameter_nm = float(match.group(2))
            
            # 读取轨迹数据
            df = pd.read_csv(file_path)
            
            if len(df) < 2:  # 至少需要2个点才能计算
                continue
            
            total_points += len(df)
            
            # 筛选极板范围内的数据点
            x_positions = df['x'].values
            y_positions = df['y'].values
            times = df['t'].values
            
            # 判断点是否在极板范围内（左极板或右极板）
            in_plate_region = (
                ((x_positions >= params['plate1_x_min']) & (x_positions <= params['plate1_x_max'])) |
                ((x_positions >= params['plate2_x_min']) & (x_positions <= params['plate2_x_max']))
            )
            
            # 只保留极板范围内的数据
            filtered_indices = np.where(in_plate_region)[0]
            filtered_count = len(filtered_indices)
            filtered_points += filtered_count
            
            if filtered_count == 0:
                continue
            
            # 计算存活时间（使用原始轨迹数据）
            survival_time = times[-1] - times[0] if len(times) > 0 else 0
            
            # 计算平均速度（使用原始轨迹数据）
            speeds = np.sqrt(df['vx']**2 + df['vy']**2)
            avg_speed = np.mean(speeds) if len(speeds) > 0 else 0
            
            # 存储数据（使用原始统计，不筛选）
            if material in particle_data:
                particle_data[material]['diameters'].append(diameter_nm)
                particle_data[material]['survival_times'].append(survival_time)
                particle_data[material]['speeds'].append(avg_speed)
            
            # 收集极板范围内的y坐标和时间数据
            if material == 'N2':
                n2_y_positions.extend(y_positions[filtered_indices])
                n2_times.extend(times[filtered_indices])
                n2_x_positions.extend(x_positions[filtered_indices])
            elif material == 'Au':
                au_y_positions.extend(y_positions[filtered_indices])
                au_times.extend(times[filtered_indices])
                au_x_positions.extend(x_positions[filtered_indices])
                
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"数据筛选统计:")
    print(f"总数据点: {total_points}")
    print(f"极板范围内数据点: {filtered_points} ({filtered_points/total_points*100:.1f}%)")
    print(f"氮气分子有效点: {len(n2_y_positions)}")
    print(f"金纳米粒子有效点: {len(au_y_positions)}")
    
    # 调试信息：显示x坐标范围
    if n2_x_positions:
        print(f"氮气分子x范围: [{min(n2_x_positions)*1e6:.1f}, {max(n2_x_positions)*1e6:.1f}] μm")
    if au_x_positions:
        print(f"金纳米粒子x范围: [{min(au_x_positions)*1e6:.1f}, {max(au_x_positions)*1e6:.1f}] μm")
    
    return particle_data, n2_y_positions, n2_times, au_y_positions, au_times

def get_last_10_percent_data(y_positions: List, times: List) -> List:
    """获取最后10%时间内的数据"""
    if not times:
        return []
    
    # 找到最大时间
    max_time = max(times)
    cutoff_time = max_time * 0.9  # 最后10%的起始时间
    
    # 筛选最后10%时间内的数据
    last_10_percent_y = []
    for i, time in enumerate(times):
        if time >= cutoff_time:
            last_10_percent_y.append(y_positions[i])
    
    return last_10_percent_y

def analyze_survival_time(particle_data: Dict, output_dir: str):
    """分析粒子运动时间分布和平均时间"""
    print("\n=== 分析粒子运动时间分布 ===")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色设置
    colors = {'Au': '#FF6B6B', 'N2': '#118AB2'}
    
    for material, data in particle_data.items():
        if not data['survival_times']:
            continue
            
        survival_times = np.array(data['survival_times']) * 1e6  # 转换为微秒
        diameters = np.array(data['diameters'])
        
        print(f"\n{material} 粒子统计:")
        print(f"  粒子数量: {len(survival_times)}")
        print(f"  平均存活时间: {np.mean(survival_times):.2f} μs")
        print(f"  最短存活时间: {np.min(survival_times):.2f} μs")
        print(f"  最长存活时间: {np.max(survival_times):.2f} μs")
        print(f"  标准差: {np.std(survival_times):.2f} μs")
        
        # 1. 存活时间直方图
        ax1.hist(survival_times, bins=50, alpha=0.7, color=colors[material], 
                edgecolor='black', label=f'{material} (n={len(survival_times)})')
        
        # 2. 箱线图比较
        positions = list(particle_data.keys()).index(material) + 1
        box_data = ax2.boxplot(survival_times, positions=[positions], widths=0.6,
                              patch_artist=True, labels=[material])
        for patch in box_data['boxes']:
            patch.set_facecolor(colors[material])
            patch.set_alpha(0.7)
        
        # 3. 直径vs存活时间散点图
        if len(diameters) > 0:
            scatter = ax3.scatter(diameters, survival_times, alpha=0.6, 
                                 color=colors[material], label=material, s=30)
        
        # 4. 累积分布函数
        sorted_times = np.sort(survival_times)
        cdf = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
        ax4.plot(sorted_times, cdf, color=colors[material], linewidth=2, 
                label=f'{material} (n={len(survival_times)})')
    
    # 设置图表属性
    ax1.set_xlabel('存活时间 (μs)')
    ax1.set_ylabel('粒子数量')
    ax1.set_title('粒子存活时间分布直方图（极板范围内）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('存活时间 (μs)')
    ax2.set_title('粒子存活时间箱线图比较（极板范围内）')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlabel('粒子直径 (nm)')
    ax3.set_ylabel('存活时间 (μs)')
    ax3.set_title('直径与存活时间关系（极板范围内）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlabel('存活时间 (μs)')
    ax4.set_ylabel('累积概率')
    ax4.set_title('存活时间累积分布函数（极板范围内）')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_survival_time_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成详细统计表
    stats_file = os.path.join(output_dir, 'survival_time_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("粒子存活时间统计分析报告（极板范围内）\n")
        f.write("=" * 60 + "\n\n")
        
        for material, data in particle_data.items():
            if data['survival_times']:
                times = np.array(data['survival_times']) * 1e6
                diameters = np.array(data['diameters'])
                
                f.write(f"{material} 粒子统计:\n")
                f.write(f"  总粒子数: {len(times)}\n")
                f.write(f"  平均存活时间: {np.mean(times):.2f} μs\n")
                f.write(f"  中位数: {np.median(times):.2f} μs\n")
                f.write(f"  标准差: {np.std(times):.2f} μs\n")
                f.write(f"  最小值: {np.min(times):.2f} μs\n")
                f.write(f"  最大值: {np.max(times):.2f} μs\n")
                f.write(f"  25%分位数: {np.percentile(times, 25):.2f} μs\n")
                f.write(f"  75%分位数: {np.percentile(times, 75):.2f} μs\n")
                
                # 按直径分组统计
                unique_diameters = np.unique(diameters)
                if len(unique_diameters) > 1:
                    f.write(f"\n  按直径分组统计:\n")
                    for diam in unique_diameters:
                        mask = diameters == diam
                        diam_times = times[mask]
                        if len(diam_times) > 0:
                            f.write(f"    {diam}nm: {len(diam_times)}个粒子, "
                                  f"平均{np.mean(diam_times):.2f}μs\n")
                f.write("\n" + "-" * 30 + "\n\n")

def analyze_n2_distribution(n2_y_positions: List, n2_times: List, output_dir: str):
    """单独分析氮气分子(N2)在y轴上的分布（最后10%时间内，极板范围内）"""
    print("\n=== 分析氮气分子(N2) y轴分布（最后10%时间内，极板范围内）===")
    
    # 获取最后10%时间内的数据
    n2_y_last_10 = get_last_10_percent_data(n2_y_positions, n2_times)
    
    print(f"氮气分子 - 极板范围内数据点: {len(n2_y_positions)}, 最后10%数据点: {len(n2_y_last_10)}")
    
    if not n2_y_last_10:
        print("没有最后10%时间内的氮气分子数据")
        return
    
    # 创建氮气分子专用分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色设置
    color = '#118AB2'  # 氮气分子专用蓝色
    
    y_positions = np.array(n2_y_last_10)
    
    print(f"\n氮气分子(N2) 最后10%时间分布统计（极板范围内）:")
    print(f"  y坐标范围: {y_positions.min()*1e6:.2f} - {y_positions.max()*1e6:.2f} μm")
    print(f"  平均值: {np.mean(y_positions)*1e6:.2f} μm")
    print(f"  标准差: {np.std(y_positions)*1e6:.2f} μm")
    
    # 1. 氮气分子直方图
    n_bins = 150
    hist, bin_edges = np.histogram(y_positions, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    density = hist / (bin_width * 1e6 * len(y_positions))
    
    ax1.bar(bin_centers * 1e6, density, width=bin_width * 1e6 * 0.8,
           alpha=0.7, color=color, edgecolor='black',
           label=f'N2 (n={len(y_positions)})')
    
    # 添加统计线
    mean_y = np.mean(y_positions) * 1e6
    std_y = np.std(y_positions) * 1e6
    ax1.axvline(mean_y, color='red', linestyle='--', linewidth=2, 
               label=f'平均值: {mean_y:.2f} μm')
    ax1.axvline(mean_y - std_y, color='orange', linestyle='--', linewidth=1, 
               label=f'-1σ: {mean_y-std_y:.2f} μm')
    ax1.axvline(mean_y + std_y, color='orange', linestyle='--', linewidth=1, 
               label=f'+1σ: {mean_y+std_y:.2f} μm')
    
    # 设置第一个图表属性
    ax1.set_xlabel('y 位置 (μm)')
    ax1.set_ylabel('相对密度')
    ax1.set_title('氮气分子(N2) y方向密度分布（最后10%时间内，极板范围内）')
    ax1.set_ylim(0, 1.5)
    ax1.set_xlim(1, 5.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 概率密度分布
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(y_positions)
    y_range = np.linspace(y_positions.min(), y_positions.max(), 200)
    pdf = kde(y_range)
    ax2.plot(y_range * 1e6, pdf, color=color, linewidth=2, 
            label=f'N2 (n={len(y_positions)})', alpha=0.8)
    
    # 添加统计线到PDF图
    ax2.axvline(mean_y, color='red', linestyle='--', linewidth=2, 
               label=f'平均值: {mean_y:.2f} μm')
    ax2.axvline(mean_y - std_y, color='orange', linestyle='--', linewidth=1)
    ax2.axvline(mean_y + std_y, color='orange', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('y 位置 (μm)')
    ax2.set_ylabel('概率密度')
    ax2.set_title('氮气分子(N2) 概率密度分布（最后10%时间内，极板范围内）')
    ax2.set_ylim(0, 12e5)
    ax2.set_xlim(1, 5.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    y_sorted = np.sort(y_positions)
    cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    ax3.plot(y_sorted * 1e6, cdf, color=color, linewidth=2, 
            label=f'N2 (n={len(y_positions)})')
    
    # 添加统计线到CDF图
    median_y = np.median(y_positions) * 1e6
    ax3.axvline(median_y, color='green', linestyle='--', linewidth=2, 
               label=f'中位数: {median_y:.2f} μm')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('y 位置 (μm)')
    ax3.set_ylabel('累积概率')
    ax3.set_title('氮气分子(N2) 累积分布函数（最后10%时间内，极板范围内）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图
    box_data = ax4.boxplot(y_positions * 1e6, patch_artist=True)
    box_data['boxes'][0].set_facecolor(color)
    box_data['boxes'][0].set_alpha(0.7)
    
    # 添加统计信息到箱线图
    q1 = np.percentile(y_positions, 25) * 1e6
    q3 = np.percentile(y_positions, 75) * 1e6
    median_val = np.median(y_positions) * 1e6
    ax4.text(1.1, median_val, f'中位数: {median_val:.2f}', va='center', fontsize=10)
    ax4.text(1.1, q1, f'Q1: {q1:.2f}', va='center', fontsize=10)
    ax4.text(1.1, q3, f'Q3: {q3:.2f}', va='center', fontsize=10)
    
    ax4.set_ylabel('y 位置 (μm)')
    ax4.set_title('氮气分子(N2) y位置分布箱线图（最后10%时间内，极板范围内）')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'n2_y_distribution_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成氮气分子分布统计报告
    stats_file = os.path.join(output_dir, 'n2_distribution_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("氮气分子(N2) y方向分布统计报告（最后10%时间内，极板范围内）\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("氮气分子 (N2) 统计:\n")
        f.write(f"  数据点数: {len(y_positions)}\n")
        f.write(f"  y坐标范围: {y_positions.min()*1e6:.3f} - {y_positions.max()*1e6:.3f} μm\n")
        f.write(f"  平均值: {np.mean(y_positions)*1e6:.3f} μm\n")
        f.write(f"  中位数: {np.median(y_positions)*1e6:.3f} μm\n")
        f.write(f"  标准差: {np.std(y_positions)*1e6:.3f} μm\n")
        
        f.write("  关键百分位数值:\n")
        for percentile in [10, 25, 50, 75, 90]:
            value = np.percentile(y_positions, percentile) * 1e6
            f.write(f"    {percentile}%: {value:.3f} μm\n")
        
        f.write(f"\n  分布特征:\n")
        f.write(f"  偏度: {((np.mean(y_positions) - np.median(y_positions))/np.std(y_positions)):.3f}\n")
        f.write(f"  峰度: {np.mean(((y_positions - np.mean(y_positions))/np.std(y_positions))**4) - 3:.3f}\n")

def analyze_au_distribution(au_y_positions: List, au_times: List, output_dir: str):
    """单独分析金纳米粒子(Au)在y轴上的分布（最后10%时间内，极板范围内）"""
    print("\n=== 分析金纳米粒子(Au) y轴分布（最后10%时间内，极板范围内）===")
    
    # 获取最后10%时间内的数据
    au_y_last_10 = get_last_10_percent_data(au_y_positions, au_times)
    
    print(f"金纳米粒子 - 极板范围内数据点: {len(au_y_positions)}, 最后10%数据点: {len(au_y_last_10)}")
    
    if not au_y_last_10:
        print("没有最后10%时间内的金纳米粒子数据")
        return
    
    # 创建金纳米粒子专用分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色设置
    color = '#FF6B6B'  # 金纳米粒子专用红色
    
    y_positions = np.array(au_y_last_10)
    
    print(f"\n金纳米粒子(Au) 最后10%时间分布统计（极板范围内）:")
    print(f"  y坐标范围: {y_positions.min()*1e6:.2f} - {y_positions.max()*1e6:.2f} μm")
    print(f"  平均值: {np.mean(y_positions)*1e6:.2f} μm")
    print(f"  标准差: {np.std(y_positions)*1e6:.2f} μm")
    
    # 1. 金纳米粒子直方图
    n_bins = 150
    hist, bin_edges = np.histogram(y_positions, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    density = hist / (bin_width * 1e6 * len(y_positions))
    
    ax1.bar(bin_centers * 1e6, density, width=bin_width * 1e6 * 0.8,
           alpha=0.7, color=color, edgecolor='black',
           label=f'Au (n={len(y_positions)})')
    
    # 添加统计线
    mean_y = np.mean(y_positions) * 1e6
    std_y = np.std(y_positions) * 1e6
    ax1.axvline(mean_y, color='red', linestyle='--', linewidth=2, 
               label=f'平均值: {mean_y:.2f} μm')
    ax1.axvline(mean_y - std_y, color='orange', linestyle='--', linewidth=1, 
               label=f'-1σ: {mean_y-std_y:.2f} μm')
    ax1.axvline(mean_y + std_y, color='orange', linestyle='--', linewidth=1, 
               label=f'+1σ: {mean_y+std_y:.2f} μm')
    
    # 设置第一个图表属性
    ax1.set_xlabel('y 位置 (μm)')
    ax1.set_ylabel('相对密度')
    ax1.set_title('金纳米粒子(Au) y方向密度分布（最后10%时间内，极板范围内）')
    ax1.set_ylim(0, 35)
    ax1.set_xlim(1, 5.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 概率密度分布
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(y_positions)
    y_range = np.linspace(y_positions.min(), y_positions.max(), 200)
    pdf = kde(y_range)
    ax2.plot(y_range * 1e6, pdf, color=color, linewidth=2, 
            label=f'Au (n={len(y_positions)})', alpha=0.8)
    
    # 添加统计线到PDF图
    ax2.axvline(mean_y, color='red', linestyle='--', linewidth=2, 
               label=f'平均值: {mean_y:.2f} μm')
    ax2.axvline(mean_y - std_y, color='orange', linestyle='--', linewidth=1)
    ax2.axvline(mean_y + std_y, color='orange', linestyle='--', linewidth=1)
    
    ax2.set_xlabel('y 位置 (μm)')
    ax2.set_ylabel('概率密度')
    ax2.set_title('金纳米粒子(Au) 概率密度分布（最后10%时间内，极板范围内）')
    ax2.set_ylim(0, 10e6)
    ax2.set_xlim(1, 5.5) 
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布函数
    y_sorted = np.sort(y_positions)
    cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    ax3.plot(y_sorted * 1e6, cdf, color=color, linewidth=2, 
            label=f'Au (n={len(y_positions)})')
    
    # 添加统计线到CDF图
    median_y = np.median(y_positions) * 1e6
    ax3.axvline(median_y, color='green', linestyle='--', linewidth=2, 
               label=f'中位数: {median_y:.2f} μm')
    ax3.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    
    ax3.set_xlabel('y 位置 (μm)')
    ax3.set_ylabel('累积概率')
    ax3.set_title('金纳米粒子(Au) 累积分布函数（最后10%时间内，极板范围内）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图
    box_data = ax4.boxplot(y_positions * 1e6, patch_artist=True)
    box_data['boxes'][0].set_facecolor(color)
    box_data['boxes'][0].set_alpha(0.7)
    
    # 添加统计信息到箱线图
    q1 = np.percentile(y_positions, 25) * 1e6
    q3 = np.percentile(y_positions, 75) * 1e6
    median_val = np.median(y_positions) * 1e6
    ax4.text(1.1, median_val, f'中位数: {median_val:.2f}', va='center', fontsize=10)
    ax4.text(1.1, q1, f'Q1: {q1:.2f}', va='center', fontsize=10)
    ax4.text(1.1, q3, f'Q3: {q3:.2f}', va='center', fontsize=10)
    
    ax4.set_ylabel('y 位置 (μm)')
    ax4.set_title('金纳米粒子(Au) y位置分布箱线图（最后10%时间内，极板范围内）')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'au_y_distribution_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成金纳米粒子分布统计报告
    stats_file = os.path.join(output_dir, 'au_distribution_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("金纳米粒子(Au) y方向分布统计报告（最后10%时间内，极板范围内）\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("金纳米粒子 (Au) 统计:\n")
        f.write(f"  数据点数: {len(y_positions)}\n")
        f.write(f"  y坐标范围: {y_positions.min()*1e6:.3f} - {y_positions.max()*1e6:.3f} μm\n")
        f.write(f"  平均值: {np.mean(y_positions)*1e6:.3f} μm\n")
        f.write(f"  中位数: {np.median(y_positions)*1e6:.3f} μm\n")
        f.write(f"  标准差: {np.std(y_positions)*1e6:.3f} μm\n")
        
        f.write("  关键百分位数值:\n")
        for percentile in [10, 25, 50, 75, 90]:
            value = np.percentile(y_positions, percentile) * 1e6
            f.write(f"    {percentile}%: {value:.3f} μm\n")
        
        f.write(f"\n  分布特征:\n")
        f.write(f"  偏度: {((np.mean(y_positions) - np.median(y_positions))/np.std(y_positions)):.3f}\n")
        f.write(f"  峰度: {np.mean(((y_positions - np.mean(y_positions))/np.std(y_positions))**4) - 3:.3f}\n")

def analyze_y_distribution(n2_y_positions: List, n2_times: List, 
                          au_y_positions: List, au_times: List, output_dir: str):
    """分析粒子在y轴上的分布（最后10%时间内，极板范围内）- 原有的对比分析"""
    print("\n=== 分析粒子y轴分布（最后10%时间内，极板范围内）- 对比分析===")
    
    # 获取最后10%时间内的数据
    n2_y_last_10 = get_last_10_percent_data(n2_y_positions, n2_times)
    au_y_last_10 = get_last_10_percent_data(au_y_positions, au_times)
    
    print(f"氮气分子 - 极板范围内数据点: {len(n2_y_positions)}, 最后10%数据点: {len(n2_y_last_10)}")
    print(f"金纳米粒子 - 极板范围内数据点: {len(au_y_positions)}, 最后10%数据点: {len(au_y_last_10)}")
    
    if not n2_y_last_10 and not au_y_last_10:
        print("没有最后10%时间内的数据")
        return
    
    # 创建对比分析图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 颜色设置
    colors = {'Au': '#FF6B6B', 'N2': '#118AB2'}
    
    # 分析氮气分子
    if n2_y_last_10:
        y_positions = np.array(n2_y_last_10)
        material = 'N2'
        
        print(f"\n{material} 最后10%时间分布统计（极板范围内）:")
        print(f"  y坐标范围: {y_positions.min()*1e6:.2f} - {y_positions.max()*1e6:.2f} μm")
        print(f"  平均值: {np.mean(y_positions)*1e6:.2f} μm")
        print(f"  标准差: {np.std(y_positions)*1e6:.2f} μm")
        
        # 1. 氮气分子直方图
        n_bins = 150
        hist, bin_edges = np.histogram(y_positions, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        density = hist / (bin_width * 1e6 * len(y_positions))
        
        ax1.bar(bin_centers * 1e6, density, width=bin_width * 1e6 * 0.8,
               alpha=0.7, color=colors[material], edgecolor='black',
               label=f'{material} (n={len(y_positions)})')
        
        # 添加统计线
        mean_y = np.mean(y_positions) * 1e6
        ax1.axvline(mean_y, color='red', linestyle='--', linewidth=2, 
                   label=f'平均值: {mean_y:.2f} μm')
    
    # 分析金纳米粒子
    if au_y_last_10:
        y_positions = np.array(au_y_last_10)
        material = 'Au'
        
        print(f"\n{material} 最后10%时间分布统计（极板范围内）:")
        print(f"  y坐标范围: {y_positions.min()*1e6:.2f} - {y_positions.max()*1e6:.2f} μm")
        print(f"  平均值: {np.mean(y_positions)*1e6:.2f} μm")
        print(f"  标准差: {np.std(y_positions)*1e6:.2f} μm")
        
        # 1. 金纳米粒子直方图（与氮气分子在同一图中）
        n_bins = 150
        hist, bin_edges = np.histogram(y_positions, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        density = hist / (bin_width * 1e6 * len(y_positions))
        
        ax1.bar(bin_centers * 1e6, density, width=bin_width * 1e6 * 0.8,
               alpha=0.7, color=colors[material], edgecolor='black',
               label=f'{material} (n={len(y_positions)})')
        
        # 添加统计线
        mean_y = np.mean(y_positions) * 1e6
        ax1.axvline(mean_y, color='orange', linestyle='--', linewidth=2, 
                   label=f'平均值: {mean_y:.2f} μm')
    
    # 设置第一个图表属性
    ax1.set_xlabel('y 位置 (μm)')
    ax1.set_ylabel('相对密度')
    ax1.set_title('粒子y方向密度分布对比（最后10%时间内，极板范围内）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 概率密度分布对比
    if n2_y_last_10:
        y_positions = np.array(n2_y_last_10)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y_positions)
        y_range = np.linspace(min(n2_y_last_10 + au_y_last_10) if au_y_last_10 else min(n2_y_last_10), 
                             max(n2_y_last_10 + au_y_last_10) if au_y_last_10 else max(n2_y_last_10), 200)
        pdf = kde(y_range)
        ax2.plot(y_range * 1e6, pdf, color=colors['N2'], linewidth=2, 
                label=f'N2 (n={len(n2_y_last_10)})', alpha=0.8)
    
    if au_y_last_10:
        y_positions = np.array(au_y_last_10)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(y_positions)
        y_range = np.linspace(min(au_y_last_10 + n2_y_last_10) if n2_y_last_10 else min(au_y_last_10), 
                             max(au_y_last_10 + n2_y_last_10) if n2_y_last_10 else max(au_y_last_10), 200)
        pdf = kde(y_range)
        ax2.plot(y_range * 1e6, pdf, color=colors['Au'], linewidth=2, 
                label=f'Au (n={len(au_y_last_10)})', alpha=0.8)
    
    ax2.set_xlabel('y 位置 (μm)')
    ax2.set_ylabel('概率密度')
    ax2.set_title('概率密度分布对比（最后10%时间内，极板范围内）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 累积分布函数对比
    if n2_y_last_10:
        y_sorted = np.sort(n2_y_last_10)
        cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        ax3.plot(y_sorted * 1e6, cdf, color=colors['N2'], linewidth=2, 
                label=f'N2 (n={len(n2_y_last_10)})')
    
    if au_y_last_10:
        y_sorted = np.sort(au_y_last_10)
        cdf = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
        ax3.plot(y_sorted * 1e6, cdf, color=colors['Au'], linewidth=2, 
                label=f'Au (n={len(au_y_last_10)})')
    
    ax3.set_xlabel('y 位置 (μm)')
    ax3.set_ylabel('累积概率')
    ax3.set_title('累积分布函数对比（最后10%时间内，极板范围内）')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 箱线图对比
    plot_data = []
    labels = []
    if n2_y_last_10:
        plot_data.append(np.array(n2_y_last_10) * 1e6)
        labels.append('N2')
    if au_y_last_10:
        plot_data.append(np.array(au_y_last_10) * 1e6)
        labels.append('Au')
    
    if plot_data:
        box_data = ax4.boxplot(plot_data, labels=labels, patch_artist=True)
        colors_list = [colors[label] for label in labels]
        for patch, color in zip(box_data['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('y 位置 (μm)')
        ax4.set_title('y位置分布箱线图对比（最后10%时间内，极板范围内）')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'particle_y_distribution_last_10percent.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 生成分布统计报告
    stats_file = os.path.join(output_dir, 'y_distribution_last_10percent_statistics.txt')
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("粒子y方向分布统计报告（最后10%时间内，极板范围内）\n")
        f.write("=" * 70 + "\n\n")
        
        # 氮气分子统计
        if n2_y_last_10:
            y_positions = np.array(n2_y_last_10)
            f.write("氮气分子 (N2) 统计:\n")
            f.write(f"  数据点数: {len(y_positions)}\n")
            f.write(f"  y坐标范围: {y_positions.min()*1e6:.3f} - {y_positions.max()*1e6:.3f} μm\n")
            f.write(f"  平均值: {np.mean(y_positions)*1e6:.3f} μm\n")
            f.write(f"  中位数: {np.median(y_positions)*1e6:.3f} μm\n")
            f.write(f"  标准差: {np.std(y_positions)*1e6:.3f} μm\n")
            
            f.write("  关键百分位数值:\n")
            for percentile in [10, 25, 50, 75, 90]:
                value = np.percentile(y_positions, percentile) * 1e6
                f.write(f"    {percentile}%: {value:.3f} μm\n")
            f.write("\n")
        
        # 金纳米粒子统计
        if au_y_last_10:
            y_positions = np.array(au_y_last_10)
            f.write("金纳米粒子 (Au) 统计:\n")
            f.write(f"  数据点数: {len(y_positions)}\n")
            f.write(f"  y坐标范围: {y_positions.min()*1e6:.3f} - {y_positions.max()*1e6:.3f} μm\n")
            f.write(f"  平均值: {np.mean(y_positions)*1e6:.3f} μm\n")
            f.write(f"  中位数: {np.median(y_positions)*1e6:.3f} μm\n")
            f.write(f"  标准差: {np.std(y_positions)*1e6:.3f} μm\n")
            
            f.write("  关键百分位数值:\n")
            for percentile in [10, 25, 50, 75, 90]:
                value = np.percentile(y_positions, percentile) * 1e6
                f.write(f"    {percentile}%: {value:.3f} μm\n")
            f.write("\n")
        
        # 对比统计
        if n2_y_last_10 and au_y_last_10:
            n2_y = np.array(n2_y_last_10)
            au_y = np.array(au_y_last_10)
            
            f.write("对比分析:\n")
            f.write(f"  平均位置差: {np.mean(au_y)*1e6 - np.mean(n2_y)*1e6:.3f} μm\n")
            f.write(f"  中位数差: {np.median(au_y)*1e6 - np.median(n2_y)*1e6:.3f} μm\n")
            f.write(f"  分布重叠度分析:\n")
            
            # 计算分布重叠区域
            n2_min, n2_max = np.min(n2_y), np.max(n2_y)
            au_min, au_max = np.min(au_y), np.max(au_y)
            overlap_min = max(n2_min, au_min)
            overlap_max = min(n2_max, au_max)
            
            if overlap_min < overlap_max:
                overlap_ratio = (overlap_max - overlap_min) / (max(n2_max, au_max) - min(n2_min, au_min))
                f.write(f"    重叠区域: {overlap_min*1e6:.3f} - {overlap_max*1e6:.3f} μm\n")
                f.write(f"    重叠比例: {overlap_ratio*100:.1f}%\n")
            else:
                f.write("    分布无重叠\n")

def main():
    """主函数"""
    # 在这里指定你的模拟目录
    sim_dir = "sim_3200particles_0.4ms_P1.00bar_T300.0K_E1.0kVcm_1.5e-06PR_20260108_114012"  # 修改为你的目录名
    
    if not os.path.exists(sim_dir):
        print(f"错误: 目录不存在: {sim_dir}")
        return
    
    # 创建输出目录
    output_dir = os.path.join(sim_dir, "simple_analysis")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始粒子运动分析")
    print("=" * 60)
    print(f"模拟目录: {sim_dir}")
    print(f"输出目录: {output_dir}")
    
    # 1. 加载数据
    print("\n1. 加载粒子轨迹数据...")
    particle_data, n2_y_positions, n2_times, au_y_positions, au_times = load_particle_data(sim_dir)
    
    if not particle_data and not n2_y_positions and not au_y_positions:
        print("没有找到有效数据")
        return
    
    # 2. 分析粒子运动时间
    print("\n2. 分析粒子运动时间分布...")
    analyze_survival_time(particle_data, output_dir)
    
    # 3. 单独分析氮气分子分布
    print("\n3. 单独分析氮气分子(N2) y轴分布...")
    analyze_n2_distribution(n2_y_positions, n2_times, output_dir)
    
    # 4. 单独分析金纳米粒子分布
    print("\n4. 单独分析金纳米粒子(Au) y轴分布...")
    analyze_au_distribution(au_y_positions, au_times, output_dir)
    
    # 5. 分析粒子y轴分布对比（原有的对比分析）
    print("\n5. 分析粒子y轴分布对比...")
    analyze_y_distribution(n2_y_positions, n2_times, au_y_positions, au_times, output_dir)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    
    # 显示生成的文件
    output_files = glob.glob(os.path.join(output_dir, "*"))
    print(f"\n生成的文件 ({len(output_files)} 个):")
    for file_path in output_files:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / 1024
        print(f"  - {file_name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()