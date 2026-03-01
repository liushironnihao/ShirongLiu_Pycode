"""
电荷密度后处理分析模块
用于分析模拟结果中的表面电荷密度随时间变化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import glob
from tqdm import tqdm
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# ===================== 配置参数 =====================
# 元电荷
e = 1.602176634e-19  # C

# 分析参数
SMOOTH_WINDOW = 5  # 平滑窗口大小
TIME_RESOLUTION = 1000  # 时间分辨率（时间点数量）

# ===================== 辅助函数 =====================
def load_simulation_data(sim_dir):
    """加载模拟数据 - 从指定目录直接加载"""
    if not os.path.exists(sim_dir):
        print(f"错误: 目录不存在: {sim_dir}")
        return None
    
    data = {}
    
    # 1. 加载冻结事件
    frozen_events_file = os.path.join(sim_dir, 'simulation_data', 'frozen_events.csv')
    if os.path.exists(frozen_events_file):
        data['frozen_events'] = pd.read_csv(frozen_events_file)
        print(f"加载冻结事件: {len(data['frozen_events'])} 条记录")
    else:
        print("警告: 未找到冻结事件文件")
        data['frozen_events'] = None
    
    # 2. 加载粒子状态
    states_file = os.path.join(sim_dir, 'simulation_data', 'particle_states.pkl')
    if os.path.exists(states_file):
        with open(states_file, 'rb') as f:
            data['particle_states'] = pickle.load(f)
        print("加载粒子状态数据")
    else:
        data['particle_states'] = None
    
    # 3. 加载电场历史
    field_history_file = os.path.join(sim_dir, 'electric_fields', 'field_history.pkl')
    if os.path.exists(field_history_file):
        with open(field_history_file, 'rb') as f:
            data['field_history'] = pickle.load(f)
        print(f"加载电场历史: {len(data['field_history'])} 次更新")
    else:
        data['field_history'] = None
    
    # 4. 加载模拟参数
    params_file = os.path.join(sim_dir, 'simulation_data', 'simulation_parameters.csv')
    if os.path.exists(params_file):
        data['params'] = pd.read_csv(params_file).iloc[0].to_dict()
        print("加载模拟参数")
    else:
        data['params'] = {}
    
    # 5. 收集所有轨迹文件
    trajectories_dir = os.path.join(sim_dir, 'particle_trajectories')
    traj_files = glob.glob(os.path.join(trajectories_dir, 'particle_*.csv'))
    if traj_files:
        data['trajectory_files'] = traj_files
        print(f"找到 {len(traj_files)} 个轨迹文件")
    else:
        data['trajectory_files'] = []
    
    return data

# ===================== 电荷密度分析函数 =====================
def analyze_charge_density_evolution(frozen_events, params):
    """分析电荷密度随时间变化"""
    if frozen_events is None or len(frozen_events) == 0:
        print("错误: 无冻结事件数据")
        return None
    
    # 按时间排序
    frozen_events = frozen_events.sort_values('time')
    
    # 创建时间序列
    max_time = frozen_events['time'].max()
    time_points = np.linspace(0, max_time, TIME_RESOLUTION)
    
    # 初始化结果数组
    cumulative_charge = np.zeros_like(time_points, dtype=int)  # 累积电荷数
    cumulative_particles = np.zeros_like(time_points, dtype=int)  # 累积粒子数
    charge_rate = np.zeros_like(time_points)  # 电荷沉积速率
    particle_rate = np.zeros_like(time_points)  # 粒子沉积速率
    
    # 计算每个时间点的累积值
    for i, t in enumerate(tqdm(time_points, desc="计算电荷密度演化")):
        mask = frozen_events['time'] <= t
        cumulative_particles[i] = mask.sum()
        cumulative_charge[i] = cumulative_particles[i]  # 每个粒子带一个元电荷
    
    # 计算沉积速率（导数）
    if len(time_points) > 1:
        dt = time_points[1] - time_points[0]
        charge_rate = np.gradient(cumulative_charge, dt)
        particle_rate = np.gradient(cumulative_particles, dt)
        
        # 平滑处理
        charge_rate = pd.Series(charge_rate).rolling(SMOOTH_WINDOW, center=True).mean().values
        particle_rate = pd.Series(particle_rate).rolling(SMOOTH_WINDOW, center=True).mean().values
    
    # 计算统计信息
    if len(frozen_events) > 0:
        total_time = max_time
        avg_rate = len(frozen_events) / total_time if total_time > 0 else 0
    else:
        total_time = 0
        avg_rate = 0
    
    # 计算线电荷密度（需要知道极板长度）
    L = params.get('L', 4e-6)  # 默认4µm
    plate_length = L  # 简化处理
    
    # 线电荷密度 = 总电荷 / 极板长度
    line_charge_density = cumulative_charge * e / plate_length  # C/m
    avg_line_charge_density = np.trapz(line_charge_density, time_points) / total_time if total_time > 0 else 0
    
    results = {
        'time_points': time_points,
        'cumulative_particles': cumulative_particles,
        'cumulative_charge': cumulative_charge,
        'line_charge_density': line_charge_density,
        'charge_rate': charge_rate,
        'particle_rate': particle_rate,
        'total_frozen': len(frozen_events),
        'total_time': total_time,
        'avg_deposition_rate': avg_rate,
        'avg_line_charge_density': avg_line_charge_density,
        'final_line_charge_density': line_charge_density[-1] if len(line_charge_density) > 0 else 0
    }
    
    return results

def analyze_spatial_distribution(frozen_events, params):
    """分析电荷的空间分布"""
    if frozen_events is None or len(frozen_events) == 0:
        return None
    
    # 获取冻结粒子的x坐标
    frozen_x = frozen_events['fixed_x'].values
    
    # 计算基本统计
    x_mean = np.mean(frozen_x)
    x_std = np.std(frozen_x)
    x_min = np.min(frozen_x)
    x_max = np.max(frozen_x)
    
    # 计算线电荷密度分布
    L = params.get('L', 4e-6)
    d = params.get('d', 6e-6)
    
    # 定义极板区域
    left_plate_center = -L/2
    right_plate_center = L/2
    plate_width = d
    
    # 计算每个区域的粒子数
    left_mask = (frozen_x >= left_plate_center - plate_width/2) & (frozen_x <= left_plate_center + plate_width/2)
    right_mask = (frozen_x >= right_plate_center - plate_width/2) & (frozen_x <= right_plate_center + plate_width/2)
    
    left_count = np.sum(left_mask)
    right_count = np.sum(right_mask)
    other_count = len(frozen_x) - left_count - right_count
    
    # 计算柱状图
    n_bins = 100
    hist, bin_edges = np.histogram(frozen_x, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # 线电荷密度
    line_charge_density = hist * e / bin_width
    
    # 使用KDE进行平滑
    if len(frozen_x) > 1:
        kde = gaussian_kde(frozen_x, bw_method=0.1)
        x_kde = np.linspace(x_min, x_max, 1000)
        density = kde(x_kde)
        smoothed_charge_density = density * len(frozen_x) * e
    else:
        x_kde = np.array([])
        smoothed_charge_density = np.array([])
    
    spatial_results = {
        'frozen_x': frozen_x,
        'stats': {
            'mean': x_mean,
            'std': x_std,
            'min': x_min,
            'max': x_max
        },
        'plate_counts': {
            'left': left_count,
            'right': right_count,
            'other': other_count
        },
        'histogram': {
            'bin_centers': bin_centers,
            'bin_width': bin_width,
            'line_charge_density': line_charge_density
        },
        'kde': {
            'x': x_kde,
            'smoothed_density': smoothed_charge_density
        }
    }
    
    return spatial_results

def analyze_temporal_distribution(frozen_events):
    """分析电荷沉积的时间分布"""
    if frozen_events is None or len(frozen_events) == 0:
        return None
    
    # 按时间排序
    times = frozen_events['time'].values
    times_sorted = np.sort(times)
    
    # 计算时间间隔
    if len(times_sorted) > 1:
        intervals = np.diff(times_sorted)
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
    else:
        intervals = np.array([])
        avg_interval = 0
        std_interval = 0
    
    # 计算沉积速率的时间变化
    time_resolution = 50
    time_bins = np.linspace(0, times_sorted[-1], time_resolution)
    deposition_counts, _ = np.histogram(times_sorted, bins=time_bins)
    time_centers = (time_bins[:-1] + time_bins[1:]) / 2
    bin_width = time_bins[1] - time_bins[0]
    deposition_rate = deposition_counts / bin_width  # 粒子/秒
    
    temporal_results = {
        'times': times_sorted,
        'intervals': intervals,
        'interval_stats': {
            'mean': avg_interval,
            'std': std_interval
        },
        'deposition_rate': {
            'time_centers': time_centers,
            'rate': deposition_rate
        }
    }
    
    return temporal_results

# ===================== 可视化函数 =====================
def plot_charge_evolution(results, params, output_dir):
    """绘制电荷密度演化图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    time_us = results['time_points'] * 1e6  # 转换为微秒
    
    # 1. 累积冻结粒子数
    ax1 = axes[0, 0]
    ax1.plot(time_us, results['cumulative_particles'], 'b-', linewidth=2)
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Cumulative Frozen Particles')
    ax1.set_title('Cumulative Frozen Particles vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.annotate(f'Total: {results["total_frozen"]} particles',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 线电荷密度
    ax2 = axes[0, 1]
    ax2.plot(time_us, results['line_charge_density'], 'r-', linewidth=2)
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Line Charge Density (C/m)')
    ax2.set_title('Surface Charge Density Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.annotate(f'Final: {results["final_line_charge_density"]:.2e} C/m',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 3. 电荷沉积速率
    ax3 = axes[1, 0]
    ax3.plot(time_us, results['charge_rate'] * e, 'g-', linewidth=2)
    ax3.set_xlabel('Time (µs)')
    ax3.set_ylabel('Charge Deposition Rate (C/s·m)')
    ax3.set_title('Charge Deposition Rate')
    ax3.grid(True, alpha=0.3)
    ax3.annotate(f'Avg: {results["avg_deposition_rate"]*e:.2e} C/s·m',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. 粒子沉积速率
    ax4 = axes[1, 1]
    ax4.plot(time_us, results['particle_rate'], 'm-', linewidth=2)
    ax4.set_xlabel('Time (µs)')
    ax4.set_ylabel('Particle Deposition Rate (particles/s)')
    ax4.set_title('Particle Deposition Rate')
    ax4.grid(True, alpha=0.3)
    ax4.annotate(f'Avg: {results["avg_deposition_rate"]:.1f} particles/s',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Charge Density Evolution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'charge_density_evolution.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_spatial_distribution(spatial_results, params, output_dir):
    """绘制空间分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    L = params.get('L', 4e-6)
    d = params.get('d', 6e-6)
    
    # 1. 直方图分布
    ax1 = axes[0, 0]
    if spatial_results is not None:
        bin_centers = spatial_results['histogram']['bin_centers']
        line_charge_density = spatial_results['histogram']['line_charge_density']
        
        ax1.bar(bin_centers*1e6, line_charge_density, 
                width=spatial_results['histogram']['bin_width']*1e6 * 0.8,
                alpha=0.7, color='steelblue', edgecolor='black')
    
    # 标记极板位置
    plate1_left = (-L/2 - d/2) * 1e6
    plate1_right = (-L/2 + d/2) * 1e6
    plate2_left = (L/2 - d/2) * 1e6
    plate2_right = (L/2 + d/2) * 1e6
    
    ylim = ax1.get_ylim()
    ax1.axvspan(plate1_left, plate1_right, alpha=0.2, color='red', label='Left Plate')
    ax1.axvspan(plate2_left, plate2_right, alpha=0.2, color='blue', label='Right Plate')
    
    ax1.set_xlabel('x position (µm)')
    ax1.set_ylabel('Line Charge Density (C/m)')
    ax1.set_title('Spatial Charge Density Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. 平滑KDE分布
    ax2 = axes[0, 1]
    if spatial_results is not None and len(spatial_results['kde']['x']) > 0:
        x_kde = spatial_results['kde']['x'] * 1e6
        smoothed_density = spatial_results['kde']['smoothed_density']
        
        ax2.plot(x_kde, smoothed_density, 'r-', linewidth=2, label='Smoothed')
        
        # 标记极板位置
        ylim2 = ax2.get_ylim()
        ax2.axvspan(plate1_left, plate1_right, alpha=0.2, color='red')
        ax2.axvspan(plate2_left, plate2_right, alpha=0.2, color='blue')
        
        ax2.set_xlabel('x position (µm)')
        ax2.set_ylabel('Line Charge Density (C/m)')
        ax2.set_title('Smoothed Charge Density Distribution')
        ax2.grid(True, alpha=0.3)
    
    # 3. 极板分布饼图
    ax3 = axes[1, 0]
    if spatial_results is not None:
        counts = spatial_results['plate_counts']
        labels = ['Left Plate', 'Right Plate', 'Other']
        sizes = [counts['left'], counts['right'], counts['other']]
        colors = ['red', 'blue', 'gray']
        
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90)
        ax3.set_title('Particle Distribution on Plates')
    
    # 4. 统计信息
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if spatial_results is not None:
        stats = spatial_results['stats']
        counts = spatial_results['plate_counts']
        
        text_str = f'Total particles: {counts["left"] + counts["right"] + counts["other"]}\n\n'
        text_str += f'Left plate: {counts["left"]} particles\n'
        text_str += f'Right plate: {counts["right"]} particles\n'
        text_str += f'Other regions: {counts["other"]} particles\n\n'
        text_str += f'Mean x: {stats["mean"]*1e6:.2f} µm\n'
        text_str += f'Std x: {stats["std"]*1e6:.2f} µm\n'
        text_str += f'Min x: {stats["min"]*1e6:.2f} µm\n'
        text_str += f'Max x: {stats["max"]*1e6:.2f} µm\n\n'
        text_str += f'Plate width: {d*1e6:.1f} µm\n'
        text_str += f'Plate spacing: {L*1e6:.1f} µm'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax4.text(0.05, 0.95, text_str, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    
    plt.suptitle('Spatial Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_distribution(temporal_results, output_dir):
    """绘制时间分布图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    if temporal_results is None:
        return
    
    # 1. 沉积时间序列
    ax1 = axes[0]
    times_us = temporal_results['times'] * 1e6
    particle_indices = np.arange(1, len(times_us) + 1)
    
    ax1.plot(times_us, particle_indices, 'b-', linewidth=2, marker='o', markersize=3)
    ax1.set_xlabel('Time (µs)')
    ax1.set_ylabel('Cumulative Particle Count')
    ax1.set_title('Particle Deposition Timeline')
    ax1.grid(True, alpha=0.3)
    
    # 2. 沉积间隔分布
    ax2 = axes[1]
    if len(temporal_results['intervals']) > 0:
        intervals_us = temporal_results['intervals'] * 1e6
        ax2.hist(intervals_us, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Interval between depositions (µs)')
        ax2.set_ylabel('Count')
        ax2.set_title('Deposition Interval Distribution')
        ax2.grid(True, alpha=0.3)
        
        stats = temporal_results['interval_stats']
        text_str = f'Mean: {stats["mean"]*1e6:.2f} µs\n'
        text_str += f'Std: {stats["std"]*1e6:.2f} µs\n'
        text_str += f'Min: {np.min(intervals_us):.2f} µs\n'
        text_str += f'Max: {np.max(intervals_us):.2f} µs'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax2.text(0.95, 0.95, text_str, transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    
    # 3. 沉积速率随时间变化
    ax3 = axes[2]
    time_centers_us = temporal_results['deposition_rate']['time_centers'] * 1e6
    deposition_rate = temporal_results['deposition_rate']['rate']
    
    ax3.plot(time_centers_us, deposition_rate, 'r-', linewidth=2)
    ax3.set_xlabel('Time (µs)')
    ax3.set_ylabel('Deposition Rate (particles/s)')
    ax3.set_title('Instantaneous Deposition Rate')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Temporal Distribution Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

# ===================== 主分析函数 =====================
def analyze_charge_density(sim_dir):
    """主分析函数"""
    print("="*50)
    print("电荷密度后处理分析")
    print("="*50)
    
    if sim_dir is None:
        print("错误: 必须指定模拟目录")
        print("用法: python analyze_charge_density.py --dir 模拟目录名")
        return
    
    if not os.path.exists(sim_dir):
        print(f"错误: 目录不存在: {sim_dir}")
        return
    
    # 加载数据
    print(f"加载模拟目录: {sim_dir}")
    data = load_simulation_data(sim_dir)
    if data is None:
        return
    
    # 创建分析结果目录
    analysis_dir = os.path.join(sim_dir, 'charge_analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"\n分析结果将保存到: {analysis_dir}")
    
    # 分析电荷密度演化
    print("\n2. 分析电荷密度随时间演化...")
    charge_results = analyze_charge_density_evolution(data['frozen_events'], data['params'])
    
    if charge_results is not None:
        # 保存结果
        results_df = pd.DataFrame({
            'time_s': charge_results['time_points'],
            'time_us': charge_results['time_points'] * 1e6,
            'cumulative_particles': charge_results['cumulative_particles'],
            'cumulative_charge': charge_results['cumulative_charge'],
            'line_charge_density_Cpm': charge_results['line_charge_density'],
            'charge_rate_Cpspm': charge_results['charge_rate'] * e,
            'particle_rate_pps': charge_results['particle_rate']
        })
        results_df.to_csv(os.path.join(analysis_dir, 'charge_density_evolution.csv'), index=False)
        
        # 绘制图表
        print("3. 绘制电荷密度演化图...")
        plot_charge_evolution(charge_results, data['params'], analysis_dir)
        
        # 打印摘要
        print("\n电荷密度演化摘要:")
        print(f"   总冻结粒子数: {charge_results['total_frozen']}")
        print(f"   总模拟时间: {charge_results['total_time']*1e6:.2f} µs")
        print(f"   平均沉积速率: {charge_results['avg_deposition_rate']:.1f} particles/s")
        print(f"   最终线电荷密度: {charge_results['final_line_charge_density']:.2e} C/m")
        print(f"   平均线电荷密度: {charge_results['avg_line_charge_density']:.2e} C/m")
    
    # 分析空间分布
    print("\n4. 分析电荷空间分布...")
    spatial_results = analyze_spatial_distribution(data['frozen_events'], data['params'])
    
    if spatial_results is not None:
        # 保存结果
        spatial_df = pd.DataFrame({
            'frozen_x_m': spatial_results['frozen_x'],
            'frozen_x_um': spatial_results['frozen_x'] * 1e6
        })
        spatial_df.to_csv(os.path.join(analysis_dir, 'spatial_distribution.csv'), index=False)
        
        # 绘制图表
        print("5. 绘制空间分布图...")
        plot_spatial_distribution(spatial_results, data['params'], analysis_dir)
        
        # 打印摘要
        counts = spatial_results['plate_counts']
        stats = spatial_results['stats']
        print("\n空间分布摘要:")
        print(f"   左极板粒子数: {counts['left']} ({counts['left']/len(spatial_results['frozen_x'])*100:.1f}%)")
        print(f"   右极板粒子数: {counts['right']} ({counts['right']/len(spatial_results['frozen_x'])*100:.1f}%)")
        print(f"   其他区域粒子数: {counts['other']} ({counts['other']/len(spatial_results['frozen_x'])*100:.1f}%)")
        print(f"   平均x位置: {stats['mean']*1e6:.2f} µm")
        print(f"   x位置标准差: {stats['std']*1e6:.2f} µm")
    
    # 分析时间分布
    print("\n6. 分析电荷沉积时间分布...")
    temporal_results = analyze_temporal_distribution(data['frozen_events'])
    
    if temporal_results is not None:
        # 保存结果
        temporal_df = pd.DataFrame({
            'deposition_time_s': temporal_results['times'],
            'deposition_time_us': temporal_results['times'] * 1e6,
            'interval_s': np.concatenate([[0], temporal_results['intervals']]),
            'interval_us': np.concatenate([[0], temporal_results['intervals']]) * 1e6
        })
        temporal_df.to_csv(os.path.join(analysis_dir, 'temporal_distribution.csv'), index=False)
        
        # 绘制图表
        print("7. 绘制时间分布图...")
        plot_temporal_distribution(temporal_results, analysis_dir)
        
        # 打印摘要
        if len(temporal_results['intervals']) > 0:
            stats = temporal_results['interval_stats']
            print("\n时间分布摘要:")
            print(f"   平均沉积间隔: {stats['mean']*1e6:.2f} µs")
            print(f"   沉积间隔标准差: {stats['std']*1e6:.2f} µs")
    
    # 生成分析报告
    print("\n8. 生成分析报告...")
    generate_analysis_report(charge_results, spatial_results, temporal_results, 
                            data['params'], analysis_dir)
    
    print(f"\n分析完成！结果保存在: {analysis_dir}")
    print("\n生成的文件:")
    print(f"  - charge_density_evolution.csv: 电荷密度演化数据")
    print(f"  - spatial_distribution.csv: 空间分布数据")
    print(f"  - temporal_distribution.csv: 时间分布数据")
    print(f"  - charge_density_evolution.png: 电荷密度演化图")
    print(f"  - spatial_distribution_analysis.png: 空间分布分析图")
    print(f"  - temporal_distribution_analysis.png: 时间分布分析图")
    print(f"  - analysis_report.txt: 分析报告")

def generate_analysis_report(charge_results, spatial_results, temporal_results, params, output_dir):
    """生成分析报告"""
    report_file = os.path.join(output_dir, 'analysis_report.txt')
    
    # 修改这里：指定UTF-8编码
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("电荷密度分析报告\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. 模拟参数\n")
        f.write("-"*40 + "\n")
        for key, value in params.items():
            if isinstance(value, float):
                f.write(f"   {key}: {value:.4e}\n")
            else:
                f.write(f"   {key}: {value}\n")
        f.write("\n")
        
        if charge_results is not None:
            f.write("2. 电荷密度演化分析\n")
            f.write("-"*40 + "\n")
            f.write(f"   总冻结粒子数: {charge_results['total_frozen']}\n")
            f.write(f"   总模拟时间: {charge_results['total_time']*1e6:.2f} µs\n")  # 这里包含µ字符
            f.write(f"   平均沉积速率: {charge_results['avg_deposition_rate']:.1f} particles/s\n")
            f.write(f"   最终线电荷密度: {charge_results['final_line_charge_density']:.2e} C/m\n")
            f.write(f"   平均线电荷密度: {charge_results['avg_line_charge_density']:.2e} C/m\n")
            f.write("\n")
        
        if spatial_results is not None:
            f.write("3. 空间分布分析\n")
            f.write("-"*40 + "\n")
            counts = spatial_results['plate_counts']
            total = counts['left'] + counts['right'] + counts['other']
            f.write(f"   左极板粒子数: {counts['left']} ({counts['left']/total*100:.1f}%)\n")
            f.write(f"   右极板粒子数: {counts['right']} ({counts['right']/total*100:.1f}%)\n")
            f.write(f"   其他区域粒子数: {counts['other']} ({counts['other']/total*100:.1f}%)\n")
            
            stats = spatial_results['stats']
            f.write(f"   平均x位置: {stats['mean']*1e6:.2f} µm\n")  # 这里也包含µ字符
            f.write(f"   x位置标准差: {stats['std']*1e6:.2f} µm\n")
            f.write(f"   x位置范围: [{stats['min']*1e6:.2f}, {stats['max']*1e6:.2f}] µm\n")
            f.write("\n")
        
        if temporal_results is not None and len(temporal_results['intervals']) > 0:
            f.write("4. 时间分布分析\n")
            f.write("-"*40 + "\n")
            stats = temporal_results['interval_stats']
            intervals = temporal_results['intervals']
            f.write(f"   平均沉积间隔: {stats['mean']*1e6:.2f} µs\n")
            f.write(f"   沉积间隔标准差: {stats['std']*1e6:.2f} µs\n")
            f.write(f"   最小沉积间隔: {np.min(intervals)*1e6:.2f} µs\n")
            f.write(f"   最大沉积间隔: {np.max(intervals)*1e6:.2f} µs\n")
            f.write("\n")
        
        f.write("5. 分析文件\n")
        f.write("-"*40 + "\n")
        f.write("   charge_density_evolution.csv - 电荷密度演化数据\n")
        f.write("   spatial_distribution.csv - 空间分布数据\n")
        f.write("   temporal_distribution.csv - 时间分布数据\n")
        f.write("   各种PNG文件 - 分析图表\n")
        f.write("\n分析完成时间: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

# ===================== 主程序 =====================

if __name__ == "__main__":
    # 在这里直接指定你的模拟目录名
    sim_dir = "sim_3080particles_0.4ms_P0.20bar_T300.0K_E1.0kVcm_1.5e-06PR_20251231_134109"  # ← 修改这一行，改成你的目录名
    
    # 调用分析函数
    analyze_charge_density(sim_dir)