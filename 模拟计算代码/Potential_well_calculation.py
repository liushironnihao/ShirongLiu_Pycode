import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid  # 替代cumtrapz

def extract_parameters_from_main():
    """Extract parameters from main code file"""
    # 查找主代码文件
    main_files = []
    for file in os.listdir('.'):
        if file.endswith('.py') and 'main' in file.lower():
            main_files.append(file)
    
    if not main_files:
        print("No main code file found")
        return None
    
    # 使用最新的主代码文件
    main_file = max(main_files, key=lambda x: os.path.getmtime(x))
    print(f"Extracting parameters from: {main_file}")
    
    parameters = {}
    
    try:
        with open(main_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取几何参数
        import re
        
        # 提取L, d, h, h1等几何参数
        patterns = {
            'L': r'L\s*=\s*([\d.e+-]+)',
            'd': r'd\s*=\s*([\d.e+-]+)', 
            'h': r'h\s*=\s*([\d.e+-]+)',
            'h1': r'h1\s*=\s*([\d.e+-]+)',
            'E1': r'E1\s*=\s*([\d.e+-]+)',
            'E2': r'E2\s*=\s*([\d.e+-]+)',
            'f': r'f\s*=\s*([\d.e+-]+)',
            'duty1': r'duty1\s*=\s*([\d.e+-]+)',
            'sigma': r'sigma\s*=\s*([\d.e+-]+)',
            'PO': r'PO\s*=\s*([\d.e+-]+)',
            'sim_time_ms': r'sim_time_ms\s*=\s*([\d.e+-]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    parameters[key] = float(match.group(1))
                    print(f"Extracted {key} = {parameters[key]}")
                except ValueError:
                    print(f"Warning: Cannot parse value for {key}")
        
        # 提取计算域边界
        x_min_match = re.search(r'x_min\s*=\s*([\d.e+-]+)', content)
        x_max_match = re.search(r'x_max\s*=\s*([\d.e+-]+)', content)
        y_min_match = re.search(r'y_min\s*=\s*([\d.e+-]+)', content)
        
        if x_min_match and x_max_match and y_min_match:
            parameters['x_min'] = float(x_min_match.group(1))
            parameters['x_max'] = float(x_max_match.group(1))
            parameters['y_min'] = float(y_min_match.group(1))
            # 计算上边界
            if 'h' in parameters and 'h1' in parameters:
                parameters['y_max'] = parameters['h'] + parameters['h1']
                parameters['upper_plate_y'] = parameters['h'] + parameters['h1']
                parameters['lower_plate_y'] = parameters['h']
        
        return parameters
        
    except Exception as e:
        print(f"Error extracting parameters: {e}")
        return None

def postprocess_electric_field():
    """Post-process electric field data, plot potential and field distribution at x=-d/2"""
    
    # 直接使用指定的目录
    sim_dir = "sim_3200particles_0.4ms_P1.00bar_T300.0K_E1.0kVcm_1.5e-06PR_20260108_114012"

    if not os.path.exists(sim_dir):
        print(f"Simulation directory not found: {sim_dir}")
        return

    print(f"Using simulation directory: {sim_dir}")
    
    # 从模拟结果目录提取参数
    params_file = os.path.join(sim_dir, "simulation_data", "simulation_parameters.csv")
    if os.path.exists(params_file):
        try:
            params_df = pd.read_csv(params_file)
            params = params_df.iloc[0].to_dict()
            print("Extracted parameters from simulation results:")
            for key, value in params.items():
                print(f"  {key} = {value}")
        except Exception as e:
            print(f"Error reading parameters file: {e}")
            params = None
    else:
        print(f"Parameters file not found: {params_file}")
        params = None
    
    # 如果从模拟结果读取失败，回退到从主代码文件提取
    if params is None:
        print("Falling back to extracting from main code file")
        params = extract_parameters_from_main()
    
    if not params:
        print("Using default parameters")
        # 默认参数（如果提取失败）
        params = {
            'L': 4e-6, 'd': 6e-6, 'h': 1.5e-6, 'h1': 4e-6,
            'E1': 1e5, 'E2': 0e5, 'f': 200e3, 'duty1': 1,
            'x_min': -4e-6, 'x_max': 4e-6, 'y_min': 0,
            'upper_plate_y': 5.5e-6, 'lower_plate_y': 1.5e-6
        }
    
    # 使用提取的参数
    L = params['L']
    d = params['d'] 
    h = params['h']
    h1 = params['h1']
    E1 = params['E1']
    E2 = params['E2']
    f = params['f']
    duty1 = params.get('duty1', 1.0)
    upper_plate_y = params.get('upper_plate_y', h + h1)  # 如果不存在则计算
    lower_plate_y = params.get('lower_plate_y', h)       # 如果不存在则使用h
    
    # 指定x位置
    x_position = -d/2  # x = -d/2
    
    # 读取电场历史数据
    field_history_file = os.path.join(sim_dir, "electric_fields", "field_history.pkl")
    
    if not os.path.exists(field_history_file):
        print(f"Electric field history file not found: {field_history_file}")
        return
    
    try:
        with open(field_history_file, 'rb') as file_obj:
            field_data = pickle.load(file_obj)
        
        if not field_data:
            print("Electric field history data is empty")
            return
        
        # 获取最后一帧的电场数据
        last_frame = field_data[-1]
        print(f"Using last frame data, time: {last_frame['time']*1e6:.2f} μs")
        
        # 获取网格数据
        xg = last_frame['xg']
        yg = last_frame['yg']
        E1_grid = last_frame['E1_grid']
        E2_grid = last_frame['E2_grid']
        
        # 根据时间确定使用哪个电场
        T_pulse = 1/f  # 频率
        t1_E1 = T_pulse * duty1
        phase = last_frame['time'] % T_pulse
        
        if phase < t1_E1:
            E_grid = E1_grid
            field_label = 'E1 field'
            E0 = E1
        else:
            E_grid = E2_grid
            field_label = 'E2 field'
            E0 = E2
        
        print(f"Using electric field: {field_label} (E0 = {E0/1e3:.1f} kV/m)")
        
        # 创建电场插值器
        E_interp = RegularGridInterpolator(
            (xg, yg), E_grid, bounds_error=False, fill_value=0
        )
        
        # 在x=-d/2附近0.5μm区域内创建多个x位置采样
        x_center = -d/2
        x_range = 0.5e-6
        x_samples = np.linspace(x_center - x_range/2, x_center + x_range/2, 5)

        # 在x范围内创建y方向采样点
        y_points = np.linspace(h, h + h1, 1000)

        # 为每个x位置计算电场和电势
        all_phi = []
        all_E_total = []

        for x_sample in x_samples:
            x_points = np.full_like(y_points, x_sample)
            positions = np.column_stack([x_points, y_points])
            
            # 获取电场分量
            E_values = E_interp(positions)
            Ey = E_values[:, 1]
            E_total = np.sqrt(E_values[:, 0]**2 + Ey**2)
            
            # 计算电势分布
            phi = cumulative_trapezoid(Ey, y_points, initial=0)
            phi = -phi
            phi = phi - phi[0]
            
            all_phi.append(phi)
            all_E_total.append(E_total)

        # 计算平均值
        phi_avg = np.mean(all_phi, axis=0)
        E_total_avg = np.mean(all_E_total, axis=0)
        
        # 寻找关键点 - 使用电场最高点
        idx_max_E = np.argmax(E_total_avg)
        y_max_E = y_points[idx_max_E]
        phi_at_max_E = phi_avg[idx_max_E]

        # 最低电势位置
        idx_min_phi = np.argmin(phi_avg)
        y_min_phi = y_points[idx_min_phi]
        phi_min = phi_avg[idx_min_phi]

        # 计算差值
        delta_phi = phi_at_max_E - phi_min
        delta_y = y_min_phi - y_max_E

        print(f"Key point information:")
        print(f"  At max E field: y = {y_max_E*1e6:.2f} μm, φ = {phi_at_max_E:.4f} V")
        print(f"  Minimum potential: y = {y_min_phi*1e6:.2f} μm, φ = {phi_min:.4f} V")
        print(f"  Potential difference Δφ = {delta_phi:.4f} V")
        print(f"  Distance difference Δy = {delta_y*1e6:.2f} μm")
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 绘制电势分布
        ax1.plot(y_points * 1e6, phi_avg, 'b-', linewidth=2, label='Potential distribution')

        # 标记关键点 - 使用电场最高点
        ax1.plot(y_max_E * 1e6, phi_at_max_E, 'ro', markersize=8, 
                label=f'At max E: φ={phi_at_max_E:.3f}V')
        ax1.plot(y_min_phi * 1e6, phi_min, 'go', markersize=8,
                label=f'Min potential: φ={phi_min:.3f}V')

        # 添加连接线
        ax1.plot([y_max_E * 1e6, y_min_phi * 1e6], 
                [phi_at_max_E, phi_min], 'k--', alpha=0.7, linewidth=1)

        # 添加差值标注
        ax1.annotate(f'Δφ = {delta_phi:.3f} V\nΔy = {delta_y*1e6:.2f} μm',
                    xy=((y_max_E + y_min_phi)/2 * 1e6, (phi_at_max_E + phi_min)/2),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('y position (μm)')
        ax1.set_ylabel('Electric Potential φ (V)')
        ax1.set_title(f'Electric Potential Distribution at x = -d/2 ± {x_range*1e6/2:.1f} μm (average)\n'
                     f'({field_label}, Time: {last_frame["time"]*1e6:.2f} μs)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 绘制电场分布
        ax2.plot(y_points * 1e6, E_total_avg / 1e3, 'r-', linewidth=2, label='Total electric field')

        # 标记最高场强点
        ax2.plot(y_max_E * 1e6, E_total_avg[idx_max_E] / 1e3, 'ro', markersize=8, 
                 label=f'Max E: {E_total_avg[idx_max_E]/1e3:.1f} kV/m')

        ax2.set_xlabel('y position (μm)')
        ax2.set_ylabel('Electric Field Strength (kV/m)')
        ax2.set_title(f'Electric Field Distribution at x = -d/2 ± {x_range*1e6/2:.1f} μm (average)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = os.path.join(sim_dir, "visualization")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "postprocess_electric_potential.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nImage saved: {output_file}")
        
        # plt.show()  # 注释掉不显示图片
        
        return {
            'y_points': y_points,
            'phi': phi_avg,
            'E_total': E_total_avg,
            'y_max_E': y_max_E,
            'phi_at_max_E': phi_at_max_E,
            'y_min_phi': y_min_phi,
            'phi_min': phi_min,
            'delta_phi': delta_phi,
            'delta_y': delta_y,
            'parameters': params
        }
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = postprocess_electric_field()
    
    if result:
        print("\nProcessing completed!")
        print(f"Geometry parameters used:")
        print(f"  L = {result['parameters']['L']*1e6:.1f} μm")
        print(f"  d = {result['parameters']['d']*1e6:.1f} μm") 
        print(f"  h = {result['parameters']['h']*1e6:.1f} μm")
        print(f"  h1 = {result['parameters']['h1']*1e6:.1f} μm")
        print(f"Potential difference: {result['delta_phi']:.4f} V")
        print(f"Distance difference: {result['delta_y']*1e6:.2f} μm")
    else:
        print("Processing failed!")