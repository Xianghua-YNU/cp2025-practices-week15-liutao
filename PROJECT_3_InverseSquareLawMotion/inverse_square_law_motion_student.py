"""
学生模板：平方反比引力场中的运动
文件：inverse_square_law_motion_student.py
作者：刘涛
日期：2025.06.04

重要：函数名称、参数名称和返回值的结构必须与参考答案保持一致！
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#设置matplotlib全局语言为中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 常量（可调整或作为参数传递）
GM = 1.0  # 引力常数 * 中心天体质量（例如，G*M_sun）
# 可以假设轨道粒子质量 m=1

def derivatives(t, state_vector, gm_val):
    """
    计算状态向量 [x, y, vx, vy] 的导数。

    笛卡尔坐标系中的运动方程为：
    dx/dt = vx
    dy/dt = vy
    dvx/dt = -GM * x / r^3
    dvy/dt = -GM * y / r^3
    其中 r = sqrt(x^2 + y^2)。

    参数：
        t (float): 当前时间（在此自治系统中不直接使用，但solve_ivp需要）
        state_vector (np.ndarray): 一维数组 [x, y, vx, vy] 表示当前状态
        gm_val (float): 引力常数 G 和中心质量 M 的乘积

    返回：
        np.ndarray: 导数的一维数组 [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """
    x, y, vx, vy = state_vector
    r_cubed = (x**2 + y**2)**1.5
    
    # 处理 r 很小时可能出现的除以零情况
    if r_cubed < 1e-12: # 防止除以零的小阈值
        ax = -gm_val * x / (1e-12) if x != 0 else 0
        ay = -gm_val * y / (1e-12) if y != 0 else 0
        return [vx, vy, ax, ay]

    ax = -gm_val * x / r_cubed
    ay = -gm_val * y / r_cubed
    return [vx, vy, ax, ay]

def solve_orbit(initial_conditions, t_span, t_eval, gm_val=GM):
    """
    使用 scipy.integrate.solve_ivp 解决轨道运动问题。

    参数：
        initial_conditions (list or np.ndarray): [x0, y0, vx0, vy0] 在 t_start 时的初始状态
        t_span (tuple): (t_start, t_end)，积分区间
        t_eval (np.ndarray): 存储解的时间点数组
        gm_val (float, optional): GM 值。默认为全局 GM

    返回：
        scipy.integrate.OdeSolution: solve_ivp 返回的解对象
                                     通过 sol.y 访问解（转置为 (N_points, N_vars)）
                                     sol.t 包含时间点
    """
    sol = solve_ivp(
        fun=derivatives, 
        t_span=t_span, 
        y0=initial_conditions, 
        t_eval=t_eval, 
        args=(gm_val,),
        method='RK45',  # 5(4)阶显式龙格-库塔方法
        rtol=1e-7,      # 相对容差
        atol=1e-9       # 绝对容差
    )
    return sol

def calculate_energy(state_vector, gm_val=GM, m=1.0):
    """
    计算粒子的比机械能（单位质量的能量）。
    E/m = 0.5 * v^2 - GM/r

    参数：
        state_vector (np.ndarray): 二维数组，每行为 [x, y, vx, vy] 或单个状态的一维数组
        gm_val (float, optional): GM 值。默认为全局 GM
        m (float, optional): 轨道粒子的质量。默认为 1.0（计算比能）

    返回：
        np.ndarray or float: 比机械能（如果 m 不为 1 则为总能量）
    """
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)

    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    r = np.sqrt(x**2 + y**2)
    v_squared = vx**2 + vy**2
    
    # 避免 r 为零时的除以零情况
    # 如果 r 为零，势能未定义（无穷大）。这在有效轨道中不应发生
    potential_energy_per_m = np.zeros_like(r)
    non_zero_r_mask = r > 1e-12
    potential_energy_per_m[non_zero_r_mask] = -gm_val / r[non_zero_r_mask]
    if np.any(~non_zero_r_mask):
        print("警告：能量计算中遇到 r=0。势能奇异。")
        potential_energy_per_m[~non_zero_r_mask] = -np.inf # 或其他指示符

    kinetic_energy_per_m = 0.5 * v_squared
    specific_energy = kinetic_energy_per_m + potential_energy_per_m
    
    total_energy = m * specific_energy

    return total_energy[0] if is_single_state else total_energy

def calculate_angular_momentum(state_vector, m=1.0):
    """
    计算粒子的比角动量（z 分量）。
    Lz/m = x*vy - y*vx

    参数：
        state_vector (np.ndarray): 二维数组，每行为 [x, y, vx, vy] 或单个状态的一维数组
        m (float, optional): 轨道粒子的质量。默认为 1.0（计算比角动量）

    返回：
        np.ndarray or float: 比角动量（如果 m 不为 1 则为总 Lz）
    """
    is_single_state = state_vector.ndim == 1
    if is_single_state:
        state_vector = state_vector.reshape(1, -1)
        
    x = state_vector[:, 0]
    y = state_vector[:, 1]
    vx = state_vector[:, 2]
    vy = state_vector[:, 3]

    specific_Lz = x * vy - y * vx
    total_Lz = m * specific_Lz
    
    return total_Lz[0] if is_single_state else total_Lz


if __name__ == "__main__":
    # --- 使用演示 ---
    print("演示轨道模拟...")

    # 通用参数
    t_start = 0
    t_end_ellipse = 20  # 足够的时间让典型椭圆轨道运行几圈
    t_end_hyperbola = 5 # 双曲线轨道快速远离
    t_end_parabola = 10 # 抛物线轨道也会远离
    n_points = 1000
    mass_particle = 1.0 # 假设 m=1 以简化 E 和 L 的计算

    # 情况 1：椭圆轨道 (E < 0)
    # 初始条件：x0=1, y0=0, vx0=0, vy0=0.8
    ic_ellipse = [1.0, 0.0, 0.0, 0.8]
    t_eval_ellipse = np.linspace(t_start, t_end_ellipse, n_points)
    sol_ellipse = solve_orbit(ic_ellipse, (t_start, t_end_ellipse), t_eval_ellipse, gm_val=GM)
    x_ellipse, y_ellipse = sol_ellipse.y[0], sol_ellipse.y[1]
    energy_ellipse = calculate_energy(sol_ellipse.y.T, GM, mass_particle)
    Lz_ellipse = calculate_angular_momentum(sol_ellipse.y.T, mass_particle)
    print(f"椭圆轨道：初始 E = {energy_ellipse[0]:.3f}, 初始 Lz = {Lz_ellipse[0]:.3f}")
    print(f"椭圆轨道：最终 E = {energy_ellipse[-1]:.3f}, 最终 Lz = {Lz_ellipse[-1]:.3f}（能量/角动量守恒检查）")

    # 情况 2：抛物线轨道 (E = 0)
    # 对于 E=0，逃逸速度 v_escape = sqrt(2*GM/r)。如果 x0=1, y0=0，则 vy0 = sqrt(2*GM/1) = sqrt(2)
    ic_parabola = [1.0, 0.0, 0.0, np.sqrt(2*GM)]
    t_eval_parabola = np.linspace(t_start, t_end_parabola, n_points)
    sol_parabola = solve_orbit(ic_parabola, (t_start, t_end_parabola), t_eval_parabola, gm_val=GM)
    x_parabola, y_parabola = sol_parabola.y[0], sol_parabola.y[1]
    energy_parabola = calculate_energy(sol_parabola.y.T, GM, mass_particle)
    print(f"抛物线轨道：初始 E = {energy_parabola[0]:.3f}")

    # 情况 3：双曲线轨道 (E > 0)
    # 如果 vy0 > 逃逸速度，例如 vy0 = 1.5 * sqrt(2*GM)
    ic_hyperbola = [1.0, 0.0, 0.0, 1.2 * np.sqrt(2*GM)] # 速度大于逃逸速度
    t_eval_hyperbola = np.linspace(t_start, t_end_hyperbola, n_points)
    sol_hyperbola = solve_orbit(ic_hyperbola, (t_start, t_end_hyperbola), t_eval_hyperbola, gm_val=GM)
    x_hyperbola, y_hyperbola = sol_hyperbola.y[0], sol_hyperbola.y[1]
    energy_hyperbola = calculate_energy(sol_hyperbola.y.T, GM, mass_particle)
    print(f"双曲线轨道：初始 E = {energy_hyperbola[0]:.3f}")

    # 绘制轨道
    plt.figure(figsize=(10, 8))
    plt.plot(x_ellipse, y_ellipse, label=f'椭圆轨道 (E={energy_ellipse[0]:.2f})')
    plt.plot(x_parabola, y_parabola, label=f'抛物线轨道 (E={energy_parabola[0]:.2f})')
    plt.plot(x_hyperbola, y_hyperbola, label=f'双曲线轨道 (E={energy_hyperbola[0]:.2f})')
    plt.plot(0, 0, 'ko', markersize=10, label='中心天体 (太阳)') # 中心天体
    plt.title('平方反比定律引力场中的轨道')
    plt.xlabel('x (任意单位)')
    plt.ylabel('y (任意单位)')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axis('equal') # 对轨道的正确纵横比至关重要
    plt.show()

    # --- 任务 3 演示：在 E < 0 时变化角动量 ---
    print("\n演示在 E < 0 时变化角动量...")
    E_target = -0.2 # 目标负能量（必须 < 0 才能形成椭圆轨道）
    r0 = 1.5       # 初始距离中心的位置（在 x 轴上）
    # E = 0.5*m*v0y^2 - GM*m/r0  => v0y = sqrt(2/m * (E_target + GM*m/r0))
    # 确保 (E_target + GM*m/r0) 为正以获得实数 v0y
    if E_target + GM * mass_particle / r0 < 0:
        print(f"错误：无法在 r0={r0} 处达到 E_target={E_target}。E_target 必须 > -GM*m/r0。")
        print(f"要求 E_target > {-GM*mass_particle/r0}")
    else:
        vy_base = np.sqrt(2/mass_particle * (E_target + GM * mass_particle / r0))
        
        initial_conditions_L = []
                
        v0_for_E_target = np.sqrt(2/mass_particle * (E_target + GM*mass_particle/r0))
        print(f"对于 r0={r0} 处的 E_target={E_target}，所需速度 v0={v0_for_E_target:.3f}")

        plt.figure(figsize=(10, 8))
        plt.plot(0, 0, 'ko', markersize=10, label='中心天体')

        # 发射角度（theta）以变化 Lz，保持 v0（从而 E）不变
        launch_angles_deg = [90, 60, 45] # 速度向量与正 x 轴的角度（度）
        
        for i, angle_deg in enumerate(launch_angles_deg):
            angle_rad = np.deg2rad(angle_deg)
            vx0 = v0_for_E_target * np.cos(angle_rad)
            vy0 = v0_for_E_target * np.sin(angle_rad)
            ic = [r0, 0, vx0, vy0]
            
            current_E = calculate_energy(np.array(ic), GM, mass_particle)
            current_Lz = calculate_angular_momentum(np.array(ic), mass_particle)
            print(f"  角度 {angle_deg}度：计算 E={current_E:.3f}（目标 E={E_target:.3f}），Lz={current_Lz:.3f}")

            sol = solve_orbit(ic, (t_start, t_end_ellipse*1.5), np.linspace(t_start, t_end_ellipse*1.5, n_points), gm_val=GM)
            plt.plot(sol.y[0], sol.y[1], label=f'Lz={current_Lz:.2f}（发射角度 {angle_deg}°）')

        plt.title(f'固定能量 (E ≈ {E_target:.2f}) 和变化角动量的椭圆轨道')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axis('equal')
        plt.show()
