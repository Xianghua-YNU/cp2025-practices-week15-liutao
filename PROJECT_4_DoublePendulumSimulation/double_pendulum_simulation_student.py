"""
学生模板：双摆模拟
课程：计算物理
说明：请实现标记为 TODO 的函数。
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import matplotlib.animation as animation

# 可以在函数中使用的常量
G_CONST = 9.81  # 重力加速度 (m/s^2)
L_CONST = 0.4   # 每个摆臂的长度 (m)
M_CONST = 1.0   # 每个摆锤的质量 (kg)

def derivatives(y, t, L1, L2, m1, m2, g):
    """
    Returns the time derivatives of the double pendulum state vector y.
    """
    theta1, omega1, theta2, omega2 = y
    
    dtheta1_dt = omega1
    dtheta2_dt = omega2

    # Numerator and denominator for domega1_dt
    num1 = -omega1**2 * np.sin(2*theta1 - 2*theta2) \
           - 2 * omega2**2 * np.sin(theta1 - theta2) \
           - (g_param/L1) * (np.sin(theta1 - 2*theta2) + 3*np.sin(theta1))
    den1 = 3 - np.cos(2*theta1 - 2*theta2)
    
    domega1_dt = num1 / den1

    # Numerator and denominator for domega2_dt
    num2 = 4 * omega1**2 * np.sin(theta1 - theta2) \
           + omega2**2 * np.sin(2*theta1 - 2*theta2) \
           + 2 * (g_param/L1) * (np.sin(2*theta1 - theta2) - np.sin(theta2))
    den2 = 3 - np.cos(2*theta1 - 2*theta2)
    
    domega2_dt = num2 / den2
    
    return [dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt]

def solve_double_pendulum(initial_conditions, t_span, t_points, L_param=L_CONST, g_param=G_CONST):
    """
    Solves the double pendulum ODEs using odeint with high precision settings.
    """
    y0 = [initial_conditions['theta1'], initial_conditions['omega1'], 
          initial_conditions['theta2'], initial_conditions['omega2']]
    
    t_arr = np.linspace(t_span[0], t_span[1], t_points)
    
    # Using strict tolerances for better energy conservation
    sol_arr = odeint(derivatives, y0, t_arr, 
                     args=(L_param, L_param, M_CONST, M_CONST, g_param), 
                     rtol=1e-10, atol=1e-10)
    
    return t_arr, sol_arr
    



def calculate_energy(sol_arr, L_param=L_CONST, m_param=M_CONST, g_param=G_CONST):
    """
    Calculates the total energy (kinetic + potential) of the double pendulum system.
    """
    theta1 = sol_arr[:, 0]
    omega1 = sol_arr[:, 1]
    theta2 = sol_arr[:, 2]
    omega2 = sol_arr[:, 3]

    # Potential Energy (V)
    V = -m_param * g_param * L_param * (2 * np.cos(theta1) + np.cos(theta2))

    # Kinetic Energy (T)
    T = m_param * L_param**2 * (omega1**2 + 0.5 * omega2**2 + omega1 * omega2 * np.cos(theta1 - theta2))
    
    return T + V

def analyze_energy(t_arr, energy):
    """Detailed analysis of energy conservation"""
    # Calculate energy statistics
    energy_min = np.min(energy)
    energy_max = np.max(energy)
    energy_change = energy_max - energy_min
    
    # Calculate energy change rate
    energy_diff = np.diff(energy)
    max_diff = np.max(np.abs(energy_diff))
    
    print("\n=== Energy Conservation Analysis ===")
    print(f"Initial Energy: {energy[0]:.8f} J")
    print(f"Minimum Energy: {energy_min:.8f} J")
    print(f"Maximum Energy: {energy_max:.8f} J")
    print(f"Maximum Absolute Change: {energy_change:.3e} J")
    print(f"Maximum Instantaneous Change: {max_diff:.3e} J/step")
    
    # Create detailed energy plots
    plt.figure(figsize=(12, 8))
    
    # Total energy plot
    plt.subplot(2, 1, 1)
    plt.plot(t_arr, energy, label='Total Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (Joules)')
    plt.title('Double Pendulum: Total Energy Evolution')
    plt.grid(True)
    plt.legend()
    
    # Energy change plot
    plt.subplot(2, 1, 2)
    plt.plot(t_arr[1:], np.abs(energy_diff), 'r-')
    plt.yscale('log')
    plt.xlabel('Time (s)')
    plt.ylabel('|dE/dt| (Joules/step)')
    plt.title('Energy Change Rate (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    
    return energy_change

# --- 可选任务: 动画 --- (自动评分器不评分，但有助于可视化)
def animate_double_pendulum(t_arr, sol_arr, L_param=L_CONST, skip_frames=10):
    """
    Creates an animation of the double pendulum.
    """
    theta1_all = sol_arr[:, 0]
    theta2_all = sol_arr[:, 2]

    # Select frames for animation
    theta1_anim = theta1_all[::skip_frames]
    theta2_anim = theta2_all[::skip_frames]
    t_anim = t_arr[::skip_frames]

    # Cartesian coordinates
    x1 = L_param * np.sin(theta1_anim)
    y1 = -L_param * np.cos(theta1_anim)
    x2 = x1 + L_param * np.sin(theta2_anim)
    y2 = y1 - L_param * np.cos(theta2_anim)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, autoscale_on=False, 
                         xlim=(-2*L_param-0.1, 2*L_param+0.1), 
                         ylim=(-2*L_param-0.1, 0.1))
    ax.set_aspect('equal')
    ax.grid()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Double Pendulum Animation')
    
    # Create line for pendulum
    line, = ax.plot([], [], 'o-', lw=2, markersize=8, color='red')
    
    # Create trace if requested
    trace, = ax.plot([], [], ',-', lw=1, alpha=0.5, color='blue') if show_trace else (None,)
    
    # Text elements
    time_template = 'Time = %.1f s'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    energy_template = 'Energy Change = %.2e J'
    energy_text = ax.text(0.05, 0.85, '', transform=ax.transAxes) if show_trace else None
    
    # Store trajectory
    x2_history = []
    y2_history = []
    
    # Calculate initial energy for reference
    energy0 = calculate_energy(sol_arr[0:1])[0]

    def init():
        line.set_data([], [])
        if show_trace:
            trace.set_data([], [])
        time_text.set_text('')
        if energy_text:
            energy_text.set_text('')
        return (line, trace, time_text, energy_text) if show_trace else (line, time_text)

    def animate(i):
        # Update pendulum line
        thisx = [0, x1[i], x2[i]]
        thisy = [0, y1[i], y2[i]]
        line.set_data(thisx, thisy)
        
        # Update trace
        if show_trace:
            x2_history.append(x2[i])
            y2_history.append(y2[i])
            trace.set_data(x2_history, y2_history)
        
        # Update time text
        time_text.set_text(time_template % t_anim[i])
        
        # Update energy text
        if energy_text:
            idx = i * skip_frames
            energy_current = calculate_energy(sol_arr[idx:idx+1])[0]
            energy_change = abs(energy_current - energy0)
            energy_text.set_text(energy_template % energy_change)
        
        return (line, trace, time_text, energy_text) if show_trace else (line, time_text)

    ani = animation.FuncAnimation(
        fig, animate, frames=len(t_anim),
        interval=25, blit=True, init_func=init
    )
    
    return ani
    
    

if __name__ == '__main__':
    # 本节用于您的测试和可视化。
    # 自动评分器将导入您的函数并分别测试它们。

    print("运行学生脚本进行测试...")

    # 初始条件 (角度单位为弧度)
    initial_conditions_rad_student = {
        'theta1': np.pi/2,  # 90 度
        'omega1': 0.0,
        'theta2': np.pi/2,  # 90 度
        'omega2': 0.0
    }
    t_start_student = 0
    t_end_student = 10 # 使用较短时间进行快速测试，例如 10 秒或 20 秒
                       # 题目要求 100 秒，但这对于重复测试可能较慢。
    t_points_student = 1000 # 模拟的点数。对于 100 秒，题目建议 1000-2000 点。
                            # 为了能量守恒，可能需要更多的点或更严格的 rtol/atol。

    # --- 测试 solve_double_pendulum --- 
    try:
        print(f"\n尝试使用学生函数求解 ODE (时间从 {t_start_student}s 到 {t_end_student}s)...")
        t_sol_student, sol_student = solve_double_pendulum(
            initial_conditions_rad_student, 
            (t_start_student, t_end_student), 
            t_points_student
        )
        print("solve_double_pendulum 已执行。")
        print(f"t_sol_student 的形状: {t_sol_student.shape}")
        print(f"sol_student 的形状: {sol_student.shape}")

        # --- 测试 calculate_energy ---
        try:
            print("\n尝试使用学生函数计算能量...")
            energy_student = calculate_energy(sol_student)
            print("calculate_energy 已执行。")
            print(f"energy_student 的形状: {energy_student.shape}")
            
            # 为学生测试绘制能量图
            plt.figure(figsize=(10, 5))
            plt.plot(t_sol_student, energy_student, label='学生计算的总能量')
            plt.xlabel('时间 (s)')
            plt.ylabel('能量 (焦耳)')
            plt.title('学生：总能量 vs. 时间')
            plt.grid(True)
            plt.legend()
            
            initial_energy_student = energy_student[0]
            energy_variation_student = np.max(energy_student) - np.min(energy_student)
            print(f"学生计算的初始能量: {initial_energy_student:.7f} J")
            print(f"学生计算的最大能量变化: {energy_variation_student:.3e} J")
            if energy_variation_student < 1e-5:
                print("学生能量守恒目标 (< 1e-5 J) 在此运行中已达到。")
            else:
                print(f"学生能量守恒目标未达到。变化量: {energy_variation_student:.2e} J。请考虑在 odeint 中增加 t_points 或调整 rtol/atol。")
            plt.show()

        except NotImplementedError as e:
            print(f"calculate_energy 未实现: {e}")
        except Exception as e:
            print(f"calculate_energy 或绘图时出错: {e}")

        # --- 测试 animate_double_pendulum (可选) ---
        run_student_animation = False # 设置为 True 以测试动画
        if run_student_animation:
            try:
                print("\n尝试使用学生函数创建动画...")
                # 调整 skip_frames: t_points_student / (期望的fps * 动画时长_秒)
                # 例如: 1000 点 / (25fps * 10秒动画_对应10秒真实时间) = 4。暂时使用固定的跳帧数。
                anim_obj_student = animate_double_pendulum(t_sol_student, sol_student, skip_frames=max(1, t_points_student // 200))
                print("animate_double_pendulum 已执行。")
                plt.show() # 显示动画
            except NotImplementedError as e:
                print(f"animate_double_pendulum 未实现: {e}")
            except Exception as e:
                print(f"animate_double_pendulum 执行出错: {e}")
        else:
            print("\n学生动画测试已跳过。")

    except NotImplementedError as e:
        print(f"solve_double_pendulum 或其依赖的 derivatives 未实现: {e}")
    except Exception as e:
        print(f"学生脚本执行期间发生错误: {e}")

    print("\n学生脚本测试完成。")

"""
给学生的提示:
1.  首先实现 `derivatives`。如果可能，用简单的输入测试它，尽管它主要通过 `odeint` 进行测试。
2.  然后实现 `solve_double_pendulum`。确保正确调用 `odeint`。
3.  接下来实现 `calculate_energy`。这对于验证模拟的正确性至关重要。
4.  绘制能量图。如果能量不守恒 (或显著漂移)，请重新检查 `derivatives` 中的方程是否有误，
    或在 `solve_double_pendulum` 的 `odeint` 调用中调整 `rtol` 和 `atol`，或增加 `t_points`。
    目标是在 100 秒内能量变化 < 1e-5 J。
5.  动画是可选的，但强烈建议用于理解物理过程。它不会被自动评分。
6.  在开发过程中，使用 `if __name__ == '__main__':` 块来测试您的函数。
    `NotImplementedError` 将在第一个未实现的函数处停止执行。
"""
