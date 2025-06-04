#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目1：二阶常微分方程边值问题数值解法 - 学生代码模板
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.linalg import solve

# ============================================================================
# 方法1：有限差分法 (Finite Difference Method)
# ============================================================================

def solve_bvp_finite_difference(n):
    """
    使用有限差分法求解二阶常微分方程边值问题。
    
    方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2
    边界条件：y(0) = 0, y(5) = 3
    
    Args:
        n (int): 内部网格点数量
    
    Returns:
        tuple: (x_grid, y_solution)
            x_grid (np.ndarray): 包含边界点的完整网格
            y_solution (np.ndarray): 对应的解值
    """
    # 定义区间和步长
    a, b = 0, 5
    h = (b - a) / (n + 1)
    
    # 创建网格点 (包含边界点)
    x_grid = np.linspace(a, b, n + 2)
    
    # 初始化系数矩阵A和右端向量b
    A = np.zeros((n, n))
    b_vec = np.zeros(n)
    
    # 填充系数矩阵和右端向量
    for i in range(n):
        x_i = x_grid[i+1]  # 内部点对应的x值
        # 主对角线元素
        A[i, i] = -2/(h**2) + np.exp(x_i)
        
        # 次对角线元素 (i-1)
        if i > 0:
            A[i, i-1] = 1/(h**2) - np.sin(x_i)/(2*h)
        
        # 超对角线元素 (i+1)
        if i < n-1:
            A[i, i+1] = 1/(h**2) + np.sin(x_i)/(2*h)
        
        # 右端向量
        b_vec[i] = x_i**2
        
        # 处理边界条件对右端向量的影响
        if i == 0:
            b_vec[i] -= (1/(h**2) - np.sin(x_i)/(2*h)) * 0  # y(0)=0
        if i == n-1:
            b_vec[i] -= (1/(h**2) + np.sin(x_i)/(2*h)) * 3  # y(5)=3
    
    # 求解线性方程组
    y_internal = solve(A, b_vec)
    
    # 组合边界点和内部点
    y_solution = np.zeros(n + 2)
    y_solution[0] = 0  # y(0)=0
    y_solution[-1] = 3  # y(5)=3
    y_solution[1:-1] = y_internal
    
    return x_grid, y_solution

# ============================================================================
# 方法2：scipy.integrate.solve_bvp 方法
# ============================================================================

def ode_system_for_solve_bvp(x, y):
    """
    为 scipy.integrate.solve_bvp 定义ODE系统。
    
    系统方程：
    dy[0]/dx = y[1]
    dy[1]/dx = -sin(x) * y[1] - exp(x) * y[0] + x^2
    """
    dydx = np.zeros_like(y)
    dydx[0] = y[1]  # dy/dx = y'
    dydx[1] = -np.sin(x) * y[1] - np.exp(x) * y[0] + x**2  # dy'/dx = -sin(x)y' - e^x y + x^2
    return dydx

def boundary_conditions_for_solve_bvp(ya, yb):
    """
    为 scipy.integrate.solve_bvp 定义边界条件。
    
    边界条件：
    y(0) = 0 -> ya[0] = 0
    y(5) = 3 -> yb[0] = 3
    """
    return np.array([ya[0], yb[0] - 3])

def solve_bvp_scipy(n_initial_points=11):
    """
    使用 scipy.integrate.solve_bvp 求解BVP。
    
    Args:
        n_initial_points (int): 初始网格点数
    
    Returns:
        tuple: (x_solution, y_solution)
            x_solution (np.ndarray): 解的 x 坐标数组
            y_solution (np.ndarray): 解的 y 坐标数组
    """
    # 创建初始网格
    x_initial = np.linspace(0, 5, n_initial_points)
    
    # 创建初始猜测 (线性函数满足边界条件)
    y_initial = np.zeros((2, n_initial_points))
    y_initial[0] = 3 * x_initial / 5  # y(x) ≈ 3x/5
    
    # 调用solve_bvp
    sol = solve_bvp(ode_system_for_solve_bvp, boundary_conditions_for_solve_bvp, x_initial, y_initial)
    
    # 检查求解是否成功
    if not sol.success:
        raise RuntimeError(f"求解失败: {sol.message}")
    
    # 在更密集的网格上评估解
    x_solution = np.linspace(0, 5, 100)
    y_solution = sol.sol(x_solution)[0]
    
    return x_solution, y_solution

# ============================================================================
# 主程序：测试和比较两种方法
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("二阶常微分方程边值问题数值解法比较")
    print("方程：y''(x) + sin(x) * y'(x) + exp(x) * y(x) = x^2")
    print("边界条件：y(0) = 0, y(5) = 3")
    print("=" * 60)
    
    # 设置参数
    n_points = 50  # 有限差分法的内部网格点数
    
    try:
        # 方法1：有限差分法
        print("\n1. 有限差分法求解...")
        x_fd, y_fd = solve_bvp_finite_difference(n_points)
        print(f"   网格点数：{len(x_fd)}")
        print(f"   y(0) = {y_fd[0]:.6f}, y(5) = {y_fd[-1]:.6f}")
        
    except Exception as e:
        print(f"   有限差分法求解错误: {e}")
        x_fd, y_fd = None, None
    
    try:
        # 方法2：scipy.integrate.solve_bvp
        print("\n2. scipy.integrate.solve_bvp 求解...")
        x_scipy, y_scipy = solve_bvp_scipy()
        print(f"   网格点数：{len(x_scipy)}")
        print(f"   y(0) = {y_scipy[0]:.6f}, y(5) = {y_scipy[-1]:.6f}")
        
    except Exception as e:
        print(f"   solve_bvp 方法求解错误: {e}")
        x_scipy, y_scipy = None, None
    
    # 绘图比较
    plt.figure(figsize=(12, 8))
    
    # 子图1：解的比较
    plt.subplot(2, 1, 1)
    if x_fd is not None and y_fd is not None:
        plt.plot(x_fd, y_fd, 'b-o', markersize=3, label='Finite Difference Method', linewidth=2)
    if x_scipy is not None and y_scipy is not None:
        plt.plot(x_scipy, y_scipy, 'r--', label='scipy.integrate.solve_bvp', linewidth=2)
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Comparison of Numerical Solutions for BVP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2：解的差异（如果两种方法都实现了）
    plt.subplot(2, 1, 2)
    if (x_fd is not None and y_fd is not None and 
        x_scipy is not None and y_scipy is not None):
        
        # 将 scipy 解插值到有限差分网格上进行比较
        y_scipy_interp = np.interp(x_fd, x_scipy, y_scipy)
        difference = np.abs(y_fd - y_scipy_interp)
        
        plt.semilogy(x_fd, difference, 'g-', linewidth=2, label='|Finite Diff - solve_bvp|')
        plt.xlabel('x')
        plt.ylabel('Absolute Difference')
        plt.title('Absolute Difference Between Methods')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 数值比较
        max_diff = np.max(difference)
        mean_diff = np.mean(difference)
        print(f"\n数值比较：")
        print(f"   最大绝对误差：{max_diff:.2e}")
        print(f"   平均绝对误差：{mean_diff:.2e}")
    else:
        plt.text(0.5, 0.5, '需要两种方法都实现才能比较', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title('Difference Plot (Not Available)')
    
    plt.tight_layout()
    plt.savefig('bvp_comparison.png', dpi=300)
    plt.show()
    
    # 不同网格点数的精度比较
    plt.figure(figsize=(10, 6))
    n_points_list = [10, 20, 50, 100]
    
    # 使用高精度解作为参考
    _, y_ref = solve_bvp_scipy(n_initial_points=200)
    x_ref = np.linspace(0, 5, 100)
    
    for n in n_points_list:
        try:
            x_fd, y_fd = solve_bvp_finite_difference(n)
            y_ref_interp = np.interp(x_fd, x_ref, y_ref)
            error = np.abs(y_fd - y_ref_interp)
            plt.semilogy(x_fd, error, label=f'n={n}')
        except:
            continue
    
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Finite Difference Method Error vs Number of Points')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('fdm_error_analysis.png', dpi=300)
    plt.show()
    
    print("\n=" * 60)
    print("实验完成！")
    print("=" * 60)
